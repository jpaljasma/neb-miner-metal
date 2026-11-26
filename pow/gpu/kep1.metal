#include <metal_stdlib>
using namespace metal;

// KEP1 rows from neb_engines::table, padded to a four-byte boundary by the host.
// Conv layers have 40 int, 640 short, 40 short and 640 short entries.
// The two dense layers store (32+768) and (2+32) signed byte factors.
constant uint factor_offset = 2800;
constant uint factor_count = 834;
constant uint row_bytes = 3636;
struct Factors { char4 v[4]; };

inline uint fold(int grad, uint layer, uint weight) {
    return uint(clamp(grad, -256, 256)) ^ (7919u * layer + weight);
}

// A group shares the 16 samples selected by one nonce. Neighboring threads
// read consecutive weights of a sample. Each thread owns one digest lane.
kernel void kep1_digest(device const uchar *rows [[buffer(0)]],
                        device const uint *indices [[buffer(1)]],
                        device uint *digests [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]],
                        uint width [[threads_per_threadgroup]]) {
    threadgroup Factors factors[factor_count];
    threadgroup uint partial[4][8]; // four 32-lane SIMD groups per dispatch group
    device const uint *idx = indices + group * 16;
    for (uint e = tid; e < factor_count; e += width) {
        for (uint w = 0; w < 4; w++) {
            char4 v;
            for (uint b = 0; b < 4; b++)
                v[b] = as_type<char>(rows[idx[4*w+b] * row_bytes + factor_offset + e]);
            factors[e].v[w] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sum = 0;
    for (uint f = tid; f < 1360; f += width) {
        int grad = 0;
        for (uint b = 0; b < 16; b++) {
            device const uchar *row = rows + idx[b] * row_bytes;
            grad += f < 40 ? ((device const int *)row)[f] : ((device const short *)(row+160))[f-40];
        }
        if (f < 40) sum += fold(grad, 1, f);
        else if (f < 680) sum += fold(grad, 2, f-40);
        else if (f < 720) sum += fold(grad, 3, f-680);
        else sum += fold(grad, 4, f-720);
    }
    for (uint o = 0; o < 32; o++) {
        Factors err = factors[o];
        // An output with zero error in all samples has zero gradient.
        // Its digest entries still contribute their weight-dependent salts.
        if (!any((err.v[0] | err.v[1] | err.v[2] | err.v[3]) != 0)) {
            uint steps = 768 / width;
            sum += steps * (7919u*5 + o*768 + tid) + width*steps*(steps-1)/2;
        } else {
            for (uint q = tid; q < 768; q += width) {
                int4 products = 0;
                for (uint w = 0; w < 4; w++)
                    products += int4(err.v[w]) * int4(factors[32+q].v[w]);
                sum += fold(products.x + products.y + products.z + products.w, 5, o*768+q);
            }
        }
    }
    if (tid < 64) {
        int4 products = 0;
        for (uint w = 0; w < 4; w++)
            products += int4(factors[800+tid/32].v[w]) * int4(factors[802+tid%32].v[w]);
        sum += fold(products.x + products.y + products.z + products.w, 6, tid);
    }
    // The M1 Max has 32 lanes per SIMD group. Fold the four copies of each
    // digest lane, then combine the SIMD groups in the threadgroup.
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 16);
    if ((tid % 32) < 8) partial[tid/32][tid%8] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        uint total = 0;
        for (uint i = 0; i < width/32; i++) total += partial[i][tid];
        digests[8*group+tid] = total;
    }
}
