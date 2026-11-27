#include <metal_stdlib>
using namespace metal;

// KEP1 rows from neb_engines::table, padded to a four-byte boundary by the host.
// Conv layers have 40 int, 640 short, 40 short and 640 short entries.
// The two dense layers store (32+768) and (2+32) signed byte factors.
constant uint factor_offset = 2800;
constant uint factor_count = 834;
constant uint row_bytes = 3636;
struct Factors { char4 v[4]; };

// Dense factors are signed four-bit values. A product is at most 64 in
// magnitude, and the sum of sixteen fits exactly in a signed short.
#if SHORT_MAC
using Mac4 = short4;
#else
using Mac4 = int4;
#endif

inline uint fold(int grad, uint layer, uint weight) {
    return uint(clamp(grad, -256, 256)) ^ (7919u * layer + weight);
}

inline uint triangular(uint n) { return n*(n+1)/2; }
// Triple-prefix domains are bounded by the host before this multiplication.
inline uint tetrahedral(uint n) { return n*(n+1)*(n+2)/6; }

// A group shares the 16 samples selected by one nonce. Neighboring threads
// read consecutive weights of a sample. Each thread owns one digest lane.
kernel void kep1_digest(device const uchar *rows [[buffer(0)]],
                        device const uint *indices [[buffer(1)]],
                        device uint *digests [[buffer(2)]],
                        #if SINGLE_CACHE
                        device const uint *singles [[buffer(3)]],
                        #endif
                        #if COMBO_READ
                        device const uint *ranks [[buffer(4)]],
                        device const uint *combo_digests [[buffer(5)]],
                        #endif
                        #if COMBO_READ || COMBO_BUILD
                        constant uint *lookup_args [[buffer(6)]],
                        #endif
                        #if REUSE_PAIRS
                        device const uint *old_ranks [[buffer(7)]],
                        device const uint *old_values [[buffer(8)]],
                        constant uint &old_ready [[buffer(9)]],
                        #endif
                        uint tid [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]],
                        uint width [[threads_per_threadgroup]]) {
    #if PLANAR
    threadgroup char4 planes[4][factor_count];
    #define FACTOR(e,w) planes[w][e]
    #else
    threadgroup Factors factors[factor_count];
    #define FACTOR(e,w) factors[e].v[w]
    #endif
    threadgroup uint partial[GROUP_WIDTH/32][8];
    #if COMBO_BUILD
    uint combo = lookup_args[0] + group;
    uint component[3];
    #if COMBO_BUILD == 3
    uint lo = 0, hi = lookup_args[1];
    while (hi-lo > 1) {
        uint mid = (lo+hi)/2;
        if (tetrahedral(mid) <= combo) lo = mid; else hi = mid;
    }
    component[2] = indices[lo];
    combo -= tetrahedral(lo);
    #endif
    uint b = uint((sqrt(float(8u*combo+1u))-1.0f)*0.5f);
    while (triangular(b) > combo) --b;
    while (triangular(b+1) <= combo) ++b;
    #if REUSE_PAIRS
    uint old_a=old_ranks[combo-triangular(b)], old_b=old_ranks[b];
    if (old_a!=0xffffffffu && old_b!=0xffffffffu) {
        uint old_index=triangular(max(old_a,old_b))+min(old_a,old_b);
        if (old_index<old_ready) {
            if(tid<8) digests[group*8+tid]=old_values[old_index*8+tid];
            return;
        }
    }
    #endif
    component[0] = indices[combo-triangular(b)];
    component[1] = indices[b];
    constexpr uint samples = COMBO_BUILD;
    #define ROW(b) component[b]
    #elif COMPACT_ROWS
    device const uint *idx = indices + group * 16;
    uint samples = idx[0] >> 24;
    #define ROW(b) (idx[b] & 0x00ffffffu)
    #else
    device const uint *idx = indices + group * 16;
    constexpr uint samples = 16;
    #define ROW(b) idx[b]
    #endif
    #if COMBO_READ
    if (samples <= 1) {
        uint entry = samples == 0 ? 0 : ROW(0)+1;
        if (tid < 8) digests[group*8+tid] = singles[9*entry+1+tid];
        return;
    }
    if (samples == 2 || (COMBO_READ == 3 && samples == 3)) {
        uint a = ranks[ROW(0)], b = ranks[ROW(1)];
        uint c = samples == 2 ? 0 : ranks[ROW(2)];
        if (samples == 2 || max(max(a,b),c) < lookup_args[1]) {
            uint result;
            if (samples == 2) result = triangular(max(a,b)) + min(a,b);
            else {
                uint high = max(max(a,b),c), low = min(min(a,b),c);
                result = lookup_args[2] + tetrahedral(high) + triangular(a+b+c-high-low) + low;
            }
            if (tid < 8) digests[group*8+tid] = combo_digests[8*result+tid];
            return;
        }
    }
    #endif
    #if SINGLE_CACHE && !COMBO_READ
    // Every SIMD group classifies the same 16 rows. With at most one nonzero
    // gradient, clipping and folding have already been done for this exact sum.
    uint lane = tid % 32;
    uint entry = lane < samples ? ROW(lane)+1 : 0;
    uint live = singles[9*entry];
    uint count_live = simd_sum(live);
    if (count_live <= 1) {
        uint selected = simd_sum(live * entry);
        if (tid < 8) digests[group*8+tid] = singles[9*selected+1+tid];
        return;
    }
    #endif
    #if ROW_CACHE
    threadgroup uint offsets[16];
    if (tid < samples) offsets[tid] = ROW(tid) * row_bytes;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #define OFFSET(b) offsets[b]
    #else
    #define OFFSET(b) (ROW(b) * row_bytes)
    #endif
    for (uint e = tid; e < factor_count; e += width) {
        for (uint w = 0; w < 4; w++) {
            char4 v;
            for (uint b = 0; b < 4; b++)
                v[b] = 4*w+b < samples ? as_type<char>(rows[OFFSET(4*w+b) + factor_offset + e]) : 0;
            FACTOR(e,w) = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sum = 0;
    for (uint f = tid; f < 1360; f += width) {
        int grad = 0;
        for (uint b = 0; b < samples; b++) {
            device const uchar *row = rows + OFFSET(b);
            grad += f < 40 ? ((device const int *)row)[f] : ((device const short *)(row+160))[f-40];
        }
        if (f < 40) sum += fold(grad, 1, f);
        else if (f < 680) sum += fold(grad, 2, f-40);
        else if (f < 720) sum += fold(grad, 3, f-680);
        else sum += fold(grad, 4, f-720);
    }
    #if SALT_BASE
    // K2-style zero-gradient digest, summed once for every dense weight
    // assigned to this thread. Active outputs below add their exact delta.
    uint terms = 24576 / width;
    sum += terms*(7919u*5 + tid) + width*terms*(terms-1)/2;
    #endif
    for (uint o = 0; o < 32; o++) {
        Factors err;
        for (uint w = 0; w < 4; w++) err.v[w] = FACTOR(o,w);
        // An output with zero error in all samples has zero gradient.
        // Its digest entries still contribute their weight-dependent salts.
        if (!any((err.v[0] | err.v[1] | err.v[2] | err.v[3]) != 0)) {
            #if !SALT_BASE
            uint steps = 768 / width;
            sum += steps * (7919u*5 + o*768 + tid) + width*steps*(steps-1)/2;
            #endif
        } else {
            for (uint q = tid; q < 768; q += width) {
                Mac4 products = 0;
                for (uint w = 0; w < 4; w++)
                    #if SHORT_DOT
                    if (4*w < samples)
                    #endif
                    products += Mac4(err.v[w]) * Mac4(FACTOR(32+q,w));
                sum += fold(products.x + products.y + products.z + products.w, 5, o*768+q);
                #if SALT_BASE
                sum -= 7919u*5 + o*768 + q;
                #endif
            }
        }
    }
    if (tid < 64) {
        Mac4 products = 0;
        for (uint w = 0; w < 4; w++)
            products += Mac4(FACTOR(800+tid/32,w)) * Mac4(FACTOR(802+tid%32,w));
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
