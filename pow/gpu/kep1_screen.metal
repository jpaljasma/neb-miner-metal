#include <metal_stdlib>
using namespace metal;

// BLAKE3 compress and ChaCha8 block zero, in four parallel columns. Message
// words and all digest/target words are little endian on Apple silicon.
constant uint4 iv_lo = uint4(0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a);
constant uint4 iv_hi = uint4(0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19);
constant uchar schedule[7][16] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
{2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
{3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
{10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
{12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
{9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
{11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
struct Hash { uint4 lo, hi; };
#if COMPACT_TICKETS
struct PendingTicket { Hash key; uint nonce, digest_word; };
#endif
inline uint4 rot(uint4 x, uint k) { return (x<<k) | (x>>(32-k)); }
inline void mix(thread uint4 &a, thread uint4 &b, thread uint4 &c, thread uint4 &d, uint4 x, uint4 y) {
    a+=b+x; d=rot(d^a,16); c+=d; b=rot(b^c,20);
    a+=b+y; d=rot(d^a,24); c+=d; b=rot(b^c,25);
}
template<bool cached_header = false, typename Params>
inline Hash compress(Hash cv, thread const uint *m, uint len, uint flags, Params args) {
    uint4 a=cv.lo,b=cv.hi,c=iv_lo,d=uint4(0,0,len,flags);
    #pragma clang loop unroll(full)
    for(uint r=0;r<7;r++) {
        constant const uchar *s=schedule[r];
        if(cached_header && r==0) {
            a=uint4(args[24],args[25],args[26],args[27]);
            b=uint4(args[28],args[29],args[30],args[31]);
            c=uint4(args[32],args[33],args[34],args[35]);
            d=uint4(args[36],args[37],args[38],args[39]);
            a.y+=b.y+m[3]; d.y=rotate(d.y^a.y,24u); c.y+=d.y; b.y=rotate(b.y^c.y,25u);
        } else {
            mix(a,b,c,d,uint4(m[s[0]],m[s[2]],m[s[4]],m[s[6]]),uint4(m[s[1]],m[s[3]],m[s[5]],m[s[7]]));
        }
        b=b.yzwx;c=c.zwxy;d=d.wxyz;
        mix(a,b,c,d,uint4(m[s[8]],m[s[10]],m[s[12]],m[s[14]]),uint4(m[s[9]],m[s[11]],m[s[13]],m[s[15]]));
        b=b.wxyz;c=c.zwxy;d=d.yzwx;
    }
    return {a^c,b^d};
}
inline void qr(thread uint4 &a, thread uint4 &b, thread uint4 &c, thread uint4 &d) {
    a+=b;d=rot(d^a,16);c+=d;b=rot(b^c,12);
    a+=b;d=rot(d^a,8);c+=d;b=rot(b^c,7);
}
#if SCALAR_CHACHA
inline void qr_word(thread uint &a, thread uint &b, thread uint &c, thread uint &d) {
    a+=b; d=rotate(d^a,16u); c+=d; b=rotate(b^c,12u);
    a+=b; d=rotate(d^a,8u); c+=d; b=rotate(b^c,7u);
}
#endif
inline uint triangular_index(uint b) { return b*(b+1)/2; }
inline uint triple_index(uint c) { return c*(c+1)*(c+2)/6; }

#if RANK_LAYOUT == 2
typedef ushort RankWord;
constant uint live_tag = 1u << 15;
#else
typedef uint RankWord;
constant uint live_tag = 1u << 27;
#endif

// args: first header chaining value [8], header tail [3], starting nonce,
// dataset size, ready pairs, triple prefix, complete pair count, target [8],
// and the four cached BLAKE3 column vectors [16] when HEADER_COLUMNS is enabled.
// ranks: zero-gradient rows use UINT_MAX; live rows use contiguous ranks.
kernel void lookup_screen(constant const uint *input_args [[buffer(0)]],
                          device const RankWord *ranks [[buffer(1)]],
                          device const uint *single [[buffer(2)]],
                          device const uint *combos [[buffer(3)]],
                          device atomic_uint *counts [[buffer(4)]],
                          device uint *hits [[buffer(5)]],
                          device uint *records [[buffer(6)]],
                          #if RANK_LAYOUT && PAIRS_ONLY
                          device const uint *live_rows [[buffer(7)]],
                          #endif
                          uint i [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]) {
    #if GROUP_ARGS
    constexpr uint words_per_group = HEADER_COLUMNS ? 40u : 24u;
    threadgroup uint args[words_per_group];
    for(uint j=tid;j<words_per_group;j+=group_size) args[j]=input_args[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #else
    constant const uint *args=input_args;
    #endif
    uint block[16]={};
    for(uint j=0;j<3;j++) block[j]=args[8+j];
    uint nonce=args[11]+i;
    block[3]=nonce;
    Hash cv={uint4(args[0],args[1],args[2],args[3]),uint4(args[4],args[5],args[6],args[7])};
    Hash key=compress<HEADER_COLUMNS>(cv,block,16,10,args);
    uint4 init=uint4(0x61707865,0x3320646e,0x79622d32,0x6b206574);
    #if SCALAR_CHACHA
    uint x[16]={init.x,init.y,init.z,init.w,key.lo.x,key.lo.y,key.lo.z,key.lo.w,
                key.hi.x,key.hi.y,key.hi.z,key.hi.w,0,0,0,0};
    #pragma clang loop unroll(full)
    for(uint r=0;r<4;r++){
        qr_word(x[0],x[4],x[8],x[12]); qr_word(x[1],x[5],x[9],x[13]);
        qr_word(x[2],x[6],x[10],x[14]); qr_word(x[3],x[7],x[11],x[15]);
        qr_word(x[0],x[5],x[10],x[15]); qr_word(x[1],x[6],x[11],x[12]);
        qr_word(x[2],x[7],x[8],x[13]); qr_word(x[3],x[4],x[9],x[14]);
    }
    uint4 words[4]={uint4(x[0],x[1],x[2],x[3])+init,uint4(x[4],x[5],x[6],x[7])+key.lo,
                    uint4(x[8],x[9],x[10],x[11])+key.hi,uint4(x[12],x[13],x[14],x[15])};
    #else
    uint4 a=init,b=key.lo,c=key.hi,d=0;
    #pragma clang loop unroll(full)
    for(uint r=0;r<4;r++){
        qr(a,b,c,d); b=b.yzwx;c=c.zwxy;d=d.wxyz;
        qr(a,b,c,d); b=b.wxyz;c=c.zwxy;d=d.yzwx;
    }
    uint4 words[4]={a+init,b+key.lo,c+key.hi,d};
    #endif
    uint live=0, row=0, ids[3]={};
    #if RANK_LAYOUT && PAIRS_ONLY
    // Idle rows contribute zero. Below three tags, the low bits hold the sum of
    // at most two ranks; subtracting the largest also handles a repeated row.
    uint rank_sum=0, largest=0;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        uint4 sample=mulhi(words[j],uint4(args[12]));
        uint4 tagged=uint4(ranks[sample.x],ranks[sample.y],ranks[sample.z],ranks[sample.w]);
        rank_sum+=tagged.x+tagged.y+tagged.z+tagged.w;
        if(rank_sum>=3u*live_tag) break;
        largest=max(largest,max(max(tagged.x,tagged.y),max(tagged.z,tagged.w)));
    }
    live=rank_sum/live_tag;
    if(live==1) row=live_rows[largest-live_tag];
    if(live==2) {
        ids[1]=largest-live_tag;
        ids[0]=(rank_sum & (live_tag-1u))-ids[1];
    }
    #elif RANK_QUADS && PAIRS_ONLY
    ids[0]=0xffffffffu;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;j++) {
        uint4 sample=mulhi(words[j],uint4(args[12]));
        uint4 rank=uint4(ranks[sample.x],ranks[sample.y],ranks[sample.z],ranks[sample.w]);
        bool4 active=rank!=0xffffffffu;
        uint4 used=select(uint4(0),uint4(1),active);
        live+=used.x+used.y+used.z+used.w;
        if(live>2) break;
        ids[0]=min(ids[0],min(min(rank.x,rank.y),min(rank.z,rank.w)));
        uint4 valid=select(uint4(0),rank,active);
        ids[1]=max(ids[1],max(max(valid.x,valid.y),max(valid.z,valid.w)));
        sample=select(uint4(0),sample,active);
        row=max(row,max(max(sample.x,sample.y),max(sample.z,sample.w)));
    }
    #else
    for(uint j=0;j<16;j++) {
        uint id=mulhi(words[j/4][j%4],args[12]);
        uint rank=ranks[id];
        #if RANK_LAYOUT
        rank=rank ? rank-live_tag : 0xffffffffu;
        #endif
        if(rank!=0xffffffffu) {
            #if PAIRS_ONLY
            if(live==2){ live=3; break; }
            #endif
            if(live<3) ids[live]=rank;
            live++; row=id;
            if(live>3) break;
        }
    }
    #endif
    device const uint *digest=single+1;
    bool eligible=live<=1;
    if(live==1) digest=single+9*(row+1)+1;
    if(live==2) {
        uint entry=triangular_index(max(ids[0],ids[1]))+min(ids[0],ids[1]);
        eligible=entry<args[13];
        if(eligible) digest=combos+8*entry;
    }
    #if !PAIRS_ONLY
    else if(live==3) {
        uint high=max(max(ids[0],ids[1]),ids[2]);
        eligible=high<args[14];
        if(eligible) {
            uint low=min(min(ids[0],ids[1]),ids[2]);
            uint middle=ids[0]+ids[1]+ids[2]-low-high;
            digest=combos+8*(args[15]+triple_index(high)+triangular_index(middle)+low);
        }
    }
    #endif
    uint evaluated=simd_sum(uint(eligible));
    #if COMPACT_TICKETS && !CHECK_RECORDS
    // The host dispatches 32-lane SIMD groups. Pack eligible tickets
    // together so the final hash runs on occupied lanes.
    threadgroup PendingTicket pending[SCREEN_GROUP_THREADS];
    threadgroup uint group_counts[(SCREEN_GROUP_THREADS+31)/32];
    uint group=tid/32;
    uint rank_in_group=simd_prefix_exclusive_sum(uint(eligible));
    if(lane==0) group_counts[group]=evaluated;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before=0,total=0;
    for(uint g=0;g<(group_size+31)/32;g++) {
        uint n=group_counts[g];
        if(g<group) before+=n;
        total+=n;
    }
    if(tid==0 && total) atomic_fetch_add_explicit(counts,total,memory_order_relaxed);
    if(eligible) {
        uint address=live<2 ? uint(digest-single) : (uint(digest-combos)|0x80000000u);
        pending[before+rank_in_group]={key,nonce,address};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid>=total) return;
    PendingTicket item=pending[tid];
    key=item.key;
    nonce=item.nonce;
    digest=(item.digest_word>>31 ? combos : single)+(item.digest_word&0x7fffffffu);
    #else
    if(lane==0 && evaluated) atomic_fetch_add_explicit(counts,evaluated,memory_order_relaxed);
    #if CHECK_RECORDS
    records[9*i]=uint(eligible);
    if(!eligible) for(uint j=0;j<8;j++) records[9*i+1+j]=0;
    #endif
    if(!eligible) return;
    #endif
    for(uint j=0;j<8;j++) block[j]=digest[j];
    for(uint j=0;j<4;j++) {block[8+j]=key.lo[j];block[12+j]=key.hi[j];}
    #if HIGH_WORD && !CHECK_RECORDS
    // Only the high word is live on the common path. A tie requires the remaining 224 bits.
    uint high_word=compress({iv_lo,iv_hi},block,64,11,args).hi.w;
    if(high_word!=args[23]) {
        if(high_word<args[23]) hits[atomic_fetch_add_explicit(counts+1,1u,memory_order_relaxed)]=nonce;
        return;
    }
    #endif
    Hash ticket=compress({iv_lo,iv_hi},block,64,11,args);
    bool hit=true;
    for(int j=7;j>=0;j--) {
        uint v=j>=4 ? ticket.hi[j-4] : ticket.lo[j];
        #if CHECK_RECORDS
        records[9*i+1+j]=v;
        #endif
        if(v!=args[16+j]) {hit=v<args[16+j];
        #if !CHECK_RECORDS
        break;
        #else
        // The target verdict must use the first differing high word.
        for(int k=j-1;k>=0;k--) records[9*i+1+k]=k>=4 ? ticket.hi[k-4] : ticket.lo[k];
        break;
        #endif
        }
    }
    if(hit) hits[atomic_fetch_add_explicit(counts+1,1u,memory_order_relaxed)]=nonce;
}
