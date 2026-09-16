#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

#include <immintrin.h>

using Clock = std::chrono::steady_clock;

static constexpr int QK_K = 256;
static constexpr int Q4_BYTES = 128;
static constexpr int GROUPS = 8;
static constexpr int GROUP_Q = 32;
static constexpr int ROW_BLOCK = 4;

#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
#pragma pack(pop)
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

#pragma pack(push, 1)
struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
#pragma pack(pop)
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
            mant &= 0x3ffu;
            out = sign | (uint32_t)(e + 127) << 23 | mant << 13;
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | mant << 13;
    } else {
        out = sign | (exp - 15 + 127) << 23 | mant << 13;
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct gguf_tensor_info {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct gguf_file {
    std::vector<uint8_t> bytes;
    uint64_t data_offset = 0;
    std::vector<gguf_tensor_info> tensors;
};

struct reader {
    const uint8_t *p;
    const uint8_t *end;
    explicit reader(const std::vector<uint8_t> &b) : p(b.data()), end(b.data()+b.size()) {}
    template <class T> T get() {
        if ((size_t)(end-p) < sizeof(T)) throw std::runtime_error("truncated GGUF");
        T v; std::memcpy(&v,p,sizeof(v)); p += sizeof(v); return v;
    }
    std::string str() {
        uint64_t n=get<uint64_t>();
        if ((uint64_t)(end-p) < n) throw std::runtime_error("truncated string");
        std::string s((const char*)p,(size_t)n); p += n; return s;
    }
};

static void skip_value(reader &r, uint32_t t);
static void skip_array(reader &r) {
    uint32_t et = r.get<uint32_t>();
    uint64_t n = r.get<uint64_t>();
    for (uint64_t i=0;i<n;++i) skip_value(r,et);
}
static void skip_value(reader &r, uint32_t t) {
    switch(t) {
        case 0: r.get<uint8_t>(); break;
        case 1: r.get<int8_t>(); break;
        case 2: r.get<uint16_t>(); break;
        case 3: r.get<int16_t>(); break;
        case 4: r.get<uint32_t>(); break;
        case 5: r.get<int32_t>(); break;
        case 6: r.get<float>(); break;
        case 7: r.get<uint8_t>(); break;
        case 8: (void)r.str(); break;
        case 9: skip_array(r); break;
        case 10: r.get<uint64_t>(); break;
        case 11: r.get<int64_t>(); break;
        case 12: r.get<double>(); break;
        default: throw std::runtime_error("unsupported GGUF metadata type");
    }
}

static gguf_file load_gguf(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("cannot open model");
    f.seekg(0,std::ios::end); size_t n=(size_t)f.tellg(); f.seekg(0);
    gguf_file g; g.bytes.resize(n); f.read((char*)g.bytes.data(), n);
    reader r(g.bytes);
    if(r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("not GGUF");
    uint32_t version=r.get<uint32_t>();
    if(version < 2 || version > 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t nt=r.get<uint64_t>(), nkv=r.get<uint64_t>();
    uint64_t alignment=32;
    for(uint64_t i=0;i<nkv;++i){
        std::string key=r.str(); uint32_t t=r.get<uint32_t>();
        if(key=="general.alignment" && t==4) alignment=r.get<uint32_t>();
        else skip_value(r,t);
    }
    g.tensors.resize((size_t)nt);
    for(uint64_t i=0;i<nt;++i){
        auto &ti=g.tensors[(size_t)i]; ti.name=r.str(); uint32_t nd=r.get<uint32_t>();
        ti.dims.resize(nd); for(uint32_t d=0;d<nd;++d) ti.dims[d]=r.get<uint64_t>();
        ti.type=r.get<uint32_t>(); ti.offset=r.get<uint64_t>();
    }
    uint64_t pos=(uint64_t)(r.p-g.bytes.data());
    g.data_offset=(pos + alignment - 1) / alignment * alignment;
    return g;
}

static const gguf_tensor_info &find_tensor(const gguf_file &g,const std::string &name){
    for(auto &t:g.tensors) if(t.name==name) return t;
    throw std::runtime_error("tensor not found: "+name);
}

struct cpu_ref { uint16_t group=0; uint8_t number=0; int core_type=0; };

#if defined(_WIN32)
static int core_type_for_cpu(const cpu_ref &c) {
    GROUP_AFFINITY oldga{};
    HANDLE th=GetCurrentThread();
    if(!GetThreadGroupAffinity(th,&oldga)) return 0;
    GROUP_AFFINITY ga{}; ga.Group=c.group; ga.Mask=(KAFFINITY(1)<<c.number);
    if(!SetThreadGroupAffinity(th,&ga,nullptr)) return 0;
    int regs[4]={0}; __cpuid(regs,0); int maxleaf=regs[0]; int ct=0;
    if(maxleaf>=0x1A){ __cpuidex(regs,0x1A,0); ct=(regs[0]>>24)&0xff; }
    SetThreadGroupAffinity(th,&oldga,nullptr);
    return ct;
}
static std::vector<cpu_ref> physical_cores() {
    DWORD len=0; GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&len);
    std::vector<uint8_t> buf(len);
    if(!GetLogicalProcessorInformationEx(RelationProcessorCore,(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(),&len)) throw std::runtime_error("GLPIEx failed");
    std::vector<cpu_ref> out; uint8_t *p=buf.data(),*e=p+len;
    while(p<e){ auto *x=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p; if(x->Relationship==RelationProcessorCore){ auto &pr=x->Processor; if(pr.GroupCount){ GROUP_AFFINITY gm=pr.GroupMask[0]; for(uint8_t b=0;b<64;++b) if(gm.Mask&(KAFFINITY(1)<<b)){ cpu_ref c{gm.Group,b,0}; c.core_type=core_type_for_cpu(c); out.push_back(c); break; } } } p += x->Size; }
    std::stable_sort(out.begin(),out.end(),[](const cpu_ref&a,const cpu_ref&b){return a.core_type>b.core_type;});
    return out;
}
static void pin_thread(const cpu_ref &c){ GROUP_AFFINITY ga{}; ga.Group=c.group; ga.Mask=(KAFFINITY(1)<<c.number); SetThreadGroupAffinity(GetCurrentThread(),&ga,nullptr); }
#else
static std::vector<cpu_ref> physical_cores(){ long n=sysconf(_SC_NPROCESSORS_ONLN); std::vector<cpu_ref> out; for(int i=0;i<n;++i) out.push_back(cpu_ref{0,(uint8_t)i,0}); return out; }
static void pin_thread(const cpu_ref &c){ cpu_set_t set; CPU_ZERO(&set); CPU_SET(c.number,&set); pthread_setaffinity_np(pthread_self(),sizeof(set),&set); }
#endif

struct q8_activation {
    std::vector<block_q8_K> blocks;
};

static q8_activation make_activation(int nb) {
    q8_activation a; a.blocks.resize(nb);
    uint32_t x=0x12345678u;
    auto rng=[&](){ x ^= x<<13; x ^= x>>17; x ^= x<<5; return x; };
    for(int b=0;b<nb;++b){
        auto &q=a.blocks[b]; q.d = 0.004f + 0.00003f*(float)((b%17)+1);
        for(int i=0;i<256;++i){ int v=(int)(rng()%255)-127; q.qs[i]=(int8_t)v; }
        for(int s=0;s<16;++s){ int z=0; for(int i=0;i<16;++i) z += q.qs[s*16+i]; q.bsums[s]=(int16_t)z; }
    }
    return a;
}

static inline int q8sum32(const block_q8_K &y,int g){ return (int)y.bsums[2*g] + (int)y.bsums[2*g+1]; }

struct meta_block {
    float d, dm;
    uint8_t sc[8];
    uint8_t mn[8];
    uint8_t q[128];
};
static_assert(sizeof(meta_block)==152,"meta block");

struct x8_block {
    float d, dm;
    uint8_t sc[8];
    uint8_t mn[8];
    uint8_t q[128];
    int32_t pad[1];
};
static_assert(sizeof(x8_block)==156,"x8 block");

struct pairs_block {
    float d,dm;
    uint8_t sc[8];
    uint8_t mn[8];
    uint8_t q[128];
    uint32_t pairscale[8];
};

struct row4_tile {
    float d[4];
    float dm[4];
    uint8_t sc[8][4];
    uint8_t mn[8][4];
    uint8_t qpair[4][4][32];
};
static_assert(sizeof(row4_tile)==608,"row4 tile");

struct row4madd_tile {
    float d[4];
    float dm[4];
    uint8_t sc[8][4];
    uint8_t mn[8][4];
    uint8_t qpair[4][4][32];
};
static_assert(sizeof(row4madd_tile)==608,"row4madd tile");

struct row4madd_pre_tile {
    float d[4];
    float dm[4];
    int16_t scv[8][8];
    uint8_t mn[8][4];
    uint8_t qpair[4][4][32];
};
static_assert(sizeof(row4madd_pre_tile)==704,"row4madd_pre tile");

struct matrix_repr {
    int rows=0, nb=0;
    std::vector<meta_block> meta;
    std::vector<x8_block> x8;
    std::vector<pairs_block> pairs;
    std::vector<row4_tile> row4;
    std::vector<row4madd_tile> row4madd;
    std::vector<row4madd_pre_tile> row4madd_pre;
};

static matrix_repr build_repr(const block_q4_K *src,int rows,int nb){
    matrix_repr r; r.rows=rows; r.nb=nb;
    size_t n=(size_t)rows*nb;
    r.meta.resize(n); r.x8.resize(n); r.pairs.resize(n);
    for(int row=0;row<rows;++row) for(int b=0;b<nb;++b){
        const auto &s=src[(size_t)row*nb+b];
        float d=fp16_to_fp32(s.d),dm=fp16_to_fp32(s.dmin);
        auto &m=r.meta[(size_t)row*nb+b]; m.d=d; m.dm=dm;
        auto &x=r.x8[(size_t)row*nb+b]; x.d=d; x.dm=dm; x.pad[0]=0;
        auto &p=r.pairs[(size_t)row*nb+b]; p.d=d; p.dm=dm;
        for(int g=0;g<8;++g){ uint8_t sc,mn; get_scale_min_k4(g,s.scales,sc,mn); m.sc[g]=x.sc[g]=p.sc[g]=sc; m.mn[g]=x.mn[g]=p.mn[g]=mn; p.pairscale[g]=(uint32_t)sc | ((uint32_t)sc<<16); }
        std::memcpy(m.q,s.qs,128); std::memcpy(x.q,s.qs,128); std::memcpy(p.q,s.qs,128);
    }
    int tiles=(rows+3)/4;
    r.row4.resize((size_t)tiles*nb); r.row4madd.resize((size_t)tiles*nb); r.row4madd_pre.resize((size_t)tiles*nb);
    for(int t=0;t<tiles;++t) for(int b=0;b<nb;++b){
        auto &a=r.row4[(size_t)t*nb+b]; auto &m=r.row4madd[(size_t)t*nb+b]; auto &pr=r.row4madd_pre[(size_t)t*nb+b];
        std::memset(&a,0,sizeof(a)); std::memset(&m,0,sizeof(m)); std::memset(&pr,0,sizeof(pr));
        for(int rr=0;rr<4;++rr){ int row=t*4+rr; if(row>=rows) continue; const auto &xb=r.x8[(size_t)row*nb+b];
            a.d[rr]=m.d[rr]=pr.d[rr]=xb.d; a.dm[rr]=m.dm[rr]=pr.dm[rr]=xb.dm;
            for(int g=0;g<8;++g){ a.sc[g][rr]=m.sc[g][rr]=xb.sc[g]; a.mn[g][rr]=m.mn[g][rr]=pr.mn[g][rr]=xb.mn[g]; }
            for(int pair=0;pair<4;++pair){ const uint8_t *q=xb.q+pair*32; std::memcpy(a.qpair[pair][rr],q,32); std::memcpy(m.qpair[pair][rr],q,32); std::memcpy(pr.qpair[pair][rr],q,32); }
        }
        for(int g=0;g<8;++g) for(int k=0;k<4;++k){ pr.scv[g][2*k]=pr.scv[g][2*k+1]=m.sc[g][k]; }
    }
    return r;
}

static inline __m128i expand_sc4(const uint8_t *sc4){
    uint32_t x; std::memcpy(&x,sc4,4); __m128i b=_mm_cvtsi32_si128((int)x);
    const __m128i sh=_mm_setr_epi8(0,0,1,1,2,2,3,3,(char)0x80,(char)0x80,(char)0x80,(char)0x80,(char)0x80,(char)0x80,(char)0x80,(char)0x80);
    return _mm_shuffle_epi8(b,sh);
}

static inline __m128i dot4_group_rowlane(const uint8_t qrows[4][32], const int8_t *a, bool high){
    __m128i rows16[4];
    const __m128i mask=_mm_set1_epi8(0x0f), zero=_mm_setzero_si128();
    for(int r=0;r<4;++r){
        __m128i q0=_mm_loadu_si128((const __m128i*)(qrows[r]+0));
        __m128i q1=_mm_loadu_si128((const __m128i*)(qrows[r]+16));
        if(high){ q0=_mm_and_si128(_mm_srli_epi16(q0,4),mask); q1=_mm_and_si128(_mm_srli_epi16(q1,4),mask); }
        else { q0=_mm_and_si128(q0,mask); q1=_mm_and_si128(q1,mask); }
        __m128i a0=_mm_loadu_si128((const __m128i*)(a+0));
        __m128i a1=_mm_loadu_si128((const __m128i*)(a+16));
        __m128i p0=_mm_maddubs_epi16(q0,a0); __m128i p1=_mm_maddubs_epi16(q1,a1);
        __m128i s=_mm_add_epi16(p0,p1); rows16[r]=s;
    }
    __m128i lo01=_mm_hadd_epi16(rows16[0],rows16[1]);
    __m128i lo23=_mm_hadd_epi16(rows16[2],rows16[3]);
    __m128i lo=_mm_hadd_epi16(lo01,lo23);
    return _mm_madd_epi16(lo,_mm_set1_epi16(1));
}

static inline __m128i dot4_group_rowmadd(const uint8_t qrows[4][32], const int8_t *a, bool high, const uint8_t *sc4){
    __m128i rows16[4];
    const __m128i mask=_mm_set1_epi8(0x0f);
    for(int r=0;r<4;++r){
        __m128i q0=_mm_loadu_si128((const __m128i*)(qrows[r]+0));
        __m128i q1=_mm_loadu_si128((const __m128i*)(qrows[r]+16));
        if(high){ q0=_mm_and_si128(_mm_srli_epi16(q0,4),mask); q1=_mm_and_si128(_mm_srli_epi16(q1,4),mask); }
        else { q0=_mm_and_si128(q0,mask); q1=_mm_and_si128(q1,mask); }
        __m128i a0=_mm_loadu_si128((const __m128i*)(a+0));
        __m128i a1=_mm_loadu_si128((const __m128i*)(a+16));
        rows16[r]=_mm_add_epi16(_mm_maddubs_epi16(q0,a0),_mm_maddubs_epi16(q1,a1));
    }
    __m128i lo01=_mm_hadd_epi16(rows16[0],rows16[1]);
    __m128i lo23=_mm_hadd_epi16(rows16[2],rows16[3]);
    __m128i lo=_mm_hadd_epi16(lo01,lo23);
    return _mm_madd_epi16(lo,expand_sc4(sc4));
}

static inline __m128i dot4_group_rowmadd_i2(const uint8_t qrows[4][32], const int8_t *a, bool high, const uint8_t *sc4){
    const __m128i mask=_mm_set1_epi8(0x0f);
    __m128i a0=_mm_loadu_si128((const __m128i*)(a+0)); __m128i a1=_mm_loadu_si128((const __m128i*)(a+16));
    __m128i rows16[4];
    for(int r=0;r<4;++r){
        __m128i q0=_mm_loadu_si128((const __m128i*)(qrows[r]+0)); __m128i q1=_mm_loadu_si128((const __m128i*)(qrows[r]+16));
        if(high){ q0=_mm_and_si128(_mm_srli_epi16(q0,4),mask); q1=_mm_and_si128(_mm_srli_epi16(q1,4),mask); } else { q0=_mm_and_si128(q0,mask); q1=_mm_and_si128(q1,mask); }
        __m128i p0=_mm_maddubs_epi16(q0,a0); __m128i p1=_mm_maddubs_epi16(q1,a1); rows16[r]=_mm_add_epi16(p0,p1);
    }
    __m128i x01=_mm_hadd_epi16(rows16[0],rows16[1]); __m128i x23=_mm_hadd_epi16(rows16[2],rows16[3]); __m128i x=_mm_hadd_epi16(x01,x23);
    return _mm_madd_epi16(x,expand_sc4(sc4));
}

static inline __m128i min4_group(const uint8_t *mn4,int qsum){ int32_t a[4]; for(int i=0;i<4;++i)a[i]=(int)mn4[i]*qsum; return _mm_loadu_si128((const __m128i*)a); }

static inline void store4(float *out,__m128 v){ _mm_storeu_ps(out,v); }

static void kernel_row4lane(const matrix_repr &r,const q8_activation &a,float *out,int begin_tile,int end_tile){
    for(int t=begin_tile;t<end_tile;++t){
        __m128 acc=_mm_setzero_ps();
        for(int b=0;b<r.nb;++b){ const auto &w=r.row4[(size_t)t*r.nb+b]; const auto &y=a.blocks[b]; __m128i sumi=_mm_setzero_si128(),minc=_mm_setzero_si128();
            for(int pair=0;pair<4;++pair){ int g0=pair*2,g1=g0+1; const int8_t *qa0=y.qs+g0*32; const int8_t *qa1=y.qs+g1*32;
                __m128i d0=dot4_group_rowlane(w.qpair[pair],qa0,false); __m128i d1=dot4_group_rowlane(w.qpair[pair],qa1,true);
                __m128i s0=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*)w.sc[g0])); __m128i s1=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*)w.sc[g1]));
                sumi=_mm_add_epi32(sumi,_mm_add_epi32(_mm_mullo_epi32(d0,s0),_mm_mullo_epi32(d1,s1)));
                minc=_mm_add_epi32(minc,_mm_add_epi32(min4_group(w.mn[g0],q8sum32(y,g0)),min4_group(w.mn[g1],q8sum32(y,g1)))); }
            __m128 vf=_mm_cvtepi32_ps(sumi), vm=_mm_cvtepi32_ps(minc); __m128 wd=_mm_loadu_ps(w.d),wm=_mm_loadu_ps(w.dm); __m128 scale=_mm_set1_ps(y.d);
            acc=_mm_add_ps(acc,_mm_mul_ps(scale,_mm_sub_ps(_mm_mul_ps(wd,vf),_mm_mul_ps(wm,vm)))); }
        int base=t*4; alignas(16) float tmp[4]; store4(tmp,acc); for(int rr=0;rr<4 && base+rr<r.rows;++rr) out[base+rr]=tmp[rr]; }
}

static void kernel_row4madd(const matrix_repr &r,const q8_activation &a,float *out,int begin_tile,int end_tile,int mode){
    for(int t=begin_tile;t<end_tile;++t){ __m128 acc=_mm_setzero_ps();
        for(int b=0;b<r.nb;++b){ const auto &w=r.row4madd[(size_t)t*r.nb+b]; const auto &y=a.blocks[b]; __m128i sumi=_mm_setzero_si128(),minc=_mm_setzero_si128();
            for(int pair=0;pair<4;++pair){ int g0=pair*2,g1=g0+1; __m128i d0,d1;
                if(mode==1){ d0=dot4_group_rowmadd_i2(w.qpair[pair],y.qs+g0*32,false,w.sc[g0]); d1=dot4_group_rowmadd_i2(w.qpair[pair],y.qs+g1*32,true,w.sc[g1]); }
                else { d0=dot4_group_rowmadd(w.qpair[pair],y.qs+g0*32,false,w.sc[g0]); d1=dot4_group_rowmadd(w.qpair[pair],y.qs+g1*32,true,w.sc[g1]); }
                sumi=_mm_add_epi32(sumi,_mm_add_epi32(d0,d1)); minc=_mm_add_epi32(minc,_mm_add_epi32(min4_group(w.mn[g0],q8sum32(y,g0)),min4_group(w.mn[g1],q8sum32(y,g1)))); }
            __m128 vf=_mm_cvtepi32_ps(sumi),vm=_mm_cvtepi32_ps(minc),wd=_mm_loadu_ps(w.d),wm=_mm_loadu_ps(w.dm),scale=_mm_set1_ps(y.d);
            acc=_mm_add_ps(acc,_mm_mul_ps(scale,_mm_sub_ps(_mm_mul_ps(wd,vf),_mm_mul_ps(wm,vm)))); }
        int base=t*4; alignas(16) float tmp[4]; _mm_store_ps(tmp,acc); for(int rr=0;rr<4 && base+rr<r.rows;++rr) out[base+rr]=tmp[rr]; }
}

static void kernel_row4madd_pre(const matrix_repr &r,const q8_activation &a,float *out,int begin_tile,int end_tile){
    for(int t=begin_tile;t<end_tile;++t){ __m128 acc=_mm_setzero_ps();
        for(int b=0;b<r.nb;++b){ const auto &w=r.row4madd_pre[(size_t)t*r.nb+b]; const auto &y=a.blocks[b]; __m128i sumi=_mm_setzero_si128(),minc=_mm_setzero_si128();
            for(int pair=0;pair<4;++pair){ int g0=pair*2,g1=g0+1; auto calc=[&](int g,bool hi){ __m128i rows16[4]; const __m128i mask=_mm_set1_epi8(0x0f); __m128i a0=_mm_loadu_si128((const __m128i*)(y.qs+g*32)),a1=_mm_loadu_si128((const __m128i*)(y.qs+g*32+16)); for(int rr=0;rr<4;++rr){ __m128i q0=_mm_loadu_si128((const __m128i*)(w.qpair[pair][rr]+0)),q1=_mm_loadu_si128((const __m128i*)(w.qpair[pair][rr]+16)); if(hi){q0=_mm_and_si128(_mm_srli_epi16(q0,4),mask);q1=_mm_and_si128(_mm_srli_epi16(q1,4),mask);}else{q0=_mm_and_si128(q0,mask);q1=_mm_and_si128(q1,mask);} rows16[rr]=_mm_add_epi16(_mm_maddubs_epi16(q0,a0),_mm_maddubs_epi16(q1,a1));} __m128i x=_mm_hadd_epi16(_mm_hadd_epi16(rows16[0],rows16[1]),_mm_hadd_epi16(rows16[2],rows16[3])); return _mm_madd_epi16(x,_mm_loadu_si128((const __m128i*)w.scv[g])); };
                sumi=_mm_add_epi32(sumi,_mm_add_epi32(calc(g0,false),calc(g1,true))); minc=_mm_add_epi32(minc,_mm_add_epi32(min4_group(w.mn[g0],q8sum32(y,g0)),min4_group(w.mn[g1],q8sum32(y,g1)))); }
            __m128 vf=_mm_cvtepi32_ps(sumi),vm=_mm_cvtepi32_ps(minc),wd=_mm_loadu_ps(w.d),wm=_mm_loadu_ps(w.dm),scale=_mm_set1_ps(y.d); acc=_mm_add_ps(acc,_mm_mul_ps(scale,_mm_sub_ps(_mm_mul_ps(wd,vf),_mm_mul_ps(wm,vm)))); }
        int base=t*4; alignas(16) float tmp[4]; _mm_store_ps(tmp,acc); for(int rr=0;rr<4 && base+rr<r.rows;++rr) out[base+rr]=tmp[rr]; }
}

// Reference/meta path
static inline int dot_group_scalar(const x8_block&w,const block_q8_K&y,int g){ int sum=0; const uint8_t *q=w.q+(g/2)*32; bool hi=g&1; for(int i=0;i<32;++i){ int v=hi?(q[i]>>4):(q[i]&15); sum += v*(int)y.qs[g*32+i]; } return sum; }
static void kernel_v27(const matrix_repr&r,const q8_activation&a,float*out,int rb,int re){ for(int row=rb;row<re;++row){ float z=0; for(int b=0;b<r.nb;++b){ auto&w=r.x8[(size_t)row*r.nb+b]; auto&y=a.blocks[b]; int sumi=0,minc=0; for(int g=0;g<8;++g){sumi+=(int)w.sc[g]*dot_group_scalar(w,y,g);minc+=(int)w.mn[g]*q8sum32(y,g);} z += y.d*(w.d*sumi-w.dm*minc);} out[row]=z; } }

using kernel_fn=std::function<void(float*,int,int)>;

static void run_parallel(int rows,int threads,const std::vector<cpu_ref>&cores,const kernel_fn&fn,float*out,bool tiles){ std::vector<std::thread> ts; int units=tiles?(rows+3)/4:rows; for(int ti=0;ti<threads;++ti){ int a=units*ti/threads,b=units*(ti+1)/threads; ts.emplace_back([&,ti,a,b](){ if(ti<(int)cores.size())pin_thread(cores[ti]); fn(out,a,b); }); } for(auto&t:ts)t.join(); }

static double ms_once(int rows,int threads,const std::vector<cpu_ref>&cores,const kernel_fn&fn,float*out,bool tiles){ auto t0=Clock::now(); run_parallel(rows,threads,cores,fn,out,tiles); auto t1=Clock::now(); return std::chrono::duration<double,std::milli>(t1-t0).count(); }
static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double quantile(std::vector<double> v,double q){ std::sort(v.begin(),v.end()); if(v.empty())return 0; size_t i=(size_t)std::floor(q*(v.size()-1)); return v[i]; }

struct result { std::string name; double cand=0,base=0,speed=0,p10=0,p90=0,win=0,diff=0; };

static result paired(const std::string&name,int rows,int threads,const std::vector<cpu_ref>&cores,const kernel_fn&base,const kernel_fn&cand,bool tiles,std::vector<float>&ob,std::vector<float>&oc,int pairs){ std::vector<double> rb,rc,rs; int wins=0; for(int i=0;i<pairs;++i){ double tb,tc; if((i&1)==0){tb=ms_once(rows,threads,cores,base,ob,false);tc=ms_once(rows,threads,cores,cand,oc,tiles);}else{tc=ms_once(rows,threads,cores,cand,oc,tiles);tb=ms_once(rows,threads,cores,base,ob,false);} rb.push_back(tb);rc.push_back(tc);double s=tb/tc;rs.push_back(s);if(s>1)++wins;} double diff=0; for(int i=0;i<rows;++i)diff=std::max(diff,(double)std::fabs(ob[i]-oc[i])); return result{name,median(rc),median(rb),median(rs),quantile(rs,.10),quantile(rs,.90),(double)wins/pairs,diff}; }

static bool stable(const result&r){return r.speed>1 && r.p10>1 && r.win>=.80 && r.diff==0;}

struct args {std::string model,tensor="blk.0.ffn_gate.weight",csv;int threads=1,pairs=21,rotate=0;};
static args parse(int argc,char**argv){args a;for(int i=1;i<argc;++i){std::string s=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing arg");return std::string(argv[++i]);};if(s=="--model")a.model=need();else if(s=="--tensor")a.tensor=need();else if(s=="--threads")a.threads=std::stoi(need());else if(s=="--pairs")a.pairs=std::stoi(need());else if(s=="--rotate")a.rotate=std::stoi(need());else if(s=="--csv")a.csv=need();}return a;}

int main(int argc,char**argv){try{auto A=parse(argc,argv);if(A.model.empty())throw std::runtime_error("--model required"); auto gg=load_gguf(A.model);auto &ti=find_tensor(gg,A.tensor);if(ti.dims.size()!=2)throw std::runtime_error("tensor not 2D");int cols=(int)ti.dims[0],rows=(int)ti.dims[1];if(cols%256)throw std::runtime_error("cols not multiple 256");int nb=cols/256;if(ti.type!=12)std::cerr<<"Warning tensor type="<<ti.type<<" expected GGML_TYPE_Q4_K(12)\n";auto*src=(const block_q4_K*)(gg.bytes.data()+gg.data_offset+ti.offset);auto repr=build_repr(src,rows,nb);auto act=make_activation(nb);auto cores=physical_cores();if(A.threads>(int)cores.size())A.threads=cores.size();
std::cout<<"============================================================\nTRANSIT V38 ROWMADD\n============================================================\nInput: real GGUF Q4_K tensor\nRows="<<rows<<" Cols="<<cols<<" Q4 bytes="<<std::fixed<<std::setprecision(4)<<(double)rows*nb*144/1048576.0<<" MiB\nThreads="<<A.threads<<" node=0 NUMA nodes visible=1 physical cores visible on node="<<cores.size()<<"\nPhysical-core representatives:";for(auto&c:cores)std::cout<<" g"<<c.group<<":"<<(int)c.number<<"(E"<<c.core_type<<")";std::cout<<"\nISA contract: SSSE3/SSE4.1 + AVX allowed; no AVX2/VNNI/FMA required.\n";
std::vector<float> ref(rows),tmp(rows);auto base=[&](float*out,int a,int b){kernel_v27(repr,act,out,a,b);};run_parallel(rows,A.threads,cores,base,ref.data(),false);
auto check=[&](const char*n,const kernel_fn&f,bool tiles){run_parallel(rows,A.threads,cores,f,tmp.data(),tiles);double d=0,aa=0,bb=0,ab=0;for(int i=0;i<rows;++i){d=std::max(d,(double)std::fabs(ref[i]-tmp[i]));aa+=(double)ref[i]*ref[i];bb+=(double)tmp[i]*tmp[i];ab+=(double)ref[i]*tmp[i];}std::cout<<"CORRECT "<<n<<" diff="<<d<<" cosine="<<(ab/std::sqrt(aa*bb))<<"\n";if(d!=0)throw std::runtime_error(std::string(n)+" not exact");};
kernel_fn k37=[&](float*out,int a,int b){kernel_row4lane(repr,act,out,a,b);};kernel_fn k38=[&](float*out,int a,int b){kernel_row4madd(repr,act,out,a,b,0);};kernel_fn k38i2=[&](float*out,int a,int b){kernel_row4madd(repr,act,out,a,b,1);};kernel_fn kpre=[&](float*out,int a,int b){kernel_row4madd_pre(repr,act,out,a,b);};check("ROW4LANE",k37,true);check("ROW4MADD",k38,true);check("ROW4MADD_I2",k38i2,true);check("ROW4MADD_PRE",kpre,true);
std::vector<result> R;std::vector<float> ob(rows),oc(rows);for(auto [n,f]:std::vector<std::pair<std::string,kernel_fn>>{{"V37_ROW4LANE",k37},{"V38_ROW4MADD",k38},{"V38_ROW4MADD_I2",k38i2},{"V38_ROW4MADD_PRE",kpre}}){auto r=paired(n,rows,A.threads,cores,base,f,true,ob,oc,A.pairs);R.push_back(r);std::cout<<std::left<<std::setw(28)<<r.name<<" cand_med="<<r.cand<<" base_med="<<r.base<<" paired_speedup="<<r.speed<<" p10="<<r.p10<<" p90="<<r.p90<<" win="<<r.win<<" diff="<<r.diff<<"\n";}if(!A.csv.empty()){std::ofstream o(A.csv);o<<"name,median_ms,base_median_ms,speedup_vs_baseline,p10,p90,win_rate,max_abs_diff\n";for(auto&r:R)o<<r.name<<","<<r.cand<<","<<r.base<<","<<r.speed<<","<<r.p10<<","<<r.p90<<","<<r.win<<","<<r.diff<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
