#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int QK_K = 256;
static constexpr int K_SCALE_SIZE = 12;
static constexpr uint32_t GGML_TYPE_Q4_K = 12;

#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
};
struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
};
struct block_q4_Kx8 {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
};
struct block_q4_Kx8_meta {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[8][8];
    uint8_t mins[8][8];
    uint8_t qs[1024];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 B");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 B");
static_assert(sizeof(block_q4_Kx8) == 1152, "block_q4_Kx8 must equal 8x Q4_K");
static_assert(sizeof(block_q4_Kx8_meta) == 1184, "x8meta layout size mismatch");

static inline float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
            mant &= 0x03ffu;
            out = sign | (uint32_t)(e + 127) << 23 | mant << 13;
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | mant << 13;
    } else {
        out = sign | (exp + 112u) << 23 | mant << 13;
    }
    float f; std::memcpy(&f, &out, 4); return f;
#endif
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

static inline int32_t hsum128_epi32(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}
#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    return hsum128_epi32(_mm_add_epi32(lo, hi));
}
#endif

enum GGUFType : uint32_t {
    GGUF_UINT8=0, GGUF_INT8=1, GGUF_UINT16=2, GGUF_INT16=3, GGUF_UINT32=4,
    GGUF_INT32=5, GGUF_FLOAT32=6, GGUF_BOOL=7, GGUF_STRING=8, GGUF_ARRAY=9,
    GGUF_UINT64=10, GGUF_INT64=11, GGUF_FLOAT64=12
};

template<class T> static T read_pod(std::ifstream &f) {
    T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("Unexpected EOF while parsing GGUF");
    return v;
}
static std::string read_gguf_string(std::ifstream &f) {
    uint64_t n = read_pod<uint64_t>(f);
    if (n > (1ull<<31)) throw std::runtime_error("Unreasonable GGUF string length");
    std::string s((size_t)n, '\0');
    if (n) f.read(s.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("Unexpected EOF in GGUF string");
    return s;
}
static size_t scalar_type_size(uint32_t t) {
    switch (t) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: return 1;
        case GGUF_UINT16: case GGUF_INT16: return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;
    }
}
static void skip_value(std::ifstream &f, uint32_t t);
static void skip_array(std::ifstream &f) {
    uint32_t et = read_pod<uint32_t>(f);
    uint64_t n = read_pod<uint64_t>(f);
    size_t ss = scalar_type_size(et);
    if (ss) {
        const uint64_t bytes = n * (uint64_t)ss;
        if (n && bytes / n != ss) throw std::runtime_error("GGUF array overflow");
        f.seekg((std::streamoff)bytes, std::ios::cur);
        if (!f) throw std::runtime_error("EOF skipping GGUF array");
    } else {
        for (uint64_t i=0;i<n;++i) skip_value(f, et);
    }
}
static void skip_value(std::ifstream &f, uint32_t t) {
    size_t ss = scalar_type_size(t);
    if (ss) { f.seekg((std::streamoff)ss, std::ios::cur); if(!f) throw std::runtime_error("EOF skipping metadata"); return; }
    if (t == GGUF_STRING) { (void)read_gguf_string(f); return; }
    if (t == GGUF_ARRAY) { skip_array(f); return; }
    throw std::runtime_error("Unsupported GGUF metadata type " + std::to_string(t));
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
};
struct GGUFInfo {
    uint32_t version = 0;
    uint32_t alignment = 32;
    uint64_t data_start = 0;
    TensorInfo tensor;
};

static GGUFInfo find_tensor(const std::string &path, const std::string &target) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open model blob: " + path);
    char magic[4]; f.read(magic,4);
    if (!f || std::memcmp(magic,"GGUF",4)!=0) throw std::runtime_error("File is not a GGUF blob (missing GGUF magic)");
    GGUFInfo info;
    info.version = read_pod<uint32_t>(f);
    if (info.version < 2 || info.version > 3) throw std::runtime_error("This runner supports GGUF v2/v3; found v" + std::to_string(info.version));
    uint64_t nt = read_pod<uint64_t>(f);
    uint64_t nkv = read_pod<uint64_t>(f);

    for (uint64_t i=0;i<nkv;++i) {
        std::string key = read_gguf_string(f);
        uint32_t t = read_pod<uint32_t>(f);
        if (key == "general.alignment" && (t == GGUF_UINT32 || t == GGUF_INT32)) {
            info.alignment = read_pod<uint32_t>(f);
        } else {
            skip_value(f, t);
        }
    }

    bool found=false;
    for (uint64_t i=0;i<nt;++i) {
        TensorInfo ti;
        ti.name = read_gguf_string(f);
        uint32_t nd = read_pod<uint32_t>(f);
        ti.ne.resize(nd);
        for (uint32_t d=0;d<nd;++d) ti.ne[d] = read_pod<uint64_t>(f);
        ti.type = read_pod<uint32_t>(f);
        ti.offset = read_pod<uint64_t>(f);
        if (ti.name == target) { info.tensor = ti; found=true; }
    }
    if (!found) throw std::runtime_error("Tensor not found: " + target);
    uint64_t pos = (uint64_t)f.tellg();
    uint64_t a = info.alignment ? info.alignment : 32;
    info.data_start = (pos + a - 1) / a * a;
    return info;
}

static std::vector<block_q4_K> load_q4k_tensor(const std::string &path, const GGUFInfo &g) {
    if (g.tensor.type != GGML_TYPE_Q4_K) {
        throw std::runtime_error("Target tensor GGML type is " + std::to_string(g.tensor.type) + ", expected Q4_K (12)");
    }
    uint64_t elems=1;
    for (auto n:g.tensor.ne) elems*=n;
    if (elems % QK_K) throw std::runtime_error("Q4_K tensor element count is not divisible by 256");
    uint64_t blocks=elems/QK_K;
    if (blocks > (uint64_t)std::numeric_limits<size_t>::max()/sizeof(block_q4_K)) throw std::runtime_error("Tensor too large");
    std::vector<block_q4_K> v((size_t)blocks);
    std::ifstream f(path,std::ios::binary);
    f.seekg((std::streamoff)(g.data_start + g.tensor.offset));
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size()*sizeof(block_q4_K)));
    if (!f) throw std::runtime_error("Failed reading Q4_K tensor bytes");
    return v;
}

static int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
static std::vector<block_q8_K> make_activation(int n, uint32_t seed=0xC001D00Du) {
    if (n % QK_K) throw std::runtime_error("activation length must be multiple of 256");
    std::vector<block_q8_K> y((size_t)n/QK_K);
    uint32_t state=seed;
    auto rng=[&](){ state = state*1664525u + 1013904223u; return state; };
    for (size_t ib=0; ib<y.size(); ++ib) {
        float x[QK_K];
        float amax=0, maxv=0;
        for(int j=0;j<QK_K;++j){
            int32_t z=(int32_t)(rng()>>8) - (1<<23);
            float v=(float)z/(float)(1<<23);
            v = 0.70f*v + 0.20f*std::sin((float)(ib*QK_K+j)*0.013f) + 0.10f*std::cos((float)j*0.071f);
            x[j]=v;
            if (std::fabs(v)>amax){amax=std::fabs(v);maxv=v;}
        }
        if (amax==0) { y[ib].d=0; std::memset(y[ib].qs,0,QK_K); std::memset(y[ib].bsums,0,sizeof(y[ib].bsums)); continue; }
        float iscale=-127.0f/maxv;
        y[ib].d=1.0f/iscale;
        for(int j=0;j<QK_K;++j){
            int q=nearest_int(iscale*x[j]);
            q=std::min(127,std::max(-127,q));
            y[ib].qs[j]=(int8_t)q;
        }
        for(int g=0;g<QK_K/16;++g){int sum=0;for(int j=0;j<16;++j)sum+=y[ib].qs[g*16+j];y[ib].bsums[g]=(int16_t)sum;}
    }
    return y;
}

static float dot_scalar_row(const block_q4_K *x, const block_q8_K *y, int nb) {
    float total=0;
    for(int ib=0;ib<nb;++ib){
        int sumi=0, mincorr=0;
        for(int g=0;g<8;++g){
            uint8_t sc,mn; get_scale_min_k4(g,x[ib].scales,sc,mn);
            const int pair=g>>1;
            const bool hi=g&1;
            int sg=0;
            for(int j=0;j<32;++j){
                uint8_t b=x[ib].qs[pair*32+j];
                int q=hi?(b>>4):(b&15);
                sg += q*(int)y[ib].qs[pair*64+(hi?32:0)+j];
            }
            sumi += (int)sc*sg;
            int q8sum=y[ib].bsums[g*2]+y[ib].bsums[g*2+1];
            mincorr += (int)mn*q8sum;
        }
        float d=fp16_to_fp32(x[ib].d);
        float dm=fp16_to_fp32(x[ib].dmin);
        total += y[ib].d*(d*(float)sumi-dm*(float)mincorr);
    }
    return total;
}

static float dot_simd_row(const block_q4_K *x, const block_q8_K *y, int nb) {
    float total=0;
#if defined(__AVX2__)
    const __m256i mask4=_mm256_set1_epi8(15);
    for(int ib=0;ib<nb;++ib){
        __m256i acc=_mm256_setzero_si256();
        int mincorr=0;
        for(int pair=0;pair<4;++pair){
            const int g0=pair*2,g1=g0+1;
            uint8_t sc0,mn0,sc1,mn1; get_scale_min_k4(g0,x[ib].scales,sc0,mn0); get_scale_min_k4(g1,x[ib].scales,sc1,mn1);
            __m256i qb=_mm256_loadu_si256((const __m256i*)(x[ib].qs+pair*32));
            __m256i ql=_mm256_and_si256(qb,mask4);
            __m256i qh=_mm256_and_si256(_mm256_srli_epi16(qb,4),mask4);
            __m256i al=_mm256_loadu_si256((const __m256i*)(y[ib].qs+pair*64));
            __m256i ah=_mm256_loadu_si256((const __m256i*)(y[ib].qs+pair*64+32));
            __m256i pl=_mm256_maddubs_epi16(ql,al);
            __m256i ph=_mm256_maddubs_epi16(qh,ah);
            acc=_mm256_add_epi32(acc,_mm256_madd_epi16(pl,_mm256_set1_epi16((short)sc0)));
            acc=_mm256_add_epi32(acc,_mm256_madd_epi16(ph,_mm256_set1_epi16((short)sc1)));
            mincorr += (int)mn0*(y[ib].bsums[g0*2]+y[ib].bsums[g0*2+1]);
            mincorr += (int)mn1*(y[ib].bsums[g1*2]+y[ib].bsums[g1*2+1]);
        }
        int sumi=hsum256_epi32(acc);
        total += y[ib].d*(fp16_to_fp32(x[ib].d)*(float)sumi-fp16_to_fp32(x[ib].dmin)*(float)mincorr);
    }
#elif defined(__SSSE3__)
    const __m128i mask4=_mm_set1_epi8(15);
    for(int ib=0;ib<nb;++ib){
        __m128i acc=_mm_setzero_si128();
        int mincorr=0;
        for(int pair=0;pair<4;++pair){
            const int g0=pair*2,g1=g0+1;
            uint8_t sc0,mn0,sc1,mn1; get_scale_min_k4(g0,x[ib].scales,sc0,mn0); get_scale_min_k4(g1,x[ib].scales,sc1,mn1);
            for(int half=0;half<2;++half){
                __m128i qb=_mm_loadu_si128((const __m128i*)(x[ib].qs+pair*32+half*16));
                __m128i ql=_mm_and_si128(qb,mask4);
                __m128i qh=_mm_and_si128(_mm_srli_epi16(qb,4),mask4);
                __m128i al=_mm_loadu_si128((const __m128i*)(y[ib].qs+pair*64+half*16));
                __m128i ah=_mm_loadu_si128((const __m128i*)(y[ib].qs+pair*64+32+half*16));
                acc=_mm_add_epi32(acc,_mm_madd_epi16(_mm_maddubs_epi16(ql,al),_mm_set1_epi16((short)sc0)));
                acc=_mm_add_epi32(acc,_mm_madd_epi16(_mm_maddubs_epi16(qh,ah),_mm_set1_epi16((short)sc1)));
            }
            mincorr += (int)mn0*(y[ib].bsums[g0*2]+y[ib].bsums[g0*2+1]);
            mincorr += (int)mn1*(y[ib].bsums[g1*2]+y[ib].bsums[g1*2+1]);
        }
        int sumi=hsum128_epi32(acc);
        total += y[ib].d*(fp16_to_fp32(x[ib].d)*(float)sumi-fp16_to_fp32(x[ib].dmin)*(float)mincorr);
    }
#else
    return dot_scalar_row(x,y,nb);
#endif
    return total;
}

static void gemv_scalar(const std::vector<block_q4_K>&w,const std::vector<block_q8_K>&a,int rows,int nb,float*out){
    for(int r=0;r<rows;++r)out[r]=dot_scalar_row(w.data()+(size_t)r*nb,a.data(),nb);
}
static void gemv_packed_simd(const std::vector<block_q4_K>&w,const std::vector<block_q8_K>&a,int rows,int nb,float*out){
    for(int r=0;r<rows;++r)out[r]=dot_simd_row(w.data()+(size_t)r*nb,a.data(),nb);
}

static block_q4_Kx8 make_x8(const block_q4_K in[8]) {
    block_q4_Kx8 out{};
    for(int r=0;r<8;++r){out.d[r]=in[r].d;out.dmin[r]=in[r].dmin;}
    constexpr int blocklen=8;
    const int end=QK_K*4/blocklen;
    for(int i=0;i<end;++i){
        int src_id=i%8; int src_offset=(i/8)*blocklen; int dst_offset=i*blocklen;
        std::memcpy(out.qs+dst_offset,in[src_id].qs+src_offset,blocklen);
    }
    uint8_t s[8],m[8];
    for(int i=0;i<4;++i){
        for(int r=0;r<8;++r){s[r]=in[r].scales[i]&63;m[r]=in[r].scales[i+4]&63;}
        uint8_t*p=out.scales+i*12;
        p[0]=(s[0]&63)|((s[4]&48)<<2); p[1]=(s[1]&63)|((s[5]&48)<<2); p[2]=(s[2]&63)|((s[6]&48)<<2); p[3]=(s[3]&63)|((s[7]&48)<<2);
        p[4]=(m[0]&63)|((m[4]&48)<<2); p[5]=(m[1]&63)|((m[5]&48)<<2); p[6]=(m[2]&63)|((m[6]&48)<<2); p[7]=(m[3]&63)|((m[7]&48)<<2);
        p[8]=(s[4]&15)|((m[4]&15)<<4); p[9]=(s[5]&15)|((m[5]&15)<<4); p[10]=(s[6]&15)|((m[6]&15)<<4); p[11]=(s[7]&15)|((m[7]&15)<<4);
    }
    for(int i=0;i<4;++i){
        for(int r=0;r<8;++r){s[r]=((in[r].scales[i]&192)>>2)|(in[r].scales[i+8]&15);m[r]=((in[r].scales[i+4]&192)>>2)|((in[r].scales[i+8]&240)>>4);}
        uint8_t*p=out.scales+48+i*12;
        p[0]=(s[0]&63)|((s[4]&48)<<2); p[1]=(s[1]&63)|((s[5]&48)<<2); p[2]=(s[2]&63)|((s[6]&48)<<2); p[3]=(s[3]&63)|((s[7]&48)<<2);
        p[4]=(m[0]&63)|((m[4]&48)<<2); p[5]=(m[1]&63)|((m[5]&48)<<2); p[6]=(m[2]&63)|((m[6]&48)<<2); p[7]=(m[3]&63)|((m[7]&48)<<2);
        p[8]=(s[4]&15)|((m[4]&15)<<4); p[9]=(s[5]&15)|((m[5]&15)<<4); p[10]=(s[6]&15)|((m[6]&15)<<4); p[11]=(s[7]&15)|((m[7]&15)<<4);
    }
    return out;
}
static inline void unpack_x8_scales(const uint8_t*p,uint8_t sc[8],uint8_t mn[8]){
    for(int r=0;r<4;++r){sc[r]=p[r]&63;mn[r]=p[r+4]&63;sc[r+4]=(p[r+8]&15)|((p[r]>>6)<<4);mn[r+4]=(p[r+8]>>4)|((p[r+4]>>6)<<4);}
}
static inline uint64_t rep16_4(uint8_t x){return (uint64_t)x*UINT64_C(0x0001000100010001);}

static std::vector<block_q4_Kx8> repack_x8(const std::vector<block_q4_K>&w,int rows,int nb){
    if(rows%8)throw std::runtime_error("x8 requires output rows divisible by 8");
    std::vector<block_q4_Kx8> out((size_t)(rows/8)*nb);
    for(int rg=0;rg<rows/8;++rg)for(int ib=0;ib<nb;++ib){block_q4_K t[8];for(int r=0;r<8;++r)t[r]=w[(size_t)(rg*8+r)*nb+ib];out[(size_t)rg*nb+ib]=make_x8(t);}return out;
}
static std::vector<block_q4_Kx8_meta> repack_x8meta(const std::vector<block_q4_K>&w,int rows,int nb){
    if(rows%8)throw std::runtime_error("x8meta requires output rows divisible by 8");
    std::vector<block_q4_Kx8_meta> out((size_t)(rows/8)*nb);
    for(int rg=0;rg<rows/8;++rg)for(int ib=0;ib<nb;++ib){
        block_q4_Kx8_meta&o=out[(size_t)rg*nb+ib];
        for(int r=0;r<8;++r){const auto&b=w[(size_t)(rg*8+r)*nb+ib];o.d[r]=b.d;o.dmin[r]=b.dmin;for(int g=0;g<8;++g)get_scale_min_k4(g,b.scales,o.scales[g][r],o.mins[g][r]);}
        constexpr int bl=8;const int end=QK_K*4/bl;for(int i=0;i<end;++i){int r=i%8,src=(i/8)*bl,dst=i*bl;std::memcpy(o.qs+dst,w[(size_t)(rg*8+r)*nb+ib].qs+src,bl);}
    }return out;
}

#if defined(__AVX2__)
static inline __m256i scale4(const uint8_t*s){return _mm256_set_epi64x((long long)rep16_4(s[3]),(long long)rep16_4(s[2]),(long long)rep16_4(s[1]),(long long)rep16_4(s[0]));}
#endif
#if defined(__SSSE3__)
static inline __m128i scale2(const uint8_t*s){return _mm_set_epi64x((long long)rep16_4(s[1]),(long long)rep16_4(s[0]));}
#endif

template<class XB, bool META>
static void gemv_x8_impl(const std::vector<XB>&w,const std::vector<block_q8_K>&y,int rows,int nb,float*out){
#if defined(__AVX2__)
    const __m256i mask4=_mm256_set1_epi8(15);
    for(int rg=0;rg<rows/8;++rg){float total[8]={};const XB*xb=w.data()+(size_t)rg*nb;
        for(int ib=0;ib<nb;++ib){__m256i acc03=_mm256_setzero_si256(),acc47=_mm256_setzero_si256();int mincorr[8]={};
            for(int pair=0;pair<4;++pair){int g0=pair*2,g1=g0+1;uint8_t sc0[8],mn0[8],sc1[8],mn1[8];
                if constexpr(META){for(int r=0;r<8;++r){sc0[r]=xb[ib].scales[g0][r];mn0[r]=xb[ib].mins[g0][r];sc1[r]=xb[ib].scales[g1][r];mn1[r]=xb[ib].mins[g1][r];}}
                else{unpack_x8_scales(xb[ib].scales+g0*12,sc0,mn0);unpack_x8_scales(xb[ib].scales+g1*12,sc1,mn1);}
                __m256i sl03=scale4(sc0),sh03=scale4(sc1),sl47=scale4(sc0+4),sh47=scale4(sc1+4);
                int sum0=y[ib].bsums[g0*2]+y[ib].bsums[g0*2+1],sum1=y[ib].bsums[g1*2]+y[ib].bsums[g1*2+1];for(int r=0;r<8;++r)mincorr[r]+=(int)mn0[r]*sum0+(int)mn1[r]*sum1;
                for(int kk=0;kk<4;++kk){int k=pair*4+kk,off=pair*64+kk*8;int64_t lo64,hi64;std::memcpy(&lo64,y[ib].qs+off,8);std::memcpy(&hi64,y[ib].qs+off+32,8);__m256i al=_mm256_set1_epi64x(lo64),ah=_mm256_set1_epi64x(hi64);const uint8_t*q=xb[ib].qs+k*64;__m256i q03=_mm256_loadu_si256((const __m256i*)(q)),q47=_mm256_loadu_si256((const __m256i*)(q+32));__m256i q03l=_mm256_and_si256(q03,mask4),q03h=_mm256_and_si256(_mm256_srli_epi16(q03,4),mask4),q47l=_mm256_and_si256(q47,mask4),q47h=_mm256_and_si256(_mm256_srli_epi16(q47,4),mask4);acc03=_mm256_add_epi32(acc03,_mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q03l,al),sl03),_mm256_madd_epi16(_mm256_maddubs_epi16(q03h,ah),sh03)));acc47=_mm256_add_epi32(acc47,_mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q47l,al),sl47),_mm256_madd_epi16(_mm256_maddubs_epi16(q47h,ah),sh47)));}
            }
            alignas(32)int32_t a03[8],a47[8];_mm256_store_si256((__m256i*)a03,_mm256_hadd_epi32(acc03,acc03));_mm256_store_si256((__m256i*)a47,_mm256_hadd_epi32(acc47,acc47));int sumi[8]={a03[0],a03[1],a03[4],a03[5],a47[0],a47[1],a47[4],a47[5]};for(int r=0;r<8;++r)total[r]+=y[ib].d*(fp16_to_fp32(xb[ib].d[r])*(float)sumi[r]-fp16_to_fp32(xb[ib].dmin[r])*(float)mincorr[r]);
        }for(int r=0;r<8;++r)out[rg*8+r]=total[r];}
#elif defined(__SSSE3__)
    const __m128i mask4=_mm_set1_epi8(15);
    for(int rg=0;rg<rows/8;++rg){float total[8]={};const XB*xb=w.data()+(size_t)rg*nb;
        for(int ib=0;ib<nb;++ib){__m128i acc[4]={_mm_setzero_si128(),_mm_setzero_si128(),_mm_setzero_si128(),_mm_setzero_si128()};int mincorr[8]={};
            for(int pair=0;pair<4;++pair){int g0=pair*2,g1=g0+1;uint8_t sc0[8],mn0[8],sc1[8],mn1[8];if constexpr(META){for(int r=0;r<8;++r){sc0[r]=xb[ib].scales[g0][r];mn0[r]=xb[ib].mins[g0][r];sc1[r]=xb[ib].scales[g1][r];mn1[r]=xb[ib].mins[g1][r];}}else{unpack_x8_scales(xb[ib].scales+g0*12,sc0,mn0);unpack_x8_scales(xb[ib].scales+g1*12,sc1,mn1);}__m128i sl[4],sh[4];for(int p=0;p<4;++p){sl[p]=scale2(sc0+p*2);sh[p]=scale2(sc1+p*2);}int sum0=y[ib].bsums[g0*2]+y[ib].bsums[g0*2+1],sum1=y[ib].bsums[g1*2]+y[ib].bsums[g1*2+1];for(int r=0;r<8;++r)mincorr[r]+=(int)mn0[r]*sum0+(int)mn1[r]*sum1;
                for(int kk=0;kk<4;++kk){int k=pair*4+kk,off=pair*64+kk*8;__m128i al=_mm_loadl_epi64((const __m128i*)(y[ib].qs+off)),ah=_mm_loadl_epi64((const __m128i*)(y[ib].qs+off+32));al=_mm_unpacklo_epi64(al,al);ah=_mm_unpacklo_epi64(ah,ah);const uint8_t*q=xb[ib].qs+k*64;for(int p=0;p<4;++p){__m128i qp=_mm_loadu_si128((const __m128i*)(q+p*16)),ql=_mm_and_si128(qp,mask4),qh=_mm_and_si128(_mm_srli_epi16(qp,4),mask4);acc[p]=_mm_add_epi32(acc[p],_mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ql,al),sl[p]),_mm_madd_epi16(_mm_maddubs_epi16(qh,ah),sh[p])));}}
            }int sumi[8];for(int p=0;p<4;++p){alignas(16)int32_t t[4];_mm_store_si128((__m128i*)t,_mm_hadd_epi32(acc[p],acc[p]));sumi[p*2]=t[0];sumi[p*2+1]=t[1];}for(int r=0;r<8;++r)total[r]+=y[ib].d*(fp16_to_fp32(xb[ib].d[r])*(float)sumi[r]-fp16_to_fp32(xb[ib].dmin[r])*(float)mincorr[r]);
        }for(int r=0;r<8;++r)out[rg*8+r]=total[r];}
#else
    (void)w;(void)y;(void)rows;(void)nb;(void)out;throw std::runtime_error("Build lacks SSSE3");
#endif
}
static void gemv_x8(const std::vector<block_q4_Kx8>&w,const std::vector<block_q8_K>&a,int rows,int nb,float*out){gemv_x8_impl<block_q4_Kx8,false>(w,a,rows,nb,out);}
static void gemv_x8meta(const std::vector<block_q4_Kx8_meta>&w,const std::vector<block_q8_K>&a,int rows,int nb,float*out){gemv_x8_impl<block_q4_Kx8_meta,true>(w,a,rows,nb,out);}

static double rel_l2(const std::vector<float>&a,const std::vector<float>&b){long double n=0,d=0;for(size_t i=0;i<a.size();++i){long double e=(long double)a[i]-b[i];n+=e*e;d+=(long double)a[i]*a[i];}return std::sqrt((double)(n/(d+1e-30L)));}
static double cosine(const std::vector<float>&a,const std::vector<float>&b){long double ab=0,aa=0,bb=0;for(size_t i=0;i<a.size();++i){ab+=(long double)a[i]*b[i];aa+=(long double)a[i]*a[i];bb+=(long double)b[i]*b[i];}return (double)(ab/std::sqrt((aa+1e-30L)*(bb+1e-30L)));}
static double max_abs(const std::vector<float>&a,const std::vector<float>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,(double)std::fabs(a[i]-b[i]));return m;}

template<class F> static double bench_ms(F&&fn,int iters){using C=std::chrono::steady_clock;std::vector<double>v;v.reserve(iters);for(int i=0;i<2;++i)fn();for(int i=0;i<iters;++i){auto a=C::now();fn();auto b=C::now();v.push_back(std::chrono::duration<double,std::milli>(b-a).count());}std::sort(v.begin(),v.end());return v[v.size()/2];}

static std::string dims_str(const std::vector<uint64_t>&ne){std::string s;for(size_t i=0;i<ne.size();++i){if(i)s+=" x ";s+=std::to_string(ne[i]);}return s;}

int main(int argc,char**argv){
    try{
        std::string model,tensor="blk.0.ffn_gate.weight";int iters=15;
        for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--model"&&i+1<argc)model=argv[++i];else if(a=="--tensor"&&i+1<argc)tensor=argv[++i];else if(a=="--iters"&&i+1<argc)iters=std::max(3,std::atoi(argv[++i]));else if(a=="--help"){std::cout<<"usage: transit-v15 --model <gguf/blob> [--tensor name] [--iters N]\n";return 0;}else throw std::runtime_error("Unknown/missing argument: "+a);}
        if(model.empty())throw std::runtime_error("Use --model <path-to-GGUF-or-Ollama-blob>");
        std::cout<<"================================================================================\n REAL Q4_K x Q8_K - TRANSIT X86 V15\n================================================================================\n";
        GGUFInfo g=find_tensor(model,tensor);
        if(g.tensor.ne.size()<2)throw std::runtime_error("Expected matrix tensor with >=2 dimensions");
        int n=(int)g.tensor.ne[0],rows=(int)g.tensor.ne[1];
        if(n%256||rows%8)throw std::runtime_error("This V15 benchmark needs ne[0] divisible by 256 and rows divisible by 8");
        int nb=n/256;
        auto weights=load_q4k_tensor(model,g);
        if(weights.size()!=(size_t)rows*nb)throw std::runtime_error("Tensor byte/shape mismatch");
        auto act=make_activation(n);
        std::cout<<"Tensor                  : "<<tensor<<"\n";
        std::cout<<"GGUF version            : "<<g.version<<"\n";
        std::cout<<"GGUF dimensions         : "<<dims_str(g.tensor.ne)<<" (ne order)\n";
        std::cout<<"Interpreted GEMV        : "<<rows<<" rows x "<<n<<" cols\n";
        std::cout<<"Weights                 : "<<(uint64_t)rows*n<<"\n";
        std::cout<<"Q4_K bytes              : "<<weights.size()*sizeof(block_q4_K)<<" ("<<std::fixed<<std::setprecision(3)<<(weights.size()*sizeof(block_q4_K)/(1024.0*1024.0))<<" MiB)\n";
#if defined(__AVX2__)
        std::cout<<"Compiled SIMD path      : AVX2 + SSSE3";
#elif defined(__SSSE3__)
        std::cout<<"Compiled SIMD path      : SSSE3 (Ivy-compatible)";
#else
        std::cout<<"Compiled SIMD path      : scalar-only";
#endif
#if defined(__F16C__)
        std::cout<<" + F16C\n";
#else
        std::cout<<" + software FP16 convert\n";
#endif
        std::cout<<"Repacking x8...\n";
        auto rx8=repack_x8(weights,rows,nb);auto rmeta=repack_x8meta(weights,rows,nb);
        std::cout<<"llama x8 bytes          : "<<rx8.size()*sizeof(block_q4_Kx8)<<" ("<<std::setprecision(5)<<(double)(rx8.size()*sizeof(block_q4_Kx8))/(weights.size()*sizeof(block_q4_K))<<"x)\n";
        std::cout<<"x8meta bytes            : "<<rmeta.size()*sizeof(block_q4_Kx8_meta)<<" ("<<std::setprecision(5)<<(double)(rmeta.size()*sizeof(block_q4_Kx8_meta))/(weights.size()*sizeof(block_q4_K))<<"x)\n";
        std::vector<float>ref(rows),packed(rows),x8(rows),meta(rows);
        std::cout<<"Verifying exact operator outputs...\n";
        gemv_scalar(weights,act,rows,nb,ref.data());gemv_packed_simd(weights,act,rows,nb,packed.data());gemv_x8(rx8,act,rows,nb,x8.data());gemv_x8meta(rmeta,act,rows,nb,meta.data());
        auto pr=[&](const char*name,const std::vector<float>&v){std::cout<<std::left<<std::setw(24)<<name<<" rel-L2="<<std::scientific<<std::setprecision(3)<<rel_l2(ref,v)<<"  cosine="<<std::fixed<<std::setprecision(9)<<cosine(ref,v)<<"  max_abs="<<std::scientific<<max_abs(ref,v)<<"\n";};
        pr("packed SIMD",packed);pr("llama x8",x8);pr("x8meta",meta);
        std::cout<<"\n================================================================================\n BENCHMARK (single thread, median)\n================================================================================\n";
        volatile float sink=0;auto bscalar=[&](){gemv_scalar(weights,act,rows,nb,ref.data());sink+=ref[0];};auto bpacked=[&](){gemv_packed_simd(weights,act,rows,nb,packed.data());sink+=packed[0];};auto bx8=[&](){gemv_x8(rx8,act,rows,nb,x8.data());sink+=x8[0];};auto bmeta=[&](){gemv_x8meta(rmeta,act,rows,nb,meta.data());sink+=meta[0];};
        double ts=bench_ms(bscalar,std::max(3,iters/3)),tp=bench_ms(bpacked,iters),tx=bench_ms(bx8,iters),tm=bench_ms(bmeta,iters);
        double ops=(double)rows*n;auto line=[&](const char*name,double ms){std::cout<<std::left<<std::setw(24)<<name<<std::right<<std::fixed<<std::setprecision(3)<<std::setw(9)<<ms<<" ms   "<<std::setprecision(2)<<std::setw(8)<<(ops/(ms*1e6))<<" GOP/s logical\n";};
        line("scalar exact",ts);line("packed SIMD control",tp);line("llama-x8 fused",tx);line("x8meta fused",tm);
        std::cout<<"\nllama-x8 / packed       : "<<std::setprecision(3)<<tp/tx<<"x\n";
        std::cout<<"x8meta / packed         : "<<tp/tm<<"x\n";
        std::cout<<"x8meta RAM overhead     : "<<std::setprecision(2)<<100.0*((double)sizeof(block_q4_Kx8_meta)/(8.0*sizeof(block_q4_K))-1.0)<<"%\n";
        std::cout<<"sink                    : "<<sink<<"\n";
        std::cout<<"================================================================================\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
