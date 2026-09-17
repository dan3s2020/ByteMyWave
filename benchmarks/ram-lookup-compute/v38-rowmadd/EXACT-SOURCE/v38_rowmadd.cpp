#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using Clock = std::chrono::steady_clock;

static inline uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007fffffu;
    int32_t exp = int32_t((x >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exp);
        if (mantissa & 0x00001000u) mantissa += 0x00002000u;
        return uint16_t(sign | (mantissa >> 13));
    }
    if (exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    if (mantissa & 0x00001000u) {
        mantissa += 0x00002000u;
        if (mantissa & 0x00800000u) {
            mantissa = 0;
            ++exp;
            if (exp >= 31) return uint16_t(sign | 0x7c00u);
        }
    }
    return uint16_t(sign | (uint32_t(exp) << 10) | (mantissa >> 13));
}

static inline float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x03ffu;
    uint32_t x;
    if (exp == 0) {
        if (mantissa == 0) {
            x = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ffu;
            uint32_t e = exp + (127 - 15);
            x = sign | (e << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mantissa << 13);
    } else {
        uint32_t e = exp + (127 - 15);
        x = sign | (e << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

#pragma pack(push,1)
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline int hsum_epi32(__m128i x) {
    x = _mm_hadd_epi32(x, x);
    x = _mm_hadd_epi32(x, x);
    return _mm_cvtsi128_si32(x);
}

static inline __m128i scale16_from_u8x4(uint32_t packed4) {
    // Input bytes: [s0,s1,s2,s3]. Output int16 lanes: [s0,s0,s1,s1,s2,s2,s3,s3].
    __m128i src = _mm_cvtsi32_si128((int)packed4);
    const __m128i sh = _mm_setr_epi8(
        0, (char)0x80, 0, (char)0x80,
        1, (char)0x80, 1, (char)0x80,
        2, (char)0x80, 2, (char)0x80,
        3, (char)0x80, 3, (char)0x80);
    return _mm_shuffle_epi8(src, sh);
}

struct cpu_ref {
#ifdef _WIN32
    WORD group = 0;
    BYTE number = 0;
#else
    int cpu = 0;
#endif
    int core_type = 0;
};

#ifdef _WIN32
static bool pin_current_thread(const cpu_ref & c) {
    GROUP_AFFINITY ga{};
    ga.Group = c.group;
    ga.Mask = (KAFFINITY(1) << c.number);
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
}
static int core_type_for_cpu(const cpu_ref & c) {
    GROUP_AFFINITY old{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &old)) return 0;
    GROUP_AFFINITY ga{};
    ga.Group = c.group;
    ga.Mask = (KAFFINITY(1) << c.number);
    if (!SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr)) return 0;
    int regs[4] = {0,0,0,0};
    __cpuidex(regs, 0, 0);
    int ct = 0;
    if ((unsigned)regs[0] >= 0x1A) {
        __cpuidex(regs, 0x1A, 0);
        ct = (regs[0] >> 24) & 0xff;
    }
    SetThreadGroupAffinity(GetCurrentThread(), &old, nullptr);
    return ct;
}
static std::vector<cpu_ref> physical_cores() {
    std::vector<cpu_ref> out;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (!len) return out;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) return out;
    uint8_t * p = buf.data();
    uint8_t * e = buf.data() + len;
    while (p < e) {
        auto * x = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
        if (x->Relationship == RelationProcessorCore) {
            const auto & pr = x->Processor;
            if (pr.GroupCount > 0 && pr.GroupMask[0].Mask) {
                KAFFINITY mask = pr.GroupMask[0].Mask;
                BYTE bit = 0;
                while (bit < 64 && ((mask >> bit) & 1) == 0) ++bit;
                if (bit < 64) {
                    cpu_ref c;
                    c.group = pr.GroupMask[0].Group;
                    c.number = bit;
                    c.core_type = core_type_for_cpu(c);
                    out.push_back(c);
                }
            }
        }
        p += x->Size;
    }
    std::stable_sort(out.begin(), out.end(), [](const cpu_ref & a, const cpu_ref & b) {
        return a.core_type > b.core_type;
    });
    return out;
}
#else
static bool pin_current_thread(const cpu_ref & c) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c.cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}
static std::vector<cpu_ref> physical_cores() {
    std::vector<cpu_ref> out;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < n; ++i) {
        cpu_ref c; c.cpu = i; c.core_type = 0; out.push_back(c);
    }
    return out;
}
#endif

struct tensor_blob {
    std::vector<uint8_t> bytes;
    uint64_t offset = 0;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
};

static uint64_t read_u64(std::ifstream & f) {
    uint64_t x; f.read(reinterpret_cast<char*>(&x), 8); return x;
}
static uint32_t read_u32(std::ifstream & f) {
    uint32_t x; f.read(reinterpret_cast<char*>(&x), 4); return x;
}
static std::string read_str(std::ifstream & f) {
    uint64_t n = read_u64(f);
    std::string s(size_t(n), '\0');
    f.read(s.data(), std::streamsize(n));
    return s;
}
static void skip_value(std::ifstream & f, uint32_t t);
static void skip_array(std::ifstream & f) {
    uint32_t t = read_u32(f);
    uint64_t n = read_u64(f);
    for (uint64_t i = 0; i < n; ++i) skip_value(f, t);
}
static void skip_value(std::ifstream & f, uint32_t t) {
    switch (t) {
        case 0: case 1: case 2: case 3: case 6: case 7: f.seekg(1, std::ios::cur); break;
        case 4: case 5: case 8: case 9: f.seekg(2, std::ios::cur); break;
        case 10: case 11: case 12: case 13: f.seekg(4, std::ios::cur); break;
        case 14: case 15: case 16: case 17: f.seekg(8, std::ios::cur); break;
        case 18: { auto s = read_str(f); (void)s; break; }
        case 19: skip_array(f); break;
        default: throw std::runtime_error("unsupported GGUF metadata type");
    }
}

static tensor_blob load_gguf_tensor(const std::string & path, const std::string & tensor) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model");
    uint32_t magic = read_u32(f);
    if (magic != 0x46554747u) throw std::runtime_error("not GGUF");
    uint32_t ver = read_u32(f);
    (void)ver;
    uint64_t nt = read_u64(f);
    uint64_t nkv = read_u64(f);
    for (uint64_t i = 0; i < nkv; ++i) {
        auto key = read_str(f); (void)key;
        uint32_t type = read_u32(f);
        skip_value(f, type);
    }
    struct desc { std::string name; std::vector<uint64_t> ne; uint32_t type; uint64_t off; };
    std::vector<desc> ds;
    ds.reserve(size_t(nt));
    for (uint64_t i = 0; i < nt; ++i) {
        desc d;
        d.name = read_str(f);
        uint32_t nd = read_u32(f);
        d.ne.resize(nd);
        for (uint32_t j = 0; j < nd; ++j) d.ne[j] = read_u64(f);
        d.type = read_u32(f);
        d.off = read_u64(f);
        ds.push_back(std::move(d));
    }
    uint64_t pos = uint64_t(f.tellg());
    uint64_t align = 32;
    uint64_t data_start = (pos + align - 1) & ~(align - 1);
    const desc * found = nullptr;
    for (auto & d : ds) if (d.name == tensor) { found = &d; break; }
    if (!found) throw std::runtime_error("tensor not found");
    if (found->type != 12) throw std::runtime_error("tensor is not GGML_TYPE_Q4_K (12)");
    if (found->ne.empty()) throw std::runtime_error("bad tensor dims");
    uint64_t elems = 1;
    for (auto n : found->ne) elems *= n;
    if (elems % 256) throw std::runtime_error("Q4_K elem count not divisible by 256");
    uint64_t nblocks = elems / 256;
    uint64_t nbytes = nblocks * sizeof(block_q4_K);
    tensor_blob tb;
    tb.offset = data_start + found->off;
    tb.ne = found->ne;
    tb.type = found->type;
    tb.bytes.resize(size_t(nbytes));
    f.seekg(std::streamoff(tb.offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(tb.bytes.data()), std::streamsize(nbytes));
    if (uint64_t(f.gcount()) != nbytes) throw std::runtime_error("short tensor read");
    return tb;
}

struct x8f_block {
    float d;
    float dm;
    uint8_t scales[8];
    uint8_t mins[8];
    uint8_t qs[128];
};
static_assert(sizeof(x8f_block) == 152, "x8f_block");

struct row4tile {
    // Four rows, one 256-column block each. Payload is lane-major.
    // Pair p covers low/high groups (2p,2p+1). Each c covers 16 activations.
    alignas(16) uint8_t q[4][4][64]; // [pair][c][row*16 + byte]
    uint8_t scale[8][4];            // [group][row]
    uint8_t minv[8][4];             // [group][row]
    float d[4];
    float dm[4];
};
static_assert(sizeof(row4tile) == 1104, "row4tile");

struct row4tile_pre {
    alignas(16) uint8_t q[4][4][64];
    alignas(16) int16_t scale16[8][8]; // [group][row duplicated twice]
    uint8_t minv[8][4];
    float d[4];
    float dm[4];
};
static_assert(sizeof(row4tile_pre) == 1200, "row4tile_pre");

static std::vector<x8f_block> build_x8f(const std::vector<uint8_t>& raw) {
    size_t nb = raw.size() / sizeof(block_q4_K);
    const block_q4_K * src = reinterpret_cast<const block_q4_K*>(raw.data());
    std::vector<x8f_block> out(nb);
    for (size_t i = 0; i < nb; ++i) {
        out[i].d = fp16_bits_to_fp32(src[i].d);
        out[i].dm = fp16_bits_to_fp32(src[i].dmin);
        for (int g = 0; g < 8; ++g) {
            uint8_t s,m; get_scale_min_k4(g, src[i].scales, s, m);
            out[i].scales[g] = s;
            out[i].mins[g] = m;
        }
        std::memcpy(out[i].qs, src[i].qs, 128);
    }
    return out;
}

static std::vector<row4tile> build_row4(const std::vector<x8f_block> & x, size_t rows, size_t nbpr) {
    if (rows % 4) throw std::runtime_error("row count must be divisible by 4 for ROW4");
    std::vector<row4tile> out((rows/4)*nbpr);
    for (size_t rg = 0; rg < rows/4; ++rg) {
        for (size_t b = 0; b < nbpr; ++b) {
            auto & t = out[rg*nbpr+b];
            for (int r = 0; r < 4; ++r) {
                const auto & s = x[(rg*4+r)*nbpr+b];
                t.d[r] = s.d; t.dm[r] = s.dm;
                for (int g=0;g<8;++g) { t.scale[g][r]=s.scales[g]; t.minv[g][r]=s.mins[g]; }
                for (int p=0;p<4;++p) {
                    const uint8_t * q = s.qs + p*32;
                    for (int c=0;c<4;++c) {
                        for (int k=0;k<16;++k) t.q[p][c][r*16+k] = q[c*16+k];
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<row4tile_pre> build_row4_pre(const std::vector<row4tile> & in) {
    std::vector<row4tile_pre> out(in.size());
    for (size_t i=0;i<in.size();++i) {
        auto & d=out[i]; const auto & s=in[i];
        std::memcpy(d.q,s.q,sizeof(d.q));
        std::memcpy(d.minv,s.minv,sizeof(d.minv));
        std::memcpy(d.d,s.d,sizeof(d.d));
        std::memcpy(d.dm,s.dm,sizeof(d.dm));
        for (int g=0;g<8;++g) for (int r=0;r<4;++r) {
            d.scale16[g][2*r+0]=s.scale[g][r];
            d.scale16[g][2*r+1]=s.scale[g][r];
        }
    }
    return out;
}

static std::vector<block_q8_K> make_q8(size_t nbpr, uint64_t seed) {
    std::vector<block_q8_K> a(nbpr);
    uint64_t s=seed;
    auto rng=[&](){s^=s<<13;s^=s>>7;s^=s<<17;return s;};
    for (size_t b=0;b<nbpr;++b) {
        a[b].d = 0.005f + float((rng()%1000))/100000.0f;
        for (int i=0;i<256;++i) a[b].qs[i] = int8_t(int(rng()%255)-127);
        for (int j=0;j<16;++j) {
            int sum=0; for(int k=0;k<16;++k) sum += a[b].qs[j*16+k];
            a[b].bsums[j]=int16_t(sum);
        }
    }
    return a;
}

static void reference(const block_q4_K * w, const block_q8_K * a, float * out, size_t rows, size_t nbpr) {
    for (size_t r=0;r<rows;++r) {
        double total=0;
        for (size_t b=0;b<nbpr;++b) {
            const auto & x=w[r*nbpr+b]; const auto & y=a[b];
            float d=fp16_bits_to_fp32(x.d), dm=fp16_bits_to_fp32(x.dmin);
            int sumi=0,mincorr=0;
            for(int g=0;g<8;++g){
                uint8_t sc,mn; get_scale_min_k4(g,x.scales,sc,mn);
                int sg=0;
                const uint8_t * q=x.qs+(g/2)*32;
                for(int k=0;k<32;++k){int qv=(g&1)?(q[k]>>4):(q[k]&15);sg+=qv*int(y.qs[g*32+k]);}
                int q8sum=y.bsums[2*g]+y.bsums[2*g+1];
                sumi+=int(sc)*sg; mincorr+=int(mn)*q8sum;
            }
            total += double(y.d)*(double(d)*sumi-double(dm)*mincorr);
        }
        out[r]=float(total);
    }
}

static void row4lane_kernel(const row4tile * w, const block_q8_K * a, float * out, size_t rows, size_t nbpr, size_t rb, size_t re) {
    const __m128i mask=_mm_set1_epi8(0x0f), ones16=_mm_set1_epi16(1);
    for(size_t rg=rb/4; rg<re/4; ++rg){
        __m128 accf=_mm_setzero_ps();
        for(size_t b=0;b<nbpr;++b){
            const auto & t=w[rg*nbpr+b]; const auto & y=a[b];
            __m128i isum=_mm_setzero_si128(), imn=_mm_setzero_si128();
            for(int p=0;p<4;++p){
                int gl=2*p, gh=gl+1;
                __m128i accl=_mm_setzero_si128(), acch=_mm_setzero_si128();
                for(int c=0;c<4;++c){
                    __m128i av=_mm_loadu_si128(reinterpret_cast<const __m128i*>(y.qs+p*64+c*16));
                    __m128i s16=_mm_setzero_si128();
                    for(int r=0;r<4;++r){
                        __m128i qv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(t.q[p][c]+r*16));
                        __m128i lo=_mm_and_si128(qv,mask), hi=_mm_and_si128(_mm_srli_epi16(qv,4),mask);
                        __m128i dl=_mm_maddubs_epi16(lo,av), dh=_mm_maddubs_epi16(hi,av);
                        int sl=hsum_epi32(_mm_madd_epi16(dl,ones16));
                        int sh=hsum_epi32(_mm_madd_epi16(dh,ones16));
                        s16=_mm_insert_epi32(s16,sl,r);
                        acch=_mm_insert_epi32(acch,sh,r);
                    }
                    accl=_mm_add_epi32(accl,s16);
                }
                uint32_t ps=0, pm=0;
                for(int r=0;r<4;++r){ ps|=uint32_t(t.scale[gl][r])<<(8*r); pm|=uint32_t(t.minv[gl][r])<<(8*r); }
                __m128i sv=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ps));
                __m128i mv=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)pm));
                isum=_mm_add_epi32(isum,_mm_mullo_epi32(accl,sv));
                int qsuml=y.bsums[2*gl]+y.bsums[2*gl+1];
                imn=_mm_add_epi32(imn,_mm_mullo_epi32(mv,_mm_set1_epi32(qsuml)));
                ps=pm=0;
                for(int r=0;r<4;++r){ ps|=uint32_t(t.scale[gh][r])<<(8*r); pm|=uint32_t(t.minv[gh][r])<<(8*r); }
                sv=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ps)); mv=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)pm));
                isum=_mm_add_epi32(isum,_mm_mullo_epi32(acch,sv));
                int qsumh=y.bsums[2*gh]+y.bsums[2*gh+1];
                imn=_mm_add_epi32(imn,_mm_mullo_epi32(mv,_mm_set1_epi32(qsumh)));
            }
            __m128 dv=_mm_loadu_ps(t.d), dmv=_mm_loadu_ps(t.dm);
            __m128 fv=_mm_sub_ps(_mm_mul_ps(dv,_mm_cvtepi32_ps(isum)),_mm_mul_ps(dmv,_mm_cvtepi32_ps(imn)));
            accf=_mm_add_ps(accf,_mm_mul_ps(_mm_set1_ps(y.d),fv));
        }
        _mm_storeu_ps(out+rg*4,accf);
    }
}

static inline void row4madd_body(const row4tile * w, const block_q8_K * a, float * out, size_t rows, size_t nbpr, size_t rb, size_t re, bool i2) {
    const __m128i mask=_mm_set1_epi8(0x0f);
    for(size_t rg=rb/4; rg<re/4; ++rg){
        __m128 accf=_mm_setzero_ps();
        for(size_t b=0;b<nbpr;++b){
            const auto & t=w[rg*nbpr+b]; const auto & y=a[b];
            __m128i isum0=_mm_setzero_si128(), isum1=_mm_setzero_si128();
            __m128i imn=_mm_setzero_si128();
            for(int p=0;p<4;++p){
                int gl=2*p, gh=gl+1;
                __m128i pl[4], ph[4];
                for(int r=0;r<4;++r){pl[r]=_mm_setzero_si128();ph[r]=_mm_setzero_si128();}
                for(int c=0;c<4;++c){
                    __m128i av=_mm_loadu_si128(reinterpret_cast<const __m128i*>(y.qs+p*64+c*16));
                    for(int r=0;r<4;++r){
                        __m128i qv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(t.q[p][c]+r*16));
                        __m128i lo=_mm_and_si128(qv,mask), hi=_mm_and_si128(_mm_srli_epi16(qv,4),mask);
                        pl[r]=_mm_add_epi16(pl[r],_mm_maddubs_epi16(lo,av));
                        ph[r]=_mm_add_epi16(ph[r],_mm_maddubs_epi16(hi,av));
                    }
                }
                uint32_t sl=0, sh=0, ml=0, mh=0;
                for(int r=0;r<4;++r){
                    sl|=uint32_t(t.scale[gl][r])<<(8*r); sh|=uint32_t(t.scale[gh][r])<<(8*r);
                    ml|=uint32_t(t.minv[gl][r])<<(8*r); mh|=uint32_t(t.minv[gh][r])<<(8*r);
                }
                __m128i svl=scale16_from_u8x4(sl), svh=scale16_from_u8x4(sh);
                __m128i dv0=_mm_setzero_si128(), dv1=_mm_setzero_si128();
                for(int r=0;r<4;++r){
                    int dl=hsum_epi32(_mm_madd_epi16(pl[r],svl));
                    int dh=hsum_epi32(_mm_madd_epi16(ph[r],svh));
                    dv0=_mm_insert_epi32(dv0,dl,r); dv1=_mm_insert_epi32(dv1,dh,r);
                }
                if(i2){isum0=_mm_add_epi32(isum0,dv0);isum1=_mm_add_epi32(isum1,dv1);}else{isum0=_mm_add_epi32(isum0,_mm_add_epi32(dv0,dv1));}
                __m128i mvl=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ml));
                __m128i mvh=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)mh));
                int ql=y.bsums[2*gl]+y.bsums[2*gl+1], qh=y.bsums[2*gh]+y.bsums[2*gh+1];
                imn=_mm_add_epi32(imn,_mm_add_epi32(_mm_mullo_epi32(mvl,_mm_set1_epi32(ql)),_mm_mullo_epi32(mvh,_mm_set1_epi32(qh))));
            }
            __m128i isum=i2?_mm_add_epi32(isum0,isum1):isum0;
            __m128 dv=_mm_loadu_ps(t.d), dmv=_mm_loadu_ps(t.dm);
            __m128 fv=_mm_sub_ps(_mm_mul_ps(dv,_mm_cvtepi32_ps(isum)),_mm_mul_ps(dmv,_mm_cvtepi32_ps(imn)));
            accf=_mm_add_ps(accf,_mm_mul_ps(_mm_set1_ps(y.d),fv));
        }
        _mm_storeu_ps(out+rg*4,accf);
    }
}
static void row4madd_kernel(const row4tile * w,const block_q8_K*a,float*out,size_t rows,size_t nbpr,size_t rb,size_t re){row4madd_body(w,a,out,rows,nbpr,rb,re,false);} 
static void row4madd_i2_kernel(const row4tile * w,const block_q8_K*a,float*out,size_t rows,size_t nbpr,size_t rb,size_t re){row4madd_body(w,a,out,rows,nbpr,rb,re,true);} 

static void row4madd_pre_kernel(const row4tile_pre * w,const block_q8_K*a,float*out,size_t rows,size_t nbpr,size_t rb,size_t re){
    const __m128i mask=_mm_set1_epi8(0x0f);
    for(size_t rg=rb/4; rg<re/4; ++rg){
        __m128 accf=_mm_setzero_ps();
        for(size_t b=0;b<nbpr;++b){
            const auto &t=w[rg*nbpr+b]; const auto &y=a[b];
            __m128i isum=_mm_setzero_si128(), imn=_mm_setzero_si128();
            for(int p=0;p<4;++p){
                int gl=2*p,gh=gl+1; __m128i pl[4],ph[4]; for(int r=0;r<4;++r){pl[r]=_mm_setzero_si128();ph[r]=_mm_setzero_si128();}
                for(int c=0;c<4;++c){__m128i av=_mm_loadu_si128(reinterpret_cast<const __m128i*>(y.qs+p*64+c*16));for(int r=0;r<4;++r){__m128i qv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(t.q[p][c]+r*16));pl[r]=_mm_add_epi16(pl[r],_mm_maddubs_epi16(_mm_and_si128(qv,mask),av));ph[r]=_mm_add_epi16(ph[r],_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(qv,4),mask),av));}}
                __m128i svl=_mm_load_si128(reinterpret_cast<const __m128i*>(t.scale16[gl])), svh=_mm_load_si128(reinterpret_cast<const __m128i*>(t.scale16[gh]));
                __m128i dlv=_mm_setzero_si128(),dhv=_mm_setzero_si128();for(int r=0;r<4;++r){dlv=_mm_insert_epi32(dlv,hsum_epi32(_mm_madd_epi16(pl[r],svl)),r);dhv=_mm_insert_epi32(dhv,hsum_epi32(_mm_madd_epi16(ph[r],svh)),r);}isum=_mm_add_epi32(isum,_mm_add_epi32(dlv,dhv));
                uint32_t ml=0,mh=0;for(int r=0;r<4;++r){ml|=uint32_t(t.minv[gl][r])<<(8*r);mh|=uint32_t(t.minv[gh][r])<<(8*r);}__m128i mvl=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)ml)),mvh=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)mh));int ql=y.bsums[2*gl]+y.bsums[2*gl+1],qh=y.bsums[2*gh]+y.bsums[2*gh+1];imn=_mm_add_epi32(imn,_mm_add_epi32(_mm_mullo_epi32(mvl,_mm_set1_epi32(ql)),_mm_mullo_epi32(mvh,_mm_set1_epi32(qh))));
            }
            __m128 dv=_mm_loadu_ps(t.d),dmv=_mm_loadu_ps(t.dm);__m128 fv=_mm_sub_ps(_mm_mul_ps(dv,_mm_cvtepi32_ps(isum)),_mm_mul_ps(dmv,_mm_cvtepi32_ps(imn)));accf=_mm_add_ps(accf,_mm_mul_ps(_mm_set1_ps(y.d),fv));
        }
        _mm_storeu_ps(out+rg*4,accf);
    }
}

struct job_ctx {
    std::vector<cpu_ref> cpus;
};

template<class Fn>
static void run_parallel(int nth, size_t rows, const std::vector<cpu_ref>& cpus, Fn fn) {
    if(nth<=1){if(!cpus.empty())pin_current_thread(cpus[0]);fn(0,rows);return;}
    std::vector<std::thread> ts;
    size_t chunk=(rows+nth-1)/nth;
    chunk=(chunk+3)&~size_t(3);
    for(int t=0;t<nth;++t){
        size_t rb=std::min(rows,size_t(t)*chunk), re=std::min(rows,rb+chunk);
        if(rb>=re) continue;
        ts.emplace_back([&,t,rb,re](){if(t<(int)cpus.size())pin_current_thread(cpus[t]);fn(rb,re);});
    }
    for(auto &th:ts) th.join();
}

struct stat { double med=0,p10=0,p90=0,win=0; };
static double percentile(std::vector<double> v,double q){if(v.empty())return 0;std::sort(v.begin(),v.end());double x=q*(v.size()-1);size_t i=size_t(x);double f=x-i;return i+1<v.size()?v[i]*(1-f)+v[i+1]*f:v[i];}
static stat summarize(const std::vector<double>&v){stat s;s.med=percentile(v,.5);s.p10=percentile(v,.1);s.p90=percentile(v,.9);s.win=std::count_if(v.begin(),v.end(),[](double x){return x>1.0;})/double(v.size());return s;}

static double maxdiff(const std::vector<float>&a,const std::vector<float>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,double(std::abs(a[i]-b[i])));return m;}
static double cosine(const std::vector<float>&a,const std::vector<float>&b){long double d=0,aa=0,bb=0;for(size_t i=0;i<a.size();++i){d+=long double(a[i])*b[i];aa+=long double(a[i])*a[i];bb+=long double(b[i])*b[i];}return double(d/std::sqrt(aa*bb));}

struct result_row { std::string mode,name; int batch=1,rowblock=4; double median_ms=0,speed=0,p10=0,p90=0,win=0,diff=0; };

static void write_csv(const std::string&path,const std::vector<result_row>&rs){std::ofstream f(path);f<<"mode,batch,name,rowblock,median_ms,speedup_vs_baseline,p10,p90,win_rate,max_abs_diff\n";for(auto&r:rs)f<<r.mode<<","<<r.batch<<","<<r.name<<","<<r.rowblock<<","<<r.median_ms<<","<<r.speed<<","<<r.p10<<","<<r.p90<<","<<r.win<<","<<r.diff<<"\n";}
static void write_json(const std::string&path,const std::vector<result_row>&rs){std::ofstream f(path);f<<"[\n";for(size_t i=0;i<rs.size();++i){auto&r=rs[i];f<<"  {\"mode\":\""<<r.mode<<"\",\"batch\":"<<r.batch<<",\"name\":\""<<r.name<<"\",\"rowblock\":"<<r.rowblock<<",\"median_ms\":"<<r.median_ms<<",\"speedup_vs_baseline\":"<<r.speed<<",\"p10\":"<<r.p10<<",\"p90\":"<<r.p90<<",\"win_rate\":"<<r.win<<",\"max_abs_diff\":"<<r.diff<<"}"<<(i+1<rs.size()?",":"")<<"\n";}f<<"]\n";}

int main(int argc,char**argv){
    std::string model,tensor,csv,json,cache; int nth=1,warm=2,iters=7,rotate=16; bool probe=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&](){if(i+1>=argc)throw std::runtime_error("missing arg");return std::string(argv[++i]);};if(a=="--model")model=next();else if(a=="--tensor")tensor=next();else if(a=="--threads")nth=std::stoi(next());else if(a=="--warmup")warm=std::stoi(next());else if(a=="--iters")iters=std::stoi(next());else if(a=="--rotate-copies")rotate=std::stoi(next());else if(a=="--csv")csv=next();else if(a=="--json")json=next();else if(a=="--cache")cache=next();else if(a=="--probe")probe=true;else if(a=="--node")next();else if(a=="--numa"){} }
    if(model.empty()||tensor.empty()) throw std::runtime_error("model/tensor required");
    auto tb=load_gguf_tensor(model,tensor); if(probe)return 0;
    uint64_t cols=tb.ne.size()>0?tb.ne[0]:0, rows=1;for(size_t i=1;i<tb.ne.size();++i)rows*=tb.ne[i];if(cols%256||rows%4)throw std::runtime_error("unsupported geometry");size_t nbpr=cols/256;
    auto cores=physical_cores(); if(nth>(int)cores.size())nth=(int)cores.size();
    std::cout<<"============================================================\nTRANSIT V38 ROWMADD\n============================================================\n";
    std::cout<<"Input: real GGUF Q4_K tensor\nRows="<<rows<<" Cols="<<cols<<" Q4 bytes="<<(tb.bytes.size()/1048576.0)<<" MiB\nThreads="<<nth<<" node=0 NUMA nodes visible=1 physical cores visible on node="<<cores.size()<<"\nPhysical-core representatives:";for(auto&c:cores){
#ifdef _WIN32
        std::cout<<" g"<<c.group<<":"<<int(c.number)<<"(E"<<c.core_type<<")";
#else
        std::cout<<" cpu"<<c.cpu;
#endif
    }std::cout<<"\nISA contract: SSSE3/SSE4.1 + AVX allowed; no AVX2/VNNI/FMA required.\n";
    auto t0=Clock::now();auto x=build_x8f(tb.bytes);auto t1=Clock::now();auto r4=build_row4(x,rows,nbpr);auto t2=Clock::now();auto r4p=build_row4_pre(r4);auto t3=Clock::now();
    double bx=std::chrono::duration<double,std::milli>(t1-t0).count();double br=std::chrono::duration<double,std::milli>(t2-t1).count();double bp=std::chrono::duration<double,std::milli>(t3-t2).count();
    std::cout<<"Build X8F "<<bx<<" ms, ROW4LANE "<<br<<" ms, ROW4MADD_PRE "<<bp<<" ms\n";
    std::cout<<"Static bytes: X8F="<<x.size()*sizeof(x8f_block)<<" ROW4LANE="<<r4.size()*sizeof(row4tile)<<" ROW4MADD="<<r4.size()*sizeof(row4tile)<<" ROW4MADD_PRE="<<r4p.size()*sizeof(row4tile_pre)<<"\n";
    auto act=make_q8(nbpr,0x123456789ULL); std::vector<float> ref(rows),o(rows); reference(reinterpret_cast<const block_q4_K*>(tb.bytes.data()),act.data(),ref.data(),rows,nbpr);
    auto check=[&](const char*name,auto fn){std::fill(o.begin(),o.end(),0);run_parallel(nth,rows,cores,[&](size_t rb,size_t re){fn(o.data(),rb,re);});double d=maxdiff(ref,o);std::cout<<"CORRECT "<<name<<" diff="<<d<<" cosine="<<cosine(ref,o)<<"\n";return d;};
    double d37=check("ROW4LANE",[&](float*out,size_t rb,size_t re){row4lane_kernel(r4.data(),act.data(),out,rows,nbpr,rb,re);});
    double d38=check("ROW4MADD",[&](float*out,size_t rb,size_t re){row4madd_kernel(r4.data(),act.data(),out,rows,nbpr,rb,re);});
    double d38i2=check("ROW4MADD_I2",[&](float*out,size_t rb,size_t re){row4madd_i2_kernel(r4.data(),act.data(),out,rows,nbpr,rb,re);});
    double d38p=check("ROW4MADD_PRE",[&](float*out,size_t rb,size_t re){row4madd_pre_kernel(r4p.data(),act.data(),out,rows,nbpr,rb,re);});
    std::vector<std::vector<row4tile>> ring; std::vector<std::vector<row4tile_pre>> ringp; ring.reserve(rotate);ringp.reserve(rotate);for(int i=0;i<rotate;++i){ring.push_back(r4);ringp.push_back(r4p);}std::cout<<"ROTATING/beyond-LLC ring: "<<rotate<<" address-distinct copies, "<<((rotate*(r4.size()*sizeof(row4tile)+r4p.size()*sizeof(row4tile_pre)))/1048576.0)<<" MiB total. No eviction sweep in timed pairing.\n";
    auto run_ms=[&](auto fn){auto s=Clock::now();fn();return std::chrono::duration<double,std::milli>(Clock::now()-s).count();};
    auto bench_pair=[&](const std::string&mode,const std::string&name,double diff,auto cand,auto base){for(int i=0;i<warm;++i){base(i);cand(i);}std::vector<double> ratios,ct;for(int i=0;i<iters*3;++i){double b,c;if(i&1){c=run_ms([&](){cand(i);});b=run_ms([&](){base(i);});}else{b=run_ms([&](){base(i);});c=run_ms([&](){cand(i);});}ratios.push_back(b/c);ct.push_back(c);}stat s=summarize(ratios);result_row rr;rr.mode=mode;rr.name=name;rr.median_ms=percentile(ct,.5);rr.speed=s.med;rr.p10=s.p10;rr.p90=s.p90;rr.win=s.win;rr.diff=diff;return rr;};
    std::vector<result_row> rs;
    auto call37=[&](const row4tile*ww,float*out){run_parallel(nth,rows,cores,[&](size_t rb,size_t re){row4lane_kernel(ww,act.data(),out,rows,nbpr,rb,re);});};
    auto call38=[&](const row4tile*ww,float*out){run_parallel(nth,rows,cores,[&](size_t rb,size_t re){row4madd_kernel(ww,act.data(),out,rows,nbpr,rb,re);});};
    auto call38i2=[&](const row4tile*ww,float*out){run_parallel(nth,rows,cores,[&](size_t rb,size_t re){row4madd_i2_kernel(ww,act.data(),out,rows,nbpr,rb,re);});};
    auto call38p=[&](const row4tile_pre*ww,float*out){run_parallel(nth,rows,cores,[&](size_t rb,size_t re){row4madd_pre_kernel(ww,act.data(),out,rows,nbpr,rb,re);});};
    std::vector<float> tmp(rows),tmp2(rows);
    auto base_ref=[&](int){reference(reinterpret_cast<const block_q4_K*>(tb.bytes.data()),act.data(),tmp2.data(),rows,nbpr);};
    auto addmode=[&](const std::string&mode,bool rot){auto base=[&](int i){(void)i;base_ref(i);};auto idx=[&](int i){return rot?size_t(i%rotate):size_t(0);};rs.push_back(bench_pair(mode,"V37_ROW4LANE",d37,[&](int i){call37(ring[idx(i)].data(),tmp.data());},base));rs.push_back(bench_pair(mode,"V38_ROW4MADD",d38,[&](int i){call38(ring[idx(i)].data(),tmp.data());},base));rs.push_back(bench_pair(mode,"V38_ROW4MADD_I2",d38i2,[&](int i){call38i2(ring[idx(i)].data(),tmp.data());},base));rs.push_back(bench_pair(mode,"V38_ROW4MADD_PRE",d38p,[&](int i){call38p(ringp[idx(i)].data(),tmp.data());},base));};
    addmode("HOT",false);addmode("ROTATE",true);
    if(!csv.empty())write_csv(csv,rs);if(!json.empty())write_json(json,rs);
    for(auto&r:rs)std::cout<<r.mode<<" "<<std::setw(22)<<std::left<<r.name<<" cand_med="<<r.median_ms<<" paired_speedup="<<r.speed<<" p10="<<r.p10<<" p90="<<r.p90<<" win="<<r.win<<" diff="<<r.diff<<"\n";
    if(!cache.empty()){std::ofstream f(cache+"/README.txt");f<<"V38 compiled representations generated from pinned tensor.\n";}
    return 0;
}
