#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#endif

static constexpr int QK_K = 256;
static constexpr int Q4K_BLOCK_BYTES = 144;
static constexpr int Q8K_BSUMS = 16;
static volatile double g_sink = 0.0;

#pragma pack(push, 1)
struct Q4KRaw { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q4KX8Meta {
    uint16_t d[8]; uint16_t dmin[8];
    uint8_t scales[8][8]; uint8_t mins[8][8]; uint8_t qs[1024];
};
#pragma pack(pop)
struct Q8K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(Q4KRaw) == 144, "Q4KRaw size");
static_assert(sizeof(Q4KX8Meta) == 1184, "Q4KX8Meta size");
static_assert(sizeof(Q8K) == 292, "Q8K size");

static inline float half_to_float(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t sign=(uint32_t)(h&0x8000u)<<16; uint32_t exp=(h>>10)&0x1fu, mant=h&0x03ffu,bits;
    if(exp==0){if(mant==0)bits=sign;else{int e=-14;while((mant&0x0400u)==0){mant<<=1;--e;}mant&=0x03ffu;bits=sign|(uint32_t)(e+127)<<23|mant<<13;}}
    else if(exp==31)bits=sign|0x7f800000u|mant<<13; else bits=sign|(exp+112u)<<23|mant<<13;
    float f; std::memcpy(&f,&bits,4); return f;
#endif
}
static inline int nearest_int_ggml(float f){float v=f+12582912.0f;int i;std::memcpy(&i,&v,sizeof(i));return(i&0x007fffff)-0x00400000;}
static inline void get_scale_min(int j,const uint8_t*q,uint8_t&d,uint8_t&m){if(j<4){d=q[j]&63;m=q[j+4]&63;}else{d=(q[j+4]&0x0f)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}}
static std::vector<uint8_t> read_bytes(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("cannot open "+p);auto n=f.tellg();f.seekg(0);std::vector<uint8_t>v((size_t)n);if(n>0)f.read((char*)v.data(),n);if(!f)throw std::runtime_error("cannot read "+p);return v;}
static std::vector<float> read_floats(const std::string&p){auto b=read_bytes(p);if(b.size()%4)throw std::runtime_error("float file size invalid");std::vector<float>v(b.size()/4);if(!b.empty())std::memcpy(v.data(),b.data(),b.size());return v;}
static void quantize_q8k(const float*x,Q8K*y,int n){if(n%256)throw std::runtime_error("n must be multiple of 256");for(int ib=0;ib<n/256;++ib){const float*p=x+ib*256;float amax=0,maxv=0;for(int i=0;i<256;++i){float a=std::fabs(p[i]);if(a>amax){amax=a;maxv=p[i];}}if(amax==0){y[ib].d=0;std::memset(y[ib].qs,0,256);std::memset(y[ib].bsums,0,32);continue;}float iscale=-127.0f/maxv;y[ib].d=1.0f/iscale;for(int i=0;i<256;++i){int q=nearest_int_ggml(iscale*p[i]);q=std::max(-128,std::min(127,q));y[ib].qs[i]=(int8_t)q;}for(int g=0;g<16;++g){int s=0;for(int k=0;k<16;++k)s+=y[ib].qs[g*16+k];y[ib].bsums[g]=(int16_t)s;}}}
static void scalar_ref(const Q4KRaw*w,const Q8K*y,float*out,int rows,int nb){for(int r=0;r<rows;++r){float total=0;const Q4KRaw*x=w+(size_t)r*nb;for(int ib=0;ib<nb;++ib){int weighted=0,mincorr=0;for(int g=0;g<8;++g){uint8_t sc,mn;get_scale_min(g,x[ib].scales,sc,mn);int pair=g>>1,hi=g&1,dot=0;for(int j=0;j<32;++j){uint8_t qb=x[ib].qs[pair*32+j];int q=hi?(qb>>4):(qb&15);dot+=q*(int)y[ib].qs[pair*64+(hi?32:0)+j];}weighted+=(int)sc*dot;mincorr+=(int)mn*((int)y[ib].bsums[g*2]+(int)y[ib].bsums[g*2+1]);}total+=y[ib].d*(half_to_float(x[ib].d)*(float)weighted-half_to_float(x[ib].dmin)*(float)mincorr);}out[r]=total;}}
static inline float hsum256_ps(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);__m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);}
static inline __m256i mm256_set_m128i(__m128i hi,__m128i lo){return _mm256_insertf128_si256(_mm256_castsi128_si256(lo),hi,1);}
static inline __m256i scale_shuffle_k4(int i){const int mask16=0x0100+0x0202*i;return _mm256_set1_epi16((short)mask16);}

static void llama_avx2(const Q4KRaw*w,const Q8K*q8,float*out,int rows,int nb){
    const __m256i m4=_mm256_set1_epi8(0x0f);constexpr uint32_t kmask1=0x3f3f3f3fU,kmask2=0x0f0f0f0fU,kmask3=0x03030303U;
    for(int row=0;row<rows;++row){const Q4KRaw*x=w+(size_t)row*nb;__m256 acc=_mm256_setzero_ps();__m128 acc_m=_mm_setzero_ps();uint32_t utmp[4]={};
        for(int i=0;i<nb;++i){const float d=q8[i].d*half_to_float(x[i].d),dmin=-q8[i].d*half_to_float(x[i].dmin);std::memcpy(utmp,x[i].scales,12);utmp[3]=((utmp[2]>>4)&kmask2)|(((utmp[1]>>6)&kmask3)<<4);const uint32_t uaux=utmp[1]&kmask1;utmp[1]=(utmp[2]&kmask2)|(((utmp[0]>>6)&kmask3)<<4);utmp[2]=uaux;utmp[0]&=kmask1;const __m128i packed=_mm_set_epi32((int)utmp[3],(int)utmp[2],(int)utmp[1],(int)utmp[0]);const __m256i mins_scales=_mm256_cvtepu8_epi16(packed);const __m256i q8sums=_mm256_loadu_si256((const __m256i*)q8[i].bsums);const __m128i q8s=_mm_hadd_epi16(_mm256_extracti128_si256(q8sums,0),_mm256_extracti128_si256(q8sums,1));const __m128i prod=_mm_madd_epi16(_mm256_extracti128_si256(mins_scales,1),q8s);acc_m=_mm_fmadd_ps(_mm_set1_ps(dmin),_mm_cvtepi32_ps(prod),acc_m);const __m128i sc128=_mm256_extracti128_si256(mins_scales,0);const __m256i scales=mm256_set_m128i(sc128,sc128);const uint8_t*q4=x[i].qs;const int8_t*yq=q8[i].qs;__m256i sumi=_mm256_setzero_si256();for(int j=0;j<4;++j){const __m256i scale_l=_mm256_shuffle_epi8(scales,scale_shuffle_k4(2*j));const __m256i scale_h=_mm256_shuffle_epi8(scales,scale_shuffle_k4(2*j+1));const __m256i bits=_mm256_loadu_si256((const __m256i*)q4);q4+=32;const __m256i ql=_mm256_and_si256(bits,m4),qh=_mm256_and_si256(_mm256_srli_epi16(bits,4),m4);__m256i p16l=_mm256_maddubs_epi16(ql,_mm256_loadu_si256((const __m256i*)yq));yq+=32;p16l=_mm256_madd_epi16(scale_l,p16l);__m256i p16h=_mm256_maddubs_epi16(qh,_mm256_loadu_si256((const __m256i*)yq));yq+=32;p16h=_mm256_madd_epi16(scale_h,p16h);sumi=_mm256_add_epi32(sumi,_mm256_add_epi32(p16l,p16h));}acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);}acc_m=_mm_add_ps(acc_m,_mm_movehl_ps(acc_m,acc_m));acc_m=_mm_add_ss(acc_m,_mm_movehdup_ps(acc_m));out[row]=hsum256_ps(acc)+_mm_cvtss_f32(acc_m);}}

static std::vector<Q4KX8Meta> repack_x8meta(const Q4KRaw*w,int rows,int nb){if(rows%8)throw std::runtime_error("rows must be divisible by 8");std::vector<Q4KX8Meta>out((size_t)(rows/8)*nb);for(int rg=0;rg<rows/8;++rg)for(int ib=0;ib<nb;++ib){Q4KX8Meta&o=out[(size_t)rg*nb+ib];for(int r=0;r<8;++r){const Q4KRaw&b=w[(size_t)(rg*8+r)*nb+ib];o.d[r]=b.d;o.dmin[r]=b.dmin;for(int g=0;g<8;++g)get_scale_min(g,b.scales,o.scales[g][r],o.mins[g][r]);}constexpr int bl=8;const int end=QK_K*4/bl;for(int i=0;i<end;++i){int r=i%8,src=(i/8)*bl,dst=i*bl;std::memcpy(o.qs+dst,w[(size_t)(rg*8+r)*nb+ib].qs+src,bl);}}return out;}
static inline uint64_t rep16_4(uint8_t x){return(uint64_t)x*UINT64_C(0x0001000100010001);}static inline __m256i scale4(const uint8_t*s){return _mm256_set_epi64x((long long)rep16_4(s[3]),(long long)rep16_4(s[2]),(long long)rep16_4(s[1]),(long long)rep16_4(s[0]));}
static void x8meta_avx2(const Q4KX8Meta*w,const Q8K*y,float*out,int rows,int nb){const __m256i mask4=_mm256_set1_epi8(15);for(int rg=0;rg<rows/8;++rg){float total[8]={};const Q4KX8Meta*xb=w+(size_t)rg*nb;for(int ib=0;ib<nb;++ib){__m256i acc03=_mm256_setzero_si256(),acc47=_mm256_setzero_si256();int mincorr[8]={};for(int pair=0;pair<4;++pair){const int g0=pair*2,g1=g0+1;const uint8_t*sc0=xb[ib].scales[g0],*sc1=xb[ib].scales[g1],*mn0=xb[ib].mins[g0],*mn1=xb[ib].mins[g1];const __m256i sl03=scale4(sc0),sh03=scale4(sc1),sl47=scale4(sc0+4),sh47=scale4(sc1+4);const int sum0=(int)y[ib].bsums[g0*2]+(int)y[ib].bsums[g0*2+1],sum1=(int)y[ib].bsums[g1*2]+(int)y[ib].bsums[g1*2+1];for(int r=0;r<8;++r)mincorr[r]+=(int)mn0[r]*sum0+(int)mn1[r]*sum1;for(int kk=0;kk<4;++kk){const int k=pair*4+kk,off=pair*64+kk*8;int64_t lo64,hi64;std::memcpy(&lo64,y[ib].qs+off,8);std::memcpy(&hi64,y[ib].qs+off+32,8);const __m256i al=_mm256_set1_epi64x(lo64),ah=_mm256_set1_epi64x(hi64);const uint8_t*q=xb[ib].qs+k*64;const __m256i q03=_mm256_loadu_si256((const __m256i*)q),q47=_mm256_loadu_si256((const __m256i*)(q+32));const __m256i q03l=_mm256_and_si256(q03,mask4),q03h=_mm256_and_si256(_mm256_srli_epi16(q03,4),mask4),q47l=_mm256_and_si256(q47,mask4),q47h=_mm256_and_si256(_mm256_srli_epi16(q47,4),mask4);const __m256i p03l=_mm256_madd_epi16(_mm256_maddubs_epi16(q03l,al),sl03),p03h=_mm256_madd_epi16(_mm256_maddubs_epi16(q03h,ah),sh03),p47l=_mm256_madd_epi16(_mm256_maddubs_epi16(q47l,al),sl47),p47h=_mm256_madd_epi16(_mm256_maddubs_epi16(q47h,ah),sh47);acc03=_mm256_add_epi32(acc03,_mm256_add_epi32(p03l,p03h));acc47=_mm256_add_epi32(acc47,_mm256_add_epi32(p47l,p47h));}}alignas(32)int32_t a03[8],a47[8];_mm256_store_si256((__m256i*)a03,_mm256_hadd_epi32(acc03,acc03));_mm256_store_si256((__m256i*)a47,_mm256_hadd_epi32(acc47,acc47));const int sumi[8]={a03[0],a03[1],a03[4],a03[5],a47[0],a47[1],a47[4],a47[5]};const float yd=y[ib].d;for(int r=0;r<8;++r)total[r]+=yd*(half_to_float(xb[ib].d[r])*(float)sumi[r]-half_to_float(xb[ib].dmin[r])*(float)mincorr[r]);}for(int r=0;r<8;++r)out[rg*8+r]=total[r];}}

struct Metrics{double max_abs=0,rel_l2=0,cosine=0;};static Metrics metrics(const std::vector<float>&a,const std::vector<float>&b){long double se=0,aa=0,bb=0,ab=0;double mx=0;for(size_t i=0;i<a.size();++i){long double d=(long double)a[i]-b[i];se+=d*d;aa+=(long double)a[i]*a[i];bb+=(long double)b[i]*b[i];ab+=(long double)a[i]*b[i];mx=std::max(mx,std::fabs((double)d));}return{mx,std::sqrt((double)(se/(aa+1e-30L))),(double)(ab/std::sqrt((aa+1e-30L)*(bb+1e-30L)))};}
static void pin_thread(int cpu){#ifdef _WIN32
    if(cpu>=0){DWORD_PTR m=(DWORD_PTR)1<<cpu;if(!SetThreadAffinityMask(GetCurrentThread(),m))throw std::runtime_error("SetThreadAffinityMask failed");}
#else
    (void)cpu;
#endif
}
static void evict_cache(std::vector<uint64_t>&buf){uint64_t s=0;for(size_t i=0;i<buf.size();i+=8){buf[i]+=UINT64_C(0x9e3779b97f4a7c15);s^=buf[i];}g_sink+=(double)(s&0xffff)*1e-300;}
struct PairStats{double base_med,cand_med,ratio_med,base_best,cand_best;};template<class F1,class F2>static PairStats paired_bench(int warmup,int iters,F1&&base,F2&&cand,std::vector<uint64_t>*evict){for(int i=0;i<warmup;++i){if(evict)evict_cache(*evict);base();if(evict)evict_cache(*evict);cand();}std::vector<double>a,b,rat;a.reserve(iters);b.reserve(iters);rat.reserve(iters);using C=std::chrono::steady_clock;for(int i=0;i<iters;++i){double tb,tc;if((i&1)==0){if(evict)evict_cache(*evict);auto t0=C::now();base();auto t1=C::now();tb=std::chrono::duration<double,std::milli>(t1-t0).count();if(evict)evict_cache(*evict);t0=C::now();cand();t1=C::now();tc=std::chrono::duration<double,std::milli>(t1-t0).count();}else{if(evict)evict_cache(*evict);auto t0=C::now();cand();auto t1=C::now();tc=std::chrono::duration<double,std::milli>(t1-t0).count();if(evict)evict_cache(*evict);t0=C::now();base();t1=C::now();tb=std::chrono::duration<double,std::milli>(t1-t0).count();}a.push_back(tb);b.push_back(tc);rat.push_back(tb/tc);g_sink+=(tb+tc)*1e-300;}auto med=[](std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];};return{med(a),med(b),med(rat),*std::min_element(a.begin(),a.end()),*std::min_element(b.begin(),b.end())};}

int main(int argc,char**argv){try{std::string q4,xpath;int rows=0,cols=0,iters=50,warmup=5,cpu=-1,evict_mb=64;for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](){if(i+1>=argc)throw std::runtime_error("missing arg value");return std::string(argv[++i]);};if(a=="--q4")q4=need();else if(a=="--x")xpath=need();else if(a=="--rows")rows=std::stoi(need());else if(a=="--cols")cols=std::stoi(need());else if(a=="--iters")iters=std::stoi(need());else if(a=="--warmup")warmup=std::stoi(need());else if(a=="--cpu")cpu=std::stoi(need());else if(a=="--evict-mb")evict_mb=std::stoi(need());else throw std::runtime_error("unknown arg "+a);}if(q4.empty()||xpath.empty()||rows<=0||cols<=0||cols%256||rows%8)throw std::runtime_error("bad args/geometry");pin_thread(cpu);auto rawb=read_bytes(q4);auto xv=read_floats(xpath);int nb=cols/256;if(rawb.size()!=(size_t)rows*nb*sizeof(Q4KRaw)||xv.size()!=(size_t)cols)throw std::runtime_error("fixture size mismatch");const Q4KRaw*raw=(const Q4KRaw*)rawb.data();std::vector<Q8K>q8(nb);quantize_q8k(xv.data(),q8.data(),cols);auto meta=repack_x8meta(raw,rows,nb);std::vector<float>ref(rows),baseout(rows),metaout(rows);scalar_ref(raw,q8.data(),ref.data(),rows,nb);llama_avx2(raw,q8.data(),baseout.data(),rows,nb);x8meta_avx2(meta.data(),q8.data(),metaout.data(),rows,nb);auto mb=metrics(ref,baseout),mm=metrics(ref,metaout);std::cout<<"============================================================\nTransit V12 apples-to-apples: llama AVX2 vs V15 x8meta\n============================================================\n";std::cout<<"matrix          : "<<rows<<" x "<<cols<<"\nblocks/row      : "<<nb<<"\nraw Q4_K MiB    : "<<std::fixed<<std::setprecision(3)<<rawb.size()/1048576.0<<"\n";std::cout<<"x8meta MiB      : "<<meta.size()*sizeof(Q4KX8Meta)/1048576.0<<"  overhead="<<std::setprecision(3)<<(100.0*(meta.size()*sizeof(Q4KX8Meta)/(double)rawb.size()-1.0))<<"%\n";std::cout<<"cpu affinity    : "<<cpu<<" (-1=scheduler)\neviction buffer : "<<evict_mb<<" MiB\n\n";std::cout<<std::scientific<<"llama correctness  max_abs="<<mb.max_abs<<" rel_L2="<<mb.rel_l2<<" cosine="<<std::fixed<<std::setprecision(12)<<mb.cosine<<"\n";std::cout<<std::scientific<<"x8meta correctness max_abs="<<mm.max_abs<<" rel_L2="<<mm.rel_l2<<" cosine="<<std::fixed<<std::setprecision(12)<<mm.cosine<<"\n\n";auto fb=[&](){llama_avx2(raw,q8.data(),baseout.data(),rows,nb);g_sink+=baseout[0]*1e-300;};auto fm=[&](){x8meta_avx2(meta.data(),q8.data(),metaout.data(),rows,nb);g_sink+=metaout[0]*1e-300;};auto hot=paired_bench(warmup,iters,fb,fm,nullptr);std::vector<uint64_t>evict((size_t)evict_mb*1024*1024/8,1);auto cold=paired_bench(std::max(1,warmup/2),iters,fb,fm,&evict);auto pr=[](const char*name,const PairStats&s){std::cout<<std::left<<std::setw(12)<<name<<" llama_med="<<std::fixed<<std::setprecision(3)<<s.base_med<<" ms  x8meta_med="<<s.cand_med<<" ms  paired_speedup="<<s.ratio_med<<"x  best="<<s.base_best<<"/"<<s.cand_best<<"\n";};pr("HOT",hot);pr("EVICTED",cold);std::cout<<"\nDecision metric: paired_speedup > 1 means x8meta wins.\n";std::cout<<"sink="<<std::setprecision(17)<<g_sink<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
