#include "sm9_cuda.cuh"
#include "sm9_cuda_def.cuh"

#ifndef _re_define_
#define _re_define_
__device__ void *calloc_cuda(size_t num,size_t size_cuda)
{
    void *ret = NULL;
    cudaMalloc((void **)&ret, num * size_cuda);
    return ret;
}

__device__ void *malloc_cuda(size_t size_cuda)
{
    void *ret = NULL;
    cudaMalloc((void **)&ret, size_cuda);
    return ret;
}

// __device__ void *memcpy_cuda(void *str1, const void *str2, size_t n)
// {
//     void *res;
//     cudaMemcpy(str1, str2, n, cudaMemcpy);
//     return
// }
#endif

#ifndef mrcore_c
#define mrcore_c

#include <stdlib.h>
#include <string.h>


#ifdef MR_FP
#include <math.h>
#endif


/*** Multi-Threaded Support ***/

#ifndef MR_GENERIC_MT

  #ifdef MR_OPENMP_MT
    #include <omp.h>

#define MR_MIP_EXISTS

    miracl *mr_mip;
    #pragma omp threadprivate(mr_mip)
    
    __device__ miracl *get_mip_cuda()
    {
        return mr_mip; 
    }

    __device__ void mr_init_threading_cuda()
    {
    }

    __device__ void mr_end_threading_cuda()
    {
    }

  #endif

  #ifdef MR_WINDOWS_MT
    #include <windows.h>
    DWORD mr_key;   

    __device__ miracl *get_mip_cuda()
    {
        return (miracl *)TlsGetValue_cuda(mr_key); 
    }

    __device__ void mr_init_threading_cuda()
    {
        mr_key=TlsAlloc();
    }

    __device__ void mr_end_threading_cuda()
    {
        TlsFree(mr_key);
    }

  #endif

  #ifdef MR_UNIX_MT
    #include <pthread.h>
    pthread_key_t mr_key;

    __device__ miracl *get_mip_cuda()
    {
        return (miracl *)pthread_getspecific_cuda(mr_key); 
    }

    __device__ void mr_init_threading_cuda()
    {
        pthread_key_create_cuda(&mr_key,(void(*)(void *))NULL);
    }

    __device__ void mr_end_threading_cuda()
    {
        pthread_key_delete_cuda(mr_key);
    }
  #endif

  #ifndef MR_WINDOWS_MT
    #ifndef MR_UNIX_MT
      #ifndef MR_OPENMP_MT
        #ifdef MR_STATIC
          miracl mip;
          miracl *mr_mip=&mip;
        #else
          miracl *mr_mip=NULL;  /* MIRACL's one and only global variable */
        #endif
#define MR_MIP_EXISTS
        __device__ miracl *get_mip_cuda()
        {
          return (miracl *)mr_mip; 
        }
      #endif
    #endif
  #endif

#ifdef MR_MIP_EXISTS
    __device__ void set_mip_cuda(miracl *mip)
    {
        mr_mip=mip;
    }
#endif

#endif

/* See Advanced Windows by Jeffrey Richter, Chapter 12 for methods for
   creating different instances of this global for each executing thread 
   when using Windows '95/NT
*/

#ifdef MR_STATIC

#if MIRACL==8

static const int mr_small_primes[]=
{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,
107,109,113,127,0};

#else

static const int mr_small_primes[]=
{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,
107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,
223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,
337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,
457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,
593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,
719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,
857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,
997,0};

#endif

#endif

#ifndef MR_STRIPPED_DOWN
#ifndef MR_NO_STANDARD_IO

__device__ static char *names[] =
{(char *)"your program",(char *)"innum",(char *)"otnum",(char *)"jack_cuda",(char *)"normalise_cuda",
(char *)"multiply_cuda",(char *)"divide_cuda",(char *)"incr_cuda",(char *)"decr_cuda",(char *)"premult_cuda",
(char *)"subdiv_cuda",(char *)"fdsize",(char *)"egcd",(char *)"cbase",
(char *)"cinnum",(char *)"cotnum",(char *)"nroot",(char *)"power",
(char *)"powmod",(char *)"bigdig",(char *)"bigrand",(char *)"nxprime",(char *)"isprime",
(char *)"mirvar_cuda",(char *)"mad_cuda",(char *)"multi_inverse",(char *)"putdig",
(char *)"add_cuda",(char *)"subtract_cuda",(char *)"mirsys_cuda",(char *)"xgcd_cuda",
(char *)"fpack",(char *)"dconv",(char *)"mr_shift_cuda",(char *)"mround",(char *)"fmul",
(char *)"fdiv",(char *)"fadd",(char *)"fsub",(char *)"fcomp",(char *)"fconv",
(char *)"frecip",(char *)"fpmul",(char *)"fincr",(char *)"",(char *)"ftrunc",
(char *)"frand",(char *)"sftbit",(char *)"build",(char *)"logb2_cuda",(char *)"expint",
(char *)"fpower",(char *)"froot",(char *)"fpi",(char *)"fexp",(char *)"flog",(char *)"fpowf",
(char *)"ftan",(char *)"fatan",(char *)"fsin",(char *)"fasin",(char *)"fcos",(char *)"facos",
(char *)"ftanh",(char *)"fatanh",(char *)"fsinh",(char *)"fasinh",(char *)"fcosh",
(char *)"facosh",(char *)"flop",(char *)"gprime",(char *)"powltr",(char *)"fft_mult",
(char *)"crt_init",(char *)"crt",(char *)"otstr",(char *)"instr",(char *)"cotstr",(char *)"cinstr",(char *)"powmod2",
(char *)"prepare_monty_cuda",(char *)"nres_cuda",(char *)"redc_cuda",(char *)"nres_modmult_cuda",(char *)"nres_powmod",
(char *)"nres_moddiv_cuda",(char *)"nres_powltr",(char *)"divisible",(char *)"remain_cuda",
(char *)"fmodulo",(char *)"nres_modadd_cuda",(char *)"nres_modsub_cuda",(char *)"nres_negate_cuda",
(char *)"ecurve_init_cuda",(char *)"ecurve_add_cuda",(char *)"ecurve_mult_cuda",
(char *)"epoint_init_cuda",(char *)"epoint_set_cuda",(char *)"epoint_get_cuda",(char *)"nres_powmod2",
(char *)"nres_sqroot_cuda",(char *)"sqroot",(char *)"nres_premult_cuda",(char *)"ecurve_mult2",
(char *)"ecurve_sub_cuda",(char *)"trial_division",(char *)"nxsafeprime",(char *)"nres_lucas_cuda",(char *)"lucas",
(char *)"brick_init",(char *)"pow_brick",(char *)"set_user_function",
(char *)"nres_powmodn",(char *)"powmodn",(char *)"ecurve_multn",
(char *)"ebrick_init",(char *)"mul_brick",(char *)"epoint_norm_cuda",(char *)"nres_multi_inverse_cuda",(char *)"",
(char *)"nres_dotprod",(char *)"epoint_negate_cuda",(char *)"ecurve_multi_add",
(char *)"ecurve2_init",(char *)"",(char *)"epoint2_set",(char *)"epoint2_norm",(char *)"epoint2_get",
(char *)"epoint2_comp",(char *)"ecurve2_add",(char *)"epoint2_negate",(char *)"ecurve2_sub",
(char *)"ecurve2_multi_add",(char *)"ecurve2_mult",(char *)"ecurve2_multn",(char *)"ecurve2_mult2",
(char *)"ebrick2_init",(char *)"mul2_brick",(char *)"prepare_basis",(char *)"strong_bigrand",
(char *)"bytes_to_big_cuda",(char *)"big_to_bytes_cuda",(char *)"set_io_buffer_size",
(char *)"epoint_getxyz",(char *)"epoint_double_add",(char *)"nres_double_inverse_cuda",
(char *)"double_inverse",(char *)"epoint_x",(char *)"hamming",(char *)"expb2_cuda",(char *)"bigbits",
(char *)"nres_lazy_cuda",(char *)"zzn2_imul_cuda",(char *)"nres_double_modadd_cuda",(char *)"nres_double_modsub_cuda",
/*155*/(char *)"",(char *)"zzn2_from_int_cuda",(char *)"zzn2_negate_cuda",(char *)"zzn2_conj_cuda",(char *)"zzn2_add_cuda",
(char *)"zzn2_sub_cuda",(char *)"zzn2_smul_cuda",(char *)"zzn2_mul_cuda",(char *)"zzn2_inv_cuda",(char *)"zzn2_timesi_cuda",(char *)"zzn2_powl",
(char *)"zzn2_from_bigs_cuda",(char *)"zzn2_from_big_cuda",(char *)"zzn2_from_ints",
(char *)"zzn2_sadd_cuda",(char *)"zzn2_ssub_cuda",(char *)"zzn2_times_irp",(char *)"zzn2_div2_cuda",
(char *)"zzn3_from_int",(char *)"zzn3_from_ints",(char *)"zzn3_from_bigs",
(char *)"zzn3_from_big",(char *)"zzn3_negate",(char *)"zzn3_powq",(char *)"zzn3_init",
(char *)"zzn3_add",(char *)"zzn3_sadd",(char *)"zzn3_sub",(char *)"zzn3_ssub",(char *)"zzn3_smul",
(char *)"zzn3_imul",(char *)"zzn3_mul",(char *)"zzn3_inv",(char *)"zzn3_div2",(char *)"zzn3_timesi",
(char *)"epoint_multi_norm_cuda",(char *)"mr_jsf_cuda",(char *)"epoint2_multi_norm",
(char *)"ecn2_compare",(char *)"ecn2_norm_cuda",(char *)"ecn2_set_cuda",(char *)"zzn2_txx_cuda",
(char *)"zzn2_txd_cuda",(char *)"nres_div2_cuda",(char *)"nres_div3_cuda",(char *)"zzn2_div3",
(char *)"ecn2_setx",(char *)"ecn2_rhs_cuda",(char *)"zzn2_qr_cuda",(char *)"zzn2_sqrt_cuda",(char *)"ecn2_add_cuda",(char *)"ecn2_mul2_jsf",(char *)"ecn2_mul_cuda",
(char *)"nres_div5_cuda",(char *)"zzn2_div5_cuda",(char *)"zzn2_sqr_cuda",(char *)"ecn2_add_sub_cuda",(char *)"ecn2_psi_cuda",(char *)"invmodp_cuda",
(char *)"zzn2_multi_inverse_cuda",(char *)"ecn2_multi_norm_cuda",(char *)"ecn2_precomp_cuda",(char *)"ecn2_mul4_gls_v",
(char *)"ecn2_mul2",(char *)"ecn2_precomp_gls_cuda",(char *)"ecn2_mul2_gls",
(char *)"ecn2_brick_init",(char *)"ecn2_mul_brick_gls",(char *)"ecn2_multn",(char *)"zzn3_timesi2",
(char *)"nres_complex_cuda",(char *)"zzn4_from_int_cuda",(char *)"zzn4_negate_cuda",(char *)"zzn4_conj_cuda",(char *)"zzn4_add_cuda",(char *)"zzn4_sadd",(char *)"zzn4_sub_cuda",(char *)"zzn4_ssub",(char *)"zzn4_smul_cuda",(char *)"zzn4_sqr_cuda",
(char *)"zzn4_mul_cuda",(char *)"zzn4_inv_cuda",(char *)"zzn4_div2",(char *)"zzn4_powq_cuda",(char *)"zzn4_tx_cuda",(char *)"zzn4_imul",(char *)"zzn4_lmul",(char *)"zzn4_from_big",
(char *)"ecn2_mult4"};

/* 0 - 243 (244 in all) */

#endif
#endif

#ifdef MR_NOASM

/* C only versions of muldiv_cuda/muldvd_cuda/muldvd2_cuda/muldvm_cuda */
/* Note that mr_large should be twice the size_cuda of mr_small */

__device__ mr_small muldiv_cuda(mr_small a,mr_small b,mr_small c,mr_small m,mr_small *rp)
{
    mr_small q;
    mr_large ldres,p=(mr_large)a*b+c;
    q=(mr_small)(MR_LROUND(p/m));
    *rp=(mr_small)(p-(mr_large)q*m);
    return q;
}

#ifdef MR_FP_ROUNDING

__device__ mr_small imuldiv_cuda(mr_small a,mr_small b,mr_small c,mr_small m,mr_large im,mr_small *rp)
{
    mr_small q;
    mr_large ldres,p=(mr_large)a*b+c;
    q=(mr_small)MR_LROUND(p*im);
    *rp=(mr_small)(p-(mr_large)q*m);
    return q;
}

#endif

#ifndef MR_NOFULLWIDTH

__device__ mr_small muldvm_cuda(mr_small a,mr_small c,mr_small m,mr_small *rp)
{
    mr_small q;
    union doubleword dble;
    dble.h[MR_BOT]=c;
    dble.h[MR_TOP]=a;

    q=(mr_small)(dble.d/m);
    *rp=(mr_small)(dble.d-(mr_large)q*m);
    return q;
}

__device__ mr_small muldvd_cuda(mr_small a,mr_small b,mr_small c,mr_small *rp)
{
    union doubleword dble;
    dble.d=(mr_large)a*b+c;

    *rp=dble.h[MR_BOT];
    return dble.h[MR_TOP];
}

__device__ void muldvd2_cuda(mr_small a,mr_small b,mr_small *c,mr_small *rp)
{
    union doubleword dble;
    dble.d=(mr_large)a*b+*c+*rp;
    *rp=dble.h[MR_BOT];
    *c=dble.h[MR_TOP];
}

#endif
#endif

#ifdef MR_NOFULLWIDTH

/* no FULLWIDTH working, so supply dummies */

/*

mr_small muldvd_cuda(mr_small a,mr_small b,mr_small c,mr_small *rp)
{
    return (mr_small)0;
}

mr_small muldvm_cuda(mr_small a,mr_small c,mr_small m,mr_small *rp)
{
    return (mr_small)0;
}

void muldvd2_cuda(mr_small a,mr_small b,mr_small *c,mr_small *rp)
{
}

*/

#endif

#ifndef MR_NO_STANDARD_IO

__device__ static void mputs_cuda(char *s)
{ /* output a string */
    int i=0;
    while (s[i]!=0) fputc((int)s[i++],stdout);
}
#endif

__device__ void mr_berror_cuda(_MIPD_ int nerr)
{  /*  Big number error routine  */
#ifndef MR_STRIPPED_DOWN
int i;
#endif

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

if (mr_mip->ERCON)
{
    mr_mip->ERNUM=nerr;
    return;
}
#ifndef MR_NO_STANDARD_IO

#ifndef MR_STRIPPED_DOWN
// mputs((char *)"\nMIRACL error from routine ");
// if (mr_mip->depth<MR_MAXDEPTH) mputs(names[mr_mip->trace[mr_mip->depth]]);
// else                           mputs((char *)"???");
// fputc('\n',stdout);

// for (i=mr_mip->depth-1;i>=0;i--)
// {
//     mputs((char *)"              called from ");
//     if (i<MR_MAXDEPTH) mputs(names[mr_mip->trace[i]]);
//     else               mputs((char *)"???");
//     fputc('\n',stdout);
// }

// switch (nerr)
// {
// case 1 :
// mputs((char *)"Number base too big for representation\n");
// break;
// case 2 :
// mputs((char *)"Division by zero_cuda attempted\n");
// break;
// case 3 : 
// mputs((char *)"Overflow - Number too big\n");
// break;
// case 4 :
// mputs((char *)"Internal result is negative\n");
// break;
// case 5 : 
// mputs((char *)"Input format error\n");
// break;
// case 6 :
// mputs((char *)"Illegal number base\n");
// break;
// case 7 : 
// mputs((char *)"Illegal parameter usage\n");
// break;
// case 8 :
// mputs((char *)"Out of space\n");
// break;
// case 9 :
// mputs((char *)"Even root of a negative number\n");
// break;
// case 10:
// mputs((char *)"Raising integer to negative power\n");
// break;
// case 11:
// mputs((char *)"Attempt to take illegal root\n");
// break;
// case 12:
// mputs((char *)"Integer operation attempted on Flash number\n");
// break;
// case 13:
// mputs((char *)"Flash overflow\n");
// break;
// case 14:
// mputs((char *)"Numbers too big\n");
// break;
// case 15:
// mputs((char *)"Log of a non-positive number\n");
// break;
// case 16:
// mputs((char *)"Flash to double conversion failure\n");
// break;
// case 17:
// mputs((char *)"I/O buffer overflow\n");
// break;
// case 18:
// mputs((char *)"MIRACL not initialised - no call to mirsys_cuda()\n");
// break;
// case 19:
// mputs((char *)"Illegal modulus \n");
// break;
// case 20:
// mputs((char *)"No modulus defined\n");
// break;
// case 21:
// mputs((char *)"Exponent too big\n");
// break;
// case 22:
// mputs((char *)"Unsupported Feature - check mirdef.h\n");
// break;
// case 23:
// mputs((char *)"Specified double length type isn't double length\n");
// break;
// case 24:
// mputs((char *)"Specified basis is NOT irreducible\n");
// break;
// case 25:
// mputs((char *)"Unable to control Floating-point rounding\n");
// break;
// case 26:
// mputs((char *)"Base must be binary (MR_ALWAYS_BINARY defined in mirdef.h ?)\n");
// break;
// case 27:
// mputs((char *)"No irreducible basis defined\n");
// break;
// case 28:
// mputs((char *)"Composite modulus\n");
// break;
// case 29:
// mputs((char *)"Input/output error when reading from RNG device node\n");
// break;
// default:
// mputs((char *)"Undefined error\n");
// break;
// }
// exit(0);
#else
mputs((char *)"MIRACL error\n");
exit(0);
#endif

#endif
}

#ifndef MR_STRIPPED_DOWN

__device__ void mr_track_cuda(_MIPDO_ )
{ /* track course of program execution *
   * through the MIRACL routines       */

#ifndef MR_NO_STANDARD_IO

    int i;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    // for (i=0;i<mr_mip->depth;i++) fputc('-',stdout);
    // fputc('>',stdout);
    // mputs(names[mr_mip->trace[mr_mip->depth]]);
    // fputc('\n',stdout);
#endif
}

#endif

#ifndef MR_NO_RAND

__device__ mr_small brand_cuda(_MIPDO_ )
{ /* Marsaglia & Zaman random number generator */
    int i,k;
    mr_unsign32 pdiff,t;
    mr_small r;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->lg2b>32)
    { /* underlying type is > 32 bits. Assume <= 64 bits */
        mr_mip->rndptr+=2;
        if (mr_mip->rndptr<NK-1)
        {
            r=(mr_small)mr_mip->ira[mr_mip->rndptr];
            r=mr_shiftbits_cuda(r,mr_mip->lg2b-32);
            r+=(mr_small)mr_mip->ira[mr_mip->rndptr+1];
            return r;
        }
    }
    else
    {
        mr_mip->rndptr++;
        if (mr_mip->rndptr<NK) return (mr_small)mr_mip->ira[mr_mip->rndptr];
    }
    mr_mip->rndptr=0;
    for (i=0,k=NK-NJ;i<NK;i++,k++)
    { /* calculate next NK values */
        if (k==NK) k=0;
        t=mr_mip->ira[k];
        pdiff=t - mr_mip->ira[i] - mr_mip->borrow;
        if (pdiff<t) mr_mip->borrow=0;
        if (pdiff>t) mr_mip->borrow=1;
        mr_mip->ira[i]=pdiff; 
    }
    if (mr_mip->lg2b>32)
    { /* double up */
        r=(mr_small)mr_mip->ira[0];
        r=mr_shiftbits_cuda(r,mr_mip->lg2b-32);
        r+=(mr_small)mr_mip->ira[1];
        return r;
    }
    else return (mr_small)(mr_mip->ira[0]);
}

__device__ void irand_cuda(_MIPD_ mr_unsign32 seed)
{ /* initialise random number system */
    int i,in;
    mr_unsign32 t,m=1L;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    mr_mip->borrow=0L;
    mr_mip->rndptr=0;
    mr_mip->ira[0]=seed;
    for (i=1;i<NK;i++)
    { /* fill initialisation vector */
        in=(NV*i)%NK;
        mr_mip->ira[in]=m; 
        t=m;
        m=seed-m;
        seed=t;
    }
    for (i=0;i<1000;i++) brand_cuda(_MIPPO_ ); /* "warm-up" & stir the generator */
}

#endif

__device__ mr_small mr_shiftbits_cuda(mr_small x,int n)
{
#ifdef MR_FP
    int i;
    mr_small dres;
    if (n==0) return x;
    if (n>0)
    {
        for (i=0;i<n;i++) x=x+x;
        return x;
    }
    n=-n;
    for (i=0;i<n;i++) x=MR_DIV(x,2.0);
    return x;
#else
    if (n==0) return x;
    if (n>0) x<<=n;
    else x>>=(-n);
    return x;
#endif

}

__device__ mr_small mr_setbase_cuda(_MIPD_ mr_small nb)
{  /* set base. Pack as many digits as  *
    * possible into each computer word  */
    mr_small temp;
#ifdef MR_FP
    mr_small dres;
#endif
#ifndef MR_NOFULLWIDTH
    BOOL fits;
    int bits;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    fits=FALSE;
    bits=MIRACL;
    while (bits>1) 
    {
        bits/=2;
        temp=((mr_small)1<<bits);
        if (temp==nb)
        {
            fits=TRUE;
            break;
        }
        if (temp<nb || (bits%2)!=0) break;
    }
    if (fits)
    {
        mr_mip->apbase=nb;
        mr_mip->pack=MIRACL/bits;
        mr_mip->base=0;
        return 0;
    }
#endif
    mr_mip->apbase=nb;
    mr_mip->pack=1;
    mr_mip->base=nb;
#ifdef MR_SIMPLE_BASE
    return 0;
#else
    if (mr_mip->base==0) return 0;
    temp=MR_DIV(MAXBASE,nb);
    while (temp>=nb)
    {
        temp=MR_DIV(temp,nb);
        mr_mip->base*=nb;
        mr_mip->pack++;
    }
#ifdef MR_FP_ROUNDING
    mr_mip->inverse_base=mr_invert(mr_mip->base);
    return mr_mip->inverse_base;
#else
    return 0;
#endif
#endif
}

#ifdef MR_FLASH

__device__ BOOL fit_cuda(big x,big y,int f)
{ /* returns TRUE if x/y would fit flash format of length f */
    int n,d;
    n=(int)(x->len&(MR_OBITS));
    d=(int)(y->len&(MR_OBITS));
    if (n==1 && x->w[0]==1) n=0;
    if (d==1 && y->w[0]==1) d=0;
    if (n+d<=f) return TRUE;
    return FALSE;
}

#endif

__device__ int mr_lent_cuda(flash x)
{ /* return length of big or flash in words */
    mr_lentype lx;
    lx=(x->len&(MR_OBITS));
#ifdef MR_FLASH
    return (int)((lx&(MR_MSK))+((lx>>(MR_BTS))&(MR_MSK)));
#else
    return (int)lx;
#endif
}

__device__ void zero_cuda(flash x)
{ /* set big/flash number to zero_cuda */
    int i,n;
    mr_small *g_cuda;
    if (x==NULL) return;
#ifdef MR_FLASH
    n=mr_lent_cuda(x);
#else
    n=(x->len&MR_OBITS);
#endif
    g_cuda=x->w;

    for (i=0;i<n;i++)
        g_cuda[i]=0;

    x->len=0;
}

__device__ void uconvert_cuda(_MIPD_ unsigned int n ,big x)
{  /*  convert_cuda unsigned integer n to big number format  */
    int m;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zero_cuda(x);
    if (n==0) return;
    
    m=0;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
#if MR_IBITS > MIRACL
        while (n>0)
        {
            x->w[m++]=(mr_small)(n%((mr_small)1<<(MIRACL)));
            n/=((mr_small)1<<(MIRACL));
        }
#else
        x->w[m++]=(mr_small)n;
#endif
#endif
#ifndef MR_SIMPLE_BASE
    }
    else while (n>0)
    {
        x->w[m++]=MR_REMAIN((mr_small)n,mr_mip->base);
		n=(unsigned int)((mr_small)n/mr_mip->base);
    }
#endif
    x->len=m;
}

__device__ void tconvert_cuda(_MIPD_ mr_utype n,big x)
{
	mr_lentype s;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (n==0) {zero_cuda(x); return;}
    s=0;
    if (n<0)
    {
        s=MR_MSBIT;
        n=(-n);
    }
	x->w[0]=n;
	x->len=1;
    x->len|=s;
}

__device__ void convert_cuda(_MIPD_ int n ,big x)
{  /*  convert_cuda signed integer n to big number format  */
    mr_lentype s;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (n==0) {zero_cuda(x); return;}
    s=0;
    if (n<0)
    {
        s=MR_MSBIT;
        n=(-n);
    }
    uconvert_cuda(_MIPP_ (unsigned int)n,x);
    x->len|=s;
}

#ifndef MR_STATIC
#ifdef mr_dltype

__device__ void dlconv_cuda(_MIPD_ mr_dltype n,big x)
{ /* convert_cuda double length integer to big number format - rarely needed */
    int m;
    mr_lentype s;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zero_cuda(x);
    if (n==0) return;
    s=0;
    if (n<0)
    {
        s=MR_MSBIT;
        n=(-n);
    }
    m=0;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
        while (n>0)
        {
            x->w[m++]=(mr_small)(n%((mr_dltype)1<<(MIRACL)));
            n/=((mr_dltype)1<<(MIRACL));
        }
#endif
#ifndef MR_SIMPLE_BASE
    }    
    else while (n>0)
    {
        x->w[m++]=(mr_small)MR_REMAIN(n,mr_mip->base);
        n/=mr_mip->base;
    }
#endif
    x->len=(m|s);
}

#endif

__device__ void ulgconv_cuda(_MIPD_ unsigned long n,big x)
{ /* convert_cuda unsigned long integer to big number format - rarely needed */
    int m;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zero_cuda(x);
    if (n==0) return;

    m=0;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
#if MR_LBITS > MIRACL
        while (n>0)
        {
            x->w[m++]=(mr_small)(n%(1L<<(MIRACL)));
            n/=(1L<<(MIRACL));
        }
#else
        x->w[m++]=(mr_small)n;
#endif
#endif
#ifndef MR_SIMPLE_BASE
    }    
    else while (n>0)
    {
        x->w[m++]=MR_REMAIN(n,mr_mip->base);
		n=(unsigned long)((mr_small)n/mr_mip->base);
    }
#endif
    x->len=m;
}

__device__ void lgconv_cuda(_MIPD_ long n,big x)
{ /* convert_cuda signed long integer to big number format - rarely needed */
    mr_lentype s;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (n==0) {zero_cuda(x); return;}
    s=0;
    if (n<0)
    {
        s=MR_MSBIT;
        n=(-n);
    }
    ulgconv_cuda(_MIPP_ (unsigned long)n,x);

    x->len|=s;
}

__device__ flash mirvar_cuda(_MIPD_ int iv)
{ /* initialize big/flash number */
        flash x;
        int align;
        char *ptr;
    #ifdef MR_OS_THREADS
        miracl *mr_mip=get_mip_cuda();
    #endif

        if (mr_mip->ERNUM) return NULL;
        MR_IN(23);

        if (!(mr_mip->active))
        {
            mr_berror_cuda(_MIPP_ MR_ERR_NO_MIRSYS);
            MR_OUT
            return NULL;
        }

    /* OK, now I control alignment.... */

    /* Allocate space for big, the length, the pointer, and the array */
    /* Do it all in one memory allocation - this is quicker */
    /* Ensure that the array has correct alignment */

        x=(big)mr_alloc_cuda(_MIPP_ mr_size(mr_mip->nib-1),1);
        if (x==NULL)
        {
            MR_OUT
            return x;
        }

        ptr=(char *)&x->w;
        align=(unsigned long)(ptr+sizeof(mr_small *))%sizeof(mr_small);

        x->w=(mr_small *)(ptr+sizeof(mr_small *)+sizeof(mr_small)-align);

        if (iv!=0) convert_cuda(_MIPP_ iv,x);
        MR_OUT
        return x;
}

#endif

__device__ flash mirvar_mem_variable_cuda(char *mem,int index,int sz)
{
    flash x;
    int align;
    char *ptr;
    int offset,r;

/* alignment */
    offset=0;
    r=(unsigned long)mem%MR_SL;
    if (r>0) offset=MR_SL-r;

    x=(big)&mem[offset+mr_size(sz)*index];
    ptr=(char *)&x->w;
    align=(unsigned long)(ptr+sizeof(mr_small *))%sizeof(mr_small);   
    x->w=(mr_small *)(ptr+sizeof(mr_small *)+sizeof(mr_small)-align);   

    return x;
}

__device__ flash mirvar_mem_cuda(_MIPD_ char *mem,int index)
{ /* initialize big/flash number from pre-allocated memory */
 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
    if (mr_mip->ERNUM) return NULL;

    return mirvar_mem_variable_cuda(mem,index,mr_mip->nib-1);

}

__device__ void set_user_function_cuda(_MIPD_ BOOL (*user)(void))
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(111)

    if (!(mr_mip->active))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NO_MIRSYS);
        MR_OUT
        return;
    }

    mr_mip->user=user;

    MR_OUT
}

#ifndef MR_STATIC

#ifndef MR_SIMPLE_IO

__device__ void set_io_buffer_size_cuda(_MIPD_ int len)
{
    int i;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (len<0) return;
    MR_IN(142)
    for (i=0;i<mr_mip->IOBSIZ;i++) mr_mip->IOBUFF[i]=0;
    mr_free_cuda(mr_mip->IOBUFF);
    if (len==0) 
    {
        MR_OUT
        return;
    }
    mr_mip->IOBSIZ=len;
    mr_mip->IOBUFF=(char *)mr_alloc_cuda(_MIPP_ len+1,1);
    mr_mip->IOBUFF[0]='\0';
    MR_OUT
}
#endif

#endif

/* Initialise a big from ROM given its fixed length */

__device__ BOOL init_big_from_rom_cuda(big x,int len,const mr_small *rom,int romsize,int *romptr)
{
    int i;
    zero_cuda(x);
    x->len=len;
    for (i=0;i<len;i++)
    {
        if (*romptr>=romsize) return FALSE;
#ifdef MR_AVR
        x->w[i]=pgm_read_byte_near(&rom[*romptr]);
#else
        x->w[i]=rom[*romptr];
#endif
        (*romptr)++;
    }

    mr_lzero_cuda(x);
    return TRUE;
}

/* Initialise an elliptic curve point from ROM */

__device__ BOOL init_point_from_rom_cuda(epoint *P,int len,const mr_small *rom,int romsize,int *romptr)
{
    if (!init_big_from_rom_cuda(P->X,len,rom,romsize,romptr)) return FALSE;
    if (!init_big_from_rom_cuda(P->Y,len,rom,romsize,romptr)) return FALSE;
    P->marker=MR_EPOINT_NORMALIZED;
    return TRUE;
}

#ifdef MR_GENERIC_AND_STATIC
__device__ miracl *mirsys_cuda(miracl *mr_mip,int nd,mr_small nb)
#else
__device__ miracl *mirsys_cuda(int nd,mr_small nb)
#endif
{  /*  Initialize MIRACL system to   *
    *  use numbers to base nb, and   *
    *  nd digits or (-nd) bytes long */

/* In these cases mr_mip is passed as the first parameter */

#ifdef MR_GENERIC_AND_STATIC
	return mirsys_basic_cuda(mr_mip,nd,nb);
#endif

#ifdef MR_GENERIC_MT
#ifndef MR_STATIC
	miracl *mr_mip=mr_first_alloc_cuda();
    return mirsys_basic_cuda(mr_mip,nd,nb);
#endif
#endif
/* In these cases mr_mip is a "global" pointer and the mip itself is allocated from the heap. 
   In fact mr_mip (and mip) may be thread specific if some multi-threading scheme is implemented */
#ifndef MR_STATIC
 #ifdef MR_WINDOWS_MT
    miracl *mr_mip=mr_first_alloc_cuda();
    TlsSetValue(mr_key,mr_mip);
 #endif

 #ifdef MR_UNIX_MT
    miracl *mr_mip=mr_first_alloc_cuda(); 
    pthread_setspecific(mr_key,mr_mip);    
 #endif

 #ifdef MR_OPENMP_MT
    mr_mip=mr_first_alloc_cuda(); 
 #endif

 #ifndef MR_WINDOWS_MT
   #ifndef MR_UNIX_MT
     #ifndef MR_OPENMP_MT
       mr_mip=mr_first_alloc_cuda();
     #endif
   #endif
 #endif
#endif

#ifndef MR_GENERIC_MT
    mr_mip=get_mip_cuda();
#endif
    return mirsys_basic_cuda(mr_mip,nd,nb);
}

__device__ miracl *mirsys_basic_cuda(miracl *mr_mip,int nd,mr_small nb)
{
#ifndef MR_NO_RAND
    int i;
#endif
   
    mr_small b,nw;
#ifdef MR_FP
    mr_small dres;
#endif

    if (mr_mip==NULL) return NULL;

#ifndef MR_STRIPPED_DOWN
    mr_mip->depth=0;
    mr_mip->trace[0]=0;
    mr_mip->depth++;
    mr_mip->trace[mr_mip->depth]=29;
#endif           
                    /* digest hardware configuration */

#ifdef MR_NO_STANDARD_IO
    mr_mip->ERCON=TRUE;
#else
    mr_mip->ERCON=FALSE;
#endif
#ifndef MR_STATIC
    mr_mip->logN=0;
    mr_mip->degree=0;
    mr_mip->chin.NP=0;
#endif


    mr_mip->user=NULL;
    mr_mip->same=FALSE;
    mr_mip->first_one=FALSE;
    mr_mip->debug=FALSE;
	mr_mip->AA=0;
#ifndef MR_AFFINE_ONLY
    mr_mip->coord=MR_NOTSET;
#endif

#ifdef MR_NOFULLWIDTH
    if (nb==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_BASE);
        MR_OUT
        return mr_mip;
    }
#endif

#ifndef MR_FP
#ifdef mr_dltype
#ifndef MR_NOFULLWIDTH
    if (sizeof(mr_dltype)<2*sizeof(mr_utype))
    { /* double length type, isn't */
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_DOUBLE_LEN);
        MR_OUT
        return mr_mip;
    }
#endif
#endif
#endif

    if (nb==1 || nb>MAXBASE)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_BASE);
        MR_OUT
        return mr_mip;
    }

#ifdef MR_FP_ROUNDING
    if (mr_setbase_cuda(_MIPP_ nb)==0)
    { /* unable in fact to control FP rounding */
        mr_berror_cuda(_MIPP_ MR_ERR_NO_ROUNDING);
        MR_OUT
        return mr_mip;
    }
#else
    mr_setbase_cuda(_MIPP_ nb);
#endif

    b=mr_mip->base;

#ifdef MR_SIMPLE_BASE
    if (b!=0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_BASE);
        MR_OUT
        return mr_mip;
    }
#endif

    mr_mip->lg2b=0;
    mr_mip->base2=1;
#ifndef MR_SIMPLE_BASE
    if (b==0)
    {
#endif
        mr_mip->lg2b=MIRACL;
        mr_mip->base2=0;
#ifndef MR_SIMPLE_BASE
    }
    else while (b>1)
    {
        b=MR_DIV(b,2);
        mr_mip->lg2b++;
        mr_mip->base2*=2;
    }
#endif

#ifdef MR_ALWAYS_BINARY
    if (mr_mip->base!=mr_mip->base2) 
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_BINARY);
        MR_OUT
        return mr_mip;
    }
#endif

/* calculate total space for bigs */
/*

 big -> |int len|small *ptr| alignment space | size_cuda in words +1| alignment up to multiple of 4 | 


*/
    if (nd>0) nw=MR_ROUNDUP(nd,mr_mip->pack);
    else      nw=MR_ROUNDUP(8*(-nd),mr_mip->lg2b);

    if (nw<1) nw=1;
    mr_mip->nib=(int)(nw+1);   /* add_cuda one extra word for small overflows */

#ifdef MR_STATIC
    if (nw>MR_STATIC)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_TOO_BIG);
        MR_OUT
        return mr_mip;
    }
#endif

   /* mr_mip->nib=(int)(nw+1);    add_cuda one extra word for small overflows */     

#ifdef MR_FLASH
    mr_mip->workprec=mr_mip->nib;
    mr_mip->stprec=mr_mip->nib;
    while (mr_mip->stprec>2 && mr_mip->stprec>MR_FLASH/mr_mip->lg2b) 
        mr_mip->stprec=(mr_mip->stprec+1)/2;
    if (mr_mip->stprec<2) mr_mip->stprec=2;
   
#endif

#ifndef MR_DOUBLE_BIG
    mr_mip->check=ON;
#else
    mr_mip->check=OFF;
#endif

#ifndef MR_SIMPLE_BASE
#ifndef MR_SIMPLE_IO
    mr_mip->IOBASE=10;   /* defaults */
#endif
#endif
    mr_mip->ERNUM=0;
    
    mr_mip->NTRY=6;
    mr_mip->MONTY=ON;
#ifdef MR_FLASH
    mr_mip->EXACT=TRUE;
    mr_mip->RPOINT=OFF;
#endif
#ifndef MR_STRIPPED_DOWN
    mr_mip->TRACER=OFF;
#endif

#ifndef MR_SIMPLE_IO
    mr_mip->INPLEN=0;
    mr_mip->IOBSIZ=MR_DEFAULT_BUFFER_SIZE;
#endif

#ifdef MR_STATIC
    mr_mip->PRIMES=mr_small_primes;
#else
    mr_mip->PRIMES=NULL;
#ifndef MR_SIMPLE_IO
    mr_mip->IOBUFF=(char *)mr_alloc_cuda(_MIPP_ MR_DEFAULT_BUFFER_SIZE+1,1);
#endif
#endif
#ifndef MR_SIMPLE_IO
    mr_mip->IOBUFF[0]='\0';
#endif
    mr_mip->qnr=0;
    mr_mip->cnr=0;
    mr_mip->TWIST=0;
    mr_mip->pmod8=0;
	mr_mip->pmod9=0;

/* quick start for rng. irand(.) should be called first before serious use.. */

#ifndef MR_NO_RAND
    mr_mip->ira[0]=0x55555555;
    mr_mip->ira[1]=0x12345678;

    for (i=2;i<NK;i++) 
        mr_mip->ira[i]=mr_mip->ira[i-1]+mr_mip->ira[i-2]+0x1379BDF1;
    mr_mip->rndptr=NK;
    mr_mip->borrow=0;
#endif

    mr_mip->nib=2*mr_mip->nib+1;
#ifdef MR_FLASH
    if (mr_mip->nib!=(mr_mip->nib&(MR_MSK)))
#else
    if (mr_mip->nib!=(int)(mr_mip->nib&(MR_OBITS)))
#endif
    {
        mr_berror_cuda(_MIPP_ MR_ERR_TOO_BIG);
        mr_mip->nib=(mr_mip->nib-1)/2;
        MR_OUT
        return mr_mip;
    }
#ifndef MR_STATIC
    mr_mip->workspace=(char *)memalloc_cuda(_MIPP_ MR_SPACES);  /* grab workspace */
#else
    memset(mr_mip->workspace,0,MR_BIG_RESERVE(MR_SPACES));
#endif

    mr_mip->M=0;
    mr_mip->fin=FALSE;
    mr_mip->fout=FALSE;
    mr_mip->active=ON;
    
    mr_mip->nib=(mr_mip->nib-1)/2;

/* allocate memory for workspace variables */
   
#ifndef MR_DOUBLE_BIG

    mr_mip->w0=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,0);  /* double length */
    mr_mip->w1=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,2);
    mr_mip->w2=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,3);
    mr_mip->w3=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,4);
    mr_mip->w4=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,5);
    mr_mip->w5=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,6);  /* double length */
    mr_mip->w6=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,8);  /* double length */
    mr_mip->w7=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,10); /* double length */
    mr_mip->w8=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,12);
    mr_mip->w9=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,13);
    mr_mip->w10=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,14);
    mr_mip->w11=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,15);
    mr_mip->w12=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,16);
    mr_mip->w13=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,17);
    mr_mip->w14=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,18);
    mr_mip->w15=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,19);
    mr_mip->sru=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,20);
    mr_mip->modulus=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,21);
    mr_mip->pR=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,22); /* double length */
    mr_mip->A=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,24);
    mr_mip->B=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,25);
    mr_mip->one=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,26);
#ifdef MR_KCM
    mr_mip->big_ndash=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,27);
    mr_mip->ws=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,28);
    mr_mip->wt=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,29); /* double length */
#endif
#ifdef MR_FLASH
#ifdef MR_KCM
    mr_mip->pi=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,31);
#else
    mr_mip->pi=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,27);
#endif
#endif

#else
/* w0-w7 are double normal length */
    mr_mip->w0=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,0);  /* quad length */
    mr_mip->w1=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,4);  /* double length */
    mr_mip->w2=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,6);
    mr_mip->w3=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,8);
    mr_mip->w4=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,10);
    mr_mip->w5=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,12);  /* quad length */
    mr_mip->w6=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,16);  /* quad length */
    mr_mip->w7=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,20);  /* quad length */
    mr_mip->w8=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,24);

    mr_mip->w9=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,25);
    mr_mip->w10=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,26);
    mr_mip->w11=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,27);
    mr_mip->w12=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,28);
    mr_mip->w13=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,29);
    mr_mip->w14=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,30);
    mr_mip->w15=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,31);
    mr_mip->sru=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,32);
    mr_mip->modulus=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,33);
    mr_mip->pR=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,34); /* double length */
    mr_mip->A=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,36);
    mr_mip->B=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,37);
    mr_mip->one=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,38);
#ifdef MR_KCM
    mr_mip->big_ndash=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,39);
    mr_mip->ws=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,40);
    mr_mip->wt=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,41); /* double length */
#endif
#ifdef MR_FLASH
#ifdef MR_KCM
    mr_mip->pi=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,43);
#else
    mr_mip->pi=mirvar_mem_cuda(_MIPP_ mr_mip->workspace,39);
#endif
#endif

#endif
    MR_OUT
    return mr_mip;
} 

#ifndef MR_STATIC

/* allocate space for a number of bigs from the heap */

__device__ void *memalloc_cuda(_MIPD_ int num)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    return mr_alloc_cuda(_MIPP_ mr_big_reserve(num,mr_mip->nib-1),1);
}

#endif

__device__ void memkill_cuda(_MIPD_ char *mem,int len)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mem==NULL) return;
    memset(mem,0,mr_big_reserve(len,mr_mip->nib-1));
#ifndef MR_STATIC
    mr_free_cuda(mem);
#endif
}

#ifndef MR_STATIC

__device__ void mirkill_cuda(big x)
{ /* kill a big/flash variable, that is set it to zero_cuda
     and free its memory */
    if (x==NULL) return;
    zero_cuda(x);
    mr_free_cuda(x);
}

#endif

__device__ void mirexit_cuda(_MIPDO_ )
{ /* clean up after miracl */

    int i;
#ifdef MR_WINDOWS_MT
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_UNIX_MT
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_OPENMP_MT
    miracl *mr_mip=get_mip_cuda();
#endif
    mr_mip->ERCON=FALSE;
    mr_mip->active=OFF;
    memkill_cuda(_MIPP_ mr_mip->workspace,MR_SPACES);
#ifndef MR_NO_RAND
    for (i=0;i<NK;i++) mr_mip->ira[i]=0L;
#endif
#ifndef MR_STATIC
#ifndef MR_SIMPLE_IO
    set_io_buffer_size_cuda(_MIPP_ 0);
#endif
    if (mr_mip->PRIMES!=NULL) mr_free_cuda(mr_mip->PRIMES);
#else
#ifndef MR_SIMPLE_IO
    for (i=0;i<=MR_DEFAULT_BUFFER_SIZE;i++)
        mr_mip->IOBUFF[i]=0;
#endif
#endif

#ifndef MR_STATIC
    mr_free_cuda(mr_mip);
#ifdef MR_WINDOWS_MT
	TlsSetValue(mr_key, NULL);		/* Thank you Thales */
#endif
#endif

#ifndef MR_GENERIC_MT
#ifndef MR_WINDOWS_MT
#ifndef MR_UNIX_MT
#ifndef MR_STATIC
    mr_mip=NULL;
#endif
#endif   
#endif   
#endif  
    
#ifdef MR_OPENMP_MT
    mr_mip=NULL;
#endif

}

__device__ int exsign_cuda(flash x)
{ /* extract sign of big/flash number */
    if ((x->len&(MR_MSBIT))==0) return PLUS;
    else                        return MINUS;    
}

__device__ void insign_cuda(int s,flash x)
{  /* assert sign of big/flash number */
    if (x->len==0) return;
    if (s<0) x->len|=MR_MSBIT;
    else     x->len&=MR_OBITS;
}   

__device__ void mr_lzero_cuda(big x)
{  /*  strip leading zeros from big number  */
    mr_lentype s;
    int m;
    s=(x->len&(MR_MSBIT));
    m=(int)(x->len&(MR_OBITS));
    while (m>0 && x->w[m-1]==0)
        m--;
    x->len=m;
    if (m>0) x->len|=s;
}

#ifndef MR_SIMPLE_IO

__device__ int getdig_cuda(_MIPD_ big x,int i)
{  /* extract a packed digit */
    int k;
    mr_small n;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    i--;
    n=x->w[i/mr_mip->pack];

    if (mr_mip->pack==1) return (int)n;
    k=i%mr_mip->pack;
    for (i=1;i<=k;i++)
        n=MR_DIV(n,mr_mip->apbase);  
    return (int)MR_REMAIN(n,mr_mip->apbase);
}

__device__ int numdig_cuda(_MIPD_ big x)
{  /* returns number of digits in x */
    int nd;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (x->len==0) return 0;

    nd=(int)(x->len&(MR_OBITS))*mr_mip->pack;
    while (getdig_cuda(_MIPP_ x,nd)==0)
        nd--;
    return nd;
} 

__device__ void putdig_cuda(_MIPD_ int n,big x,int i)
{  /* insert a digit into a packed word */
    int j,k,lx;
    mr_small m,p;
    mr_lentype s;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(26)

    s=(x->len&(MR_MSBIT));
    lx=(int)(x->len&(MR_OBITS));
    m=getdig_cuda(_MIPP_ x,i);
    p=n;
    i--;
    j=i/mr_mip->pack;
    k=i%mr_mip->pack;
    for (i=1;i<=k;i++)
    {
        m*=mr_mip->apbase;
        p*=mr_mip->apbase;
    }
    if (j>=mr_mip->nib && (mr_mip->check || j>=2*mr_mip->nib))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
        MR_OUT
        return;
    }

    x->w[j]=(x->w[j]-m)+p;
    if (j>=lx) x->len=((j+1)|s);
    mr_lzero_cuda(x);
    MR_OUT
}

#endif

#ifndef MR_FP

__device__ void mr_and_cuda(big x,big y,big z)
{ /* z= bitwise logical AND of x and y */
    int i,nx,ny,nz,nr;
    if (x==y) 
    {
        copy_cuda(x,z);
        return;
    }

#ifdef MR_FLASH
    nx=mr_lent_cuda(x);
    ny=mr_lent_cuda(y);
    nz=mr_lent_cuda(z);
#else
    ny=(y->len&(MR_OBITS));
    nx=(x->len&(MR_OBITS));
    nz=(z->len&(MR_OBITS));
#endif
    if (ny<nx) nr=ny;
    else       nr=nx;

    for (i=0;i<nr;i++)
        z->w[i]=x->w[i]&y->w[i];
    for (i=nr;i<nz;i++) 
        z->w[i]=0;
    z->len=nr;
	mr_lzero_cuda(z);
}

__device__ void mr_xor_cuda(big x,big y,big z)
{ 
     int i,nx,ny,nz,nr;
     if (x==y)
     {
         copy_cuda(x,z);
         return;
     }

#ifdef MR_FLASH
     nx=mr_lent_cuda(x);
     ny=mr_lent_cuda(y);
     nz=mr_lent_cuda(z);
#else
     ny=(y->len&(MR_OBITS));
     nx=(x->len&(MR_OBITS));
     nz=(z->len&(MR_OBITS));
#endif
     if (ny<nx) nr=nx;
     else       nr=ny;

     for (i=0;i<nr;i++)
         z->w[i]=x->w[i]^y->w[i];
     for (i=nr;i<nz;i++)
         z->w[i]=0;
     z->len=nr;
	 mr_lzero_cuda(z);
}

#endif

__device__ void copy_cuda(flash x,flash y)
{  /* copy_cuda x to y: y=x  */
    int i,nx,ny;
    mr_small *gx,*gy;
    if (x==y || y==NULL) return;

    if (x==NULL)
    { 
        zero_cuda(y);
        return;
    }

#ifdef MR_FLASH    
    ny=mr_lent_cuda(y);
    nx=mr_lent_cuda(x);
#else
    ny=(y->len&(MR_OBITS));
    nx=(x->len&(MR_OBITS));
#endif

    gx=x->w;
    gy=y->w;

    for (i=nx;i<ny;i++)
        gy[i]=0;
    for (i=0;i<nx;i++)
        gy[i]=gx[i];
    y->len=x->len;

}

__device__ void negify_cuda(flash x,flash y)
{ /* negate a big/flash variable: y=-x */
    copy_cuda(x,y);
    if (y->len!=0) y->len^=MR_MSBIT;
}

__device__ void absol_cuda(flash x,flash y)
{ /* y=abs(x) */
    copy_cuda(x,y);
    y->len&=MR_OBITS;
}

__device__ BOOL mr_notint_cuda(flash x)
{ /* returns TRUE if x is Flash */
#ifdef MR_FLASH
    if ((((x->len&(MR_OBITS))>>(MR_BTS))&(MR_MSK))!=0) return TRUE;
#endif
    return FALSE;
}

__device__ void mr_shift_cuda(_MIPD_ big x,int n,big w)
{ /* set w=x.(mr_base^n) by shifting */
    mr_lentype s;
    int i,bl;
    mr_small *gw=w->w;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    copy_cuda(x,w);
    if (w->len==0 || n==0) return;
    MR_IN(33)

    if (mr_notint_cuda(w)) mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
    s=(w->len&(MR_MSBIT));
    bl=(int)(w->len&(MR_OBITS))+n;
    if (bl<=0)
    {
        zero_cuda(w);
        MR_OUT
        return;
    }
    if (bl>mr_mip->nib && mr_mip->check) mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
    if (mr_mip->ERNUM)
    {
        MR_OUT
        return;
    }
    if (n>0)
    {
        for (i=bl-1;i>=n;i--)
            gw[i]=gw[i-n];
        for (i=0;i<n;i++)
            gw[i]=0;
    }
    else
    {
        n=(-n);
        for (i=0;i<bl;i++)
            gw[i]=gw[i+n];
        for (i=0;i<n;i++)
            gw[bl+i]=0;
    }
    w->len=(bl|s);
    MR_OUT
}

__device__ int size_cuda(big x)
{  /*  get size_cuda of big number;  convert_cuda to *
    *  integer - if possible               */
    int n,m;
    mr_lentype s;
    if (x==NULL) return 0;
    s=(x->len&MR_MSBIT);
    m=(int)(x->len&MR_OBITS);
    if (m==0) return 0;
    if (m==1 && x->w[0]<(mr_small)MR_TOOBIG) n=(int)x->w[0];
    else                                     n=MR_TOOBIG;
    if (s==MR_MSBIT) return (-n);
    return n;
}

__device__ int mr_compare_cuda(big x,big y)
{  /* compare x and y: =1 if x>y  =-1 if x<y *
    *  =0 if x=y                             */
    int m,n,sig;
    mr_lentype sx,sy;
    if (x==y) return 0;
    sx=(x->len&MR_MSBIT);
    sy=(y->len&MR_MSBIT);
    if (sx==0) sig=PLUS;
    else       sig=MINUS;
    if (sx!=sy) return sig;
    m=(int)(x->len&MR_OBITS);
    n=(int)(y->len&MR_OBITS);
    if (m>n) return sig;
    if (m<n) return -sig;
    while (m>0)
    { /* check digit by digit */
        m--;  
        if (x->w[m]>y->w[m]) return sig;
        if (x->w[m]<y->w[m]) return -sig;
    }
    return 0;
}

#ifdef MR_FLASH

__device__ void fpack_cuda(_MIPD_ big n,big d,flash x)
{ /* create floating-slash number x=n/d from *
   * big integer numerator and denominator   */
    mr_lentype s;
    int i,ld,ln;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(31)

    ld=(int)(d->len&MR_OBITS);
    if (ld==0) mr_berror_cuda(_MIPP_ MR_ERR_FLASH_OVERFLOW);
    if (ld==1 && d->w[0]==1) ld=0;
    if (x==d) mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
    if (mr_notint_cuda(n) || mr_notint_cuda(d)) mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
    s=(n->len&MR_MSBIT);
    ln=(int)(n->len&MR_OBITS);
    if (ln==1 && n->w[0]==1) ln=0;
    if ((ld+ln>mr_mip->nib) && (mr_mip->check || ld+ln>2*mr_mip->nib)) 
        mr_berror_cuda(_MIPP_ MR_ERR_FLASH_OVERFLOW);
    if (mr_mip->ERNUM)
    {
       MR_OUT
       return;
    }
    copy_cuda(n,x);
    if (n->len==0)
    {
        MR_OUT
        return;
    }
    s^=(d->len&MR_MSBIT);
    if (ld==0)
    {
        if (x->len!=0) x->len|=s;
        MR_OUT
        return;
    }
    for (i=0;i<ld;i++)
        x->w[ln+i]=d->w[i];
    x->len=(s|(ln+((mr_lentype)ld<<MR_BTS)));
    MR_OUT
}

__device__ void numer_cuda(_MIPD_ flash x,big y)
{ /* extract numerator of x */
    int i,ln,ld;
    mr_lentype s,ly;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    if (mr_notint_cuda(x))
    {
        s=(x->len&MR_MSBIT);
        ly=(x->len&MR_OBITS);
        ln=(int)(ly&MR_MSK);
        if (ln==0)
        {
            if(s==MR_MSBIT) convert_cuda(_MIPP_ (-1),y);
            else            convert_cuda(_MIPP_ 1,y);
            return;
        }
        ld=(int)((ly>>MR_BTS)&MR_MSK);
        if (x!=y)
        {
            for (i=0;i<ln;i++) y->w[i]=x->w[i];
            for (i=ln;i<mr_lent_cuda(y);i++) y->w[i]=0;
        }
        else for (i=0;i<ld;i++) y->w[ln+i]=0;
        y->len=(ln|s);
    }
    else copy_cuda(x,y);
}

__device__ void denom_cuda(_MIPD_ flash x,big y)
{ /* extract denominator of x */
    int i,ln,ld;
    mr_lentype ly;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    if (!mr_notint_cuda(x))
    {
        convert_cuda(_MIPP_ 1,y);
        return;
    }
    ly=(x->len&MR_OBITS);
    ln=(int)(ly&MR_MSK);
    ld=(int)((ly>>MR_BTS)&MR_MSK);
    for (i=0;i<ld;i++)
        y->w[i]=x->w[ln+i];
    if (x==y) for (i=0;i<ln;i++) y->w[ld+i]=0;
    else for (i=ld;i<mr_lent_cuda(y);i++) y->w[i]=0;
    y->len=ld;
}

#endif

__device__ unsigned int igcd_cuda(unsigned int x,unsigned int y)
{ /* integer GCD, returns GCD of x and y */
    unsigned int r;
    if (y==0) return x;
    while ((r=x%y)!=0)
        x=y,y=r;
    return y;
}

__device__ unsigned long lgcd_cuda(unsigned long x,unsigned long y)
{ /* long GCD, returns GCD of x and y */
    unsigned long r;
    if (y==0) return x;
    while ((r=x%y)!=0)
        x=y,y=r;
    return y;
}

__device__ unsigned int isqrt_cuda(unsigned int num,unsigned int guess)
{ /* square root of an integer */
    unsigned int sqr;
    unsigned int oldguess=guess;
    if (num==0) return 0;
    if (num<4) return 1;
  
    for (;;)
    { /* Newtons iteration */
     /*   sqr=guess+(((num/guess)-guess)/2); */
        sqr=((num/guess)+guess)/2;
        if (sqr==guess || sqr==oldguess) 
        {
            if (sqr*sqr>num) sqr--;
            return sqr;
        }
        oldguess=guess;
        guess=sqr;
    }
}

__device__ unsigned long mr_lsqrt_cuda(unsigned long num,unsigned long guess)
{ /* square root of a long */
    unsigned long sqr;
    unsigned long oldguess=guess;
    if (num==0) return 0;
    if (num<4) return 1;
  
    for (;;)
    { /* Newtons iteration */
     /*   sqr=guess+(((num/guess)-guess)/2); */
        sqr=((num/guess)+guess)/2;
        if (sqr==guess || sqr==oldguess) 
        {
            if (sqr*sqr>num) sqr--;
            return sqr;
        }
        oldguess=guess;
        guess=sqr;
    }
}

__device__ mr_small sgcd_cuda(mr_small x,mr_small y)
{ /* integer GCD, returns GCD of x and y */
    mr_small r;
#ifdef MR_FP
    mr_small dres;
#endif
    if (y==(mr_small)0) return x;
    while ((r=MR_REMAIN(x,y))!=(mr_small)0)
        x=y,y=r;
    return y;
}

/* routines to support sliding-windows exponentiation *
 * in various contexts */

__device__ int mr_testbit_cuda(_MIPD_ big x,int n)
{ /* return value of n-th bit of big */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_FP
    mr_small m,a,dres;
    m=mr_shiftbits_cuda((mr_small)1,n%mr_mip->lg2b);

    a=x->w[n/mr_mip->lg2b];

    a=MR_DIV(a,m); 

    if ((MR_DIV(a,2.0)*2.0) != a) return 1;
#else
    if ((x->w[n/mr_mip->lg2b] & ((mr_small)1<<(n%mr_mip->lg2b))) >0) return 1;
#endif
    return 0;
}

__device__ void mr_addbit_cuda(_MIPD_ big x,int n)
{ /* add_cuda 2^n to positive x - where you know that bit is zero_cuda. Use with care! */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    mr_lentype m=n/mr_mip->lg2b;
    x->w[m]+=mr_shiftbits_cuda((mr_small)1,n%mr_mip->lg2b);
    if (x->len<m+1) x->len=m+1;
}

__device__ int recode_cuda(_MIPD_ big e,int t,int w,int i)
{ /* recode_cuda exponent for Comb method */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    int j,r;
    r=0;
    for (j=w-1;j>=0;j--)
    {
        r<<=1;
        r|=mr_testbit_cuda(_MIPP_ e,i+j*t);
    }
    return r;
}

__device__ int mr_window_cuda(_MIPD_ big x,int i,int *nbs,int * nzs,int window_size)
{ /* returns sliding window value, max. of 5 bits,         *
   * (Note from version 5.23 this can be changed by        *
   * setting parameter window_size. This can be            *
   * a useful space-saver) starting at i-th bit of big x.  *
   * nbs is number of bits processed, nzs is the number of *
   * additional trailing zeros detected. Returns valid bit *
   * pattern 1x..x1 with no two adjacent 0's. So 10101     *
   * will return 21 with nbs=5, nzs=0. 11001 will return 3,*
   * with nbs=2, nzs=2, having stopped after the first 11..*/
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    int j,r,w;
    w=window_size;

/* check for leading 0 bit */

    *nbs=1;
    *nzs=0;
    if (!mr_testbit_cuda(_MIPP_ x,i)) return 0;

/* adjust window size_cuda if not enough bits left */
   
    if (i-w+1<0) w=i+1;

    r=1;
    for (j=i-1;j>i-w;j--)
    { /* accumulate bits. Abort if two 0's in a row */
        (*nbs)++;
        r*=2;
        if (mr_testbit_cuda(_MIPP_ x,j)) r+=1;
        if (r%4==0)
        { /* oops - too many zeros - shorten window */
            r/=4;
            *nbs-=2;
            *nzs=2;
            break;
        }
    }
    if (r%2==0)
    { /* remove trailing 0 */
        r/=2;
        *nzs=1;
        (*nbs)--;
    }
    return r;
}

__device__ int mr_window2_cuda(_MIPD_ big x,big y,int i,int *nbs,int *nzs)
{ /* two bit window for double exponentiation */
    int r,w;
    BOOL a,b,c,d;
    w=2;
    *nbs=1;
    *nzs=0;

/* check for two leading 0's */
    a=mr_testbit_cuda(_MIPP_ x,i); b=mr_testbit_cuda(_MIPP_ y,i);

    if (!a && !b) return 0;
    if (i<1) w=1;

    if (a)
    {
        if (b) r=3;
        else   r=2;
    }
    else r=1;
    if (w==1) return r;

    c=mr_testbit_cuda(_MIPP_ x,i-1); d=mr_testbit_cuda(_MIPP_ y,i-1);

    if (!c && !d) 
    {
        *nzs=1;
        return r;
    }

    *nbs=2;
    r*=4;
    if (c)
    {
        if (d) r+=3;
        else   r+=2;
    }
    else r+=1;
    return r;
}

__device__ int mr_naf_window_cuda(_MIPD_ big x,big x3,int i,int *nbs,int *nzs,int store)
{ /* returns sliding window value, using fractional windows   *
   * where "store" precomputed values are precalulated and    *
   * stored. Scanning starts at the i-th bit of  x. nbs is    *
   * the number of bits processed. nzs is number of           *
   * additional trailing zeros detected. x and x3 (which is   *
   * 3*x) are combined to produce the NAF (non-adjacent       *
   * form). So if x=11011(27) and x3 is 1010001, the LSB is   *
   * ignored and the value 100T0T (32-4-1=27) processed,      *
   * where T is -1. Note x.P = (3x-x)/2.P. This value will    *
   * return +7, with nbs=4 and nzs=1, having stopped after    *
   * the first 4 bits. If it goes too far, it must backtrack  *
   * Note in an NAF non-zero_cuda elements are never side by side, *
   * so 10T10T won't happen. NOTE: return value n zero_cuda or     * 
   * odd, -21 <= n <= +21     */

    int nb,j,r,biggest;

 /* get first bit */
    nb=mr_testbit_cuda(_MIPP_ x3,i)-mr_testbit_cuda(_MIPP_ x,i);

    *nbs=1;
    *nzs=0;
    if (nb==0) return 0;
    if (i==0) return nb;

    biggest=2*store-1;

    if (nb>0) r=1;
    else      r=(-1);

    for (j=i-1;j>0;j--)
    {
        (*nbs)++;
        r*=2;
        nb=mr_testbit_cuda(_MIPP_ x3,j)-mr_testbit_cuda(_MIPP_ x,j);
        if (nb>0) r+=1;
        if (nb<0) r-=1;
        if (abs(r)>biggest) break;
    }

    if (r%2!=0 && j!=0)
    { /* backtrack */
        if (nb>0) r=(r-1)/2;
        if (nb<0) r=(r+1)/2;
        (*nbs)--;
    }
    
    while (r%2==0)
    { /* remove trailing zeros */
        r/=2;
        (*nzs)++;
        (*nbs)--;
    }     
    return r;
}

/* Some general purpose elliptic curve stuff */

__device__ BOOL point_at_infinity_cuda(epoint *p)
{
    if (p==NULL) return FALSE;
    if (p->marker==MR_EPOINT_INFINITY) return TRUE;
    return FALSE;
}

#ifndef MR_STATIC

__device__ epoint* epoint_init_cuda(_MIPDO_ )
{ /* initialise epoint to general point at infinity. */
    epoint *p;
    char *ptr;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return NULL;

    MR_IN(96)

/* Create space for whole structure in one heap access */ 

    p=(epoint *)mr_alloc_cuda(_MIPP_ mr_esize(mr_mip->nib-1),1);

    ptr=(char *)p+sizeof(epoint);
    p->X=mirvar_mem_cuda(_MIPP_ ptr,0);
    p->Y=mirvar_mem_cuda(_MIPP_ ptr,1);
#ifndef MR_AFFINE_ONLY
    p->Z=mirvar_mem_cuda(_MIPP_ ptr,2);
#endif
    p->marker=MR_EPOINT_INFINITY;

    MR_OUT

    return p;
}

#endif

__device__ epoint* epoint_init_mem_variable_cuda(_MIPD_ char *mem,int index,int sz)
{
    epoint *p;
    char *ptr;
    int offset,r;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    offset=0;
    r=(unsigned long)mem%MR_SL;
    if (r>0) offset=MR_SL-r;

#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
        p=(epoint *)&mem[offset+index*mr_esize_a(sz)];
    else
#endif
    p=(epoint *)&mem[offset+index*mr_esize(sz)];

    ptr=(char *)p+sizeof(epoint);
    p->X=mirvar_mem_variable_cuda(ptr,0,sz);
    p->Y=mirvar_mem_variable_cuda(ptr,1,sz);
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord!=MR_AFFINE) p->Z=mirvar_mem_variable_cuda(ptr,2,sz);
#endif
    p->marker=MR_EPOINT_INFINITY;
    return p;
}

__device__ epoint* epoint_init_mem_cuda(_MIPD_ char *mem,int index)
{ 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return NULL;

    return epoint_init_mem_variable_cuda(_MIPP_ mem,index,mr_mip->nib-1);
}

#ifndef MR_STATIC

/* allocate space for a number of epoints from the heap */

__device__ void *ecp_memalloc_cuda(_MIPD_ int num)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
        return mr_alloc_cuda(_MIPP_  mr_ecp_reserve_a(num,mr_mip->nib-1),1);
    else
#endif
        return mr_alloc_cuda(_MIPP_  mr_ecp_reserve(num,mr_mip->nib-1),1);
}

#endif

__device__ void ecp_memkill_cuda(_MIPD_ char *mem,int num)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mem==NULL) return;

#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
        memset(mem,0,mr_ecp_reserve_a(num,mr_mip->nib-1));
    else
#endif
        memset(mem,0,mr_ecp_reserve(num,mr_mip->nib-1));


#ifndef MR_STATIC
    mr_free_cuda(mem);
#endif
}

#ifndef MR_STATIC

__device__ void epoint_free_cuda(epoint *p)
{ /* clean up point */
 
    if (p==NULL) return;
    zero_cuda(p->X);
    zero_cuda(p->Y);
#ifndef MR_AFFINE_ONLY
    if (p->marker==MR_EPOINT_GENERAL) zero_cuda(p->Z);
#endif
    mr_free_cuda(p);
}      
#endif
#endif

#ifndef mralloc_c
#define mralloc_c

#include <stdlib.h>

#ifndef MR_STATIC

__device__ miracl *mr_first_alloc_cuda()
{
    return (miracl *)calloc_cuda(1,sizeof(miracl));
}

__device__ void *mr_alloc_cuda(_MIPD_ int num,int size_cuda)
{
    char *p; 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip==NULL) 
    {
        p=(char *)calloc_cuda(num,size_cuda);
        return (void *)p;
    }
 
    if (mr_mip->ERNUM) return NULL;

    p=(char *)calloc_cuda(num,size_cuda);
    if (p==NULL) mr_berror_cuda(_MIPP_ MR_ERR_OUT_OF_MEMORY);
    return (void *)p;

}

__device__ void mr_free_cuda(void *addr)
{
    if (addr==NULL) return;
    free(addr);
    return;
}

#endif


#endif

#ifndef mrmonty_c
#define mrmonty_C

#include <stdlib.h> 

#ifdef MR_FP
#include <math.h>
#endif

#ifdef MR_WIN64
#include <intrin.h>
#endif

#ifdef MR_COUNT_OPS
extern int fpc,fpa; 
#endif

#ifdef MR_CELL
extern void mod256(_MIPD_ big,big);
#endif

__device__ void kill_monty_cuda(_MIPDO_ )
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zero_cuda(mr_mip->modulus);
#ifdef MR_KCM
    zero_cuda(mr_mip->big_ndash);
#endif
}

__device__ mr_small prepare_monty_cuda(_MIPD_ big n)
{ /* prepare Montgomery modulus */ 
#ifdef MR_KCM
    int nl;
#endif
#ifdef MR_PENTIUM
    mr_small ndash;
    mr_small base;
    mr_small magic=13835058055282163712.0;   
    int control=0x1FFF;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return (mr_small)0;
/* Is it set-up already? */
    if (size_cuda(mr_mip->modulus)!=0)
        if (mr_compare_cuda(n,mr_mip->modulus)==0) return mr_mip->ndash;

    MR_IN(80)

    if (size_cuda(n)<=2) 
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_MODULUS);
        MR_OUT
        return (mr_small)0;
    }

    zero_cuda(mr_mip->w6);
    zero_cuda(mr_mip->w15);

/* set a small negative QNR (on the assumption that n is prime!) */
/* These defaults can be over-ridden                             */

/* Did you know that for p=2 mod 3, -3 is a QNR? */

    mr_mip->pmod8=remain_cuda(_MIPP_ n,8);
	
    switch (mr_mip->pmod8)
    {
    case 0:
    case 1:
    case 2:
    case 4:
    case 6:
        mr_mip->qnr=0;  /* none defined */
        break;
    case 3:
        mr_mip->qnr=-1;
        break;
    case 5:
        mr_mip->qnr=-2;
        break;
    case 7:
        mr_mip->qnr=-1;
        break;
    }
	mr_mip->pmod9=remain_cuda(_MIPP_ n,9);

	mr_mip->NO_CARRY=FALSE;
	if (n->w[n->len-1]>>M4 < 5) mr_mip->NO_CARRY=TRUE;

#ifdef MR_PENTIUM

mr_mip->ACTIVE=FALSE;
if (mr_mip->base!=0)
    if (MR_PENTIUM==n->len) mr_mip->ACTIVE=TRUE;
    if (MR_PENTIUM<0)
    {
        if (n->len<=(-MR_PENTIUM)) mr_mip->ACTIVE=TRUE;
        if (logb2_cuda(_MIPP_ n)%mr_mip->lg2b==0) mr_mip->ACTIVE=FALSE;
    }
#endif

#ifdef MR_DISABLE_MONTGOMERY
    mr_mip->MONTY=OFF;
#else
    mr_mip->MONTY=ON;
#endif

#ifdef MR_COMBA
    mr_mip->ACTIVE=FALSE;

    if (MR_COMBA==n->len && mr_mip->base==mr_mip->base2) 
    {
        mr_mip->ACTIVE=TRUE;
#ifdef MR_SPECIAL
        mr_mip->MONTY=OFF;      /* "special" modulus reduction */

#endif                          /* implemented in mrcomba.c    */
    }

#endif
    convert_cuda(_MIPP_ 1,mr_mip->one);
    if (!mr_mip->MONTY)
    { /* Montgomery arithmetic is turned off */
        copy_cuda(n,mr_mip->modulus);
        mr_mip->ndash=0;
        MR_OUT
        return (mr_small)0;
    }

#ifdef MR_KCM
  
/* test for base==0 & n->len=MR_KCM.2^x */

    mr_mip->ACTIVE=FALSE;
    if (mr_mip->base==0)
    {
        nl=(int)n->len;
        while (nl>=MR_KCM)
        {
            if (nl==MR_KCM)
            {
                mr_mip->ACTIVE=TRUE;
                break;
            }
            if (nl%2!=0) break;
            nl/=2;
        }
    }  
    if (mr_mip->ACTIVE)
    {
        mr_mip->w6->len=n->len+1;
        mr_mip->w6->w[n->len]=1;
        if (invmodp_cuda(_MIPP_ n,mr_mip->w6,mr_mip->w14)!=1)
        { /* problems */
            mr_berror_cuda(_MIPP_ MR_ERR_BAD_MODULUS);
            MR_OUT
            return (mr_small)0;
        }
    }
    else
    {
#endif
        mr_mip->w6->len=2;
        mr_mip->w6->w[0]=0;
        mr_mip->w6->w[1]=1;    /* w6 = base */
        mr_mip->w15->len=1;
        mr_mip->w15->w[0]=n->w[0];  /* w15 = n mod base */
        if (invmodp_cuda(_MIPP_ mr_mip->w15,mr_mip->w6,mr_mip->w14)!=1)
        { /* problems */
            mr_berror_cuda(_MIPP_ MR_ERR_BAD_MODULUS);
            MR_OUT
            return (mr_small)0;
        }
#ifdef MR_KCM
    }
    copy_cuda(mr_mip->w14,mr_mip->big_ndash);
#endif

    mr_mip->ndash=mr_mip->base-mr_mip->w14->w[0]; /* = N' mod b */
    copy_cuda(n,mr_mip->modulus);
    mr_mip->check=OFF;
    mr_shift_cuda(_MIPP_ mr_mip->modulus,(int)mr_mip->modulus->len,mr_mip->pR);
    mr_mip->check=ON;
#ifdef MR_PENTIUM
/* prime the FP stack */
    if (mr_mip->ACTIVE)
    {
        ndash=mr_mip->ndash;
        base=mr_mip->base;
        magic *=base;
        ASM
        {
            finit
            fldcw WORD PTR control
            fld QWORD PTR ndash
            fld1
            fld QWORD PTR base
            fdiv
            fld QWORD PTR magic
        }
    }
#endif
    nres_cuda(_MIPP_ mr_mip->one,mr_mip->one);
    MR_OUT

    return mr_mip->ndash;
}

__device__ void nres_cuda(_MIPD_ big x,big y)
{ /* convert_cuda x to n-residue format */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(81)

    if (size_cuda(mr_mip->modulus)==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NO_MODULUS);
        MR_OUT
        return;
    }
    copy_cuda(x,y);
    divide_cuda(_MIPP_ y,mr_mip->modulus,mr_mip->modulus);
    if (size_cuda(y)<0) add_cuda(_MIPP_ y,mr_mip->modulus,y);
    if (!mr_mip->MONTY) 
    {
        MR_OUT
        return;
    }
    mr_mip->check=OFF;

    mr_shift_cuda(_MIPP_ y,(int)mr_mip->modulus->len,mr_mip->w0);
    divide_cuda(_MIPP_ mr_mip->w0,mr_mip->modulus,mr_mip->modulus);
    mr_mip->check=ON;
    copy_cuda(mr_mip->w0,y);

    MR_OUT
}

__device__ void redc_cuda(_MIPD_ big x,big y)
{ /* Montgomery's REDC function p. 520 */
  /* also used to convert_cuda n-residues back to normal form */
    mr_small carry,delay_carry,m,ndash,*w0g,*mg;

#ifdef MR_ITANIUM
    mr_small tm;
#endif
#ifdef MR_WIN64
    mr_small tm,tr;
#endif
    int i,j,rn,rn2;
    big w0,modulus;
#ifdef MR_NOASM
    union doubleword dble;
    mr_large dbled,ldres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(82)

    w0=mr_mip->w0;        /* get these into local variables (for inline assembly) */
    modulus=mr_mip->modulus;
    ndash=mr_mip->ndash;

    copy_cuda(x,w0);
    if (!mr_mip->MONTY)
    {
/*#ifdef MR_CELL
        mod256(_MIPP_ w0,w0);
#else */
        divide_cuda(_MIPP_ w0,modulus,modulus);
/* #endif */
        copy_cuda(w0,y);
        MR_OUT
        return;
    }
    delay_carry=0;
    rn=(int)modulus->len;
    rn2=rn+rn;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0) 
    {
#endif
#ifndef MR_NOFULLWIDTH
      mg=modulus->w;
      w0g=w0->w;
      for (i=0;i<rn;i++)
      {
       /* inline - substitutes for loop below */
#if INLINE_ASM == 1
            ASM cld
            ASM mov cx,rn
            ASM mov si,i
            ASM shl si,1
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g
            ASM add_cuda bx,si
            ASM mov ax,es:[bx]
            ASM mul WORD PTR ndash
            ASM mov di,ax
            ASM lds si,DWORD PTR mg
#else
            ASM mov bx,w0g
            ASM add_cuda bx,si
            ASM mov ax,[bx]
            ASM mul WORD PTR ndash
            ASM mov di,ax
            ASM mov si,mg
#endif
            ASM push bp
            ASM xor bp,bp
          m1:
            ASM lodsw
            ASM mul di
            ASM add_cuda ax,bp
            ASM adc dx,0
#ifdef MR_LMM
            ASM add_cuda es:[bx],ax
#else
            ASM add_cuda [bx],ax
#endif
            ASM adc dx,0
            ASM inc bx
            ASM inc bx
            ASM mov bp,dx
            ASM loop m1

            ASM pop bp
            ASM mov ax,delay_carry     
#ifdef MR_LMM
            ASM add_cuda es:[bx],ax
            ASM mov ax,0
            ASM adc ax,0
            ASM add_cuda es:[bx],dx
            ASM pop es
            ASM pop ds
#else
            ASM add_cuda [bx],ax
            ASM mov ax,0
            ASM adc ax,0
            ASM add_cuda [bx],dx
#endif
            ASM adc ax,0
            ASM mov delay_carry,ax
#endif
#if INLINE_ASM == 2
            ASM cld
            ASM mov cx,rn
            ASM mov si,i
            ASM shl si,2
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g
            ASM add_cuda bx,si
            ASM mov eax,es:[bx]
            ASM mul DWORD PTR ndash
            ASM mov edi,eax
            ASM lds si,DWORD PTR mg
#else
            ASM mov bx,w0g
            ASM add_cuda bx,si
            ASM mov eax,[bx]
            ASM mul DWORD PTR ndash
            ASM mov edi,eax
            ASM mov si,mg
#endif
            ASM push ebp
            ASM xor ebp,ebp
          m1:
            ASM lodsd
            ASM mul edi
            ASM add_cuda eax,ebp
            ASM adc edx,0
#ifdef MR_LMM
            ASM add_cuda es:[bx],eax
#else
            ASM add_cuda [bx],eax
#endif
            ASM adc edx,0
            ASM add_cuda bx,4
            ASM mov ebp,edx
            ASM loop m1

            ASM pop ebp
            ASM mov eax,delay_carry    
#ifdef MR_LMM
            ASM add_cuda es:[bx],eax
            ASM mov eax,0
            ASM adc eax,0
            ASM add_cuda es:[bx],edx
            ASM pop es
            ASM pop ds
#else 
            ASM add_cuda [bx],eax
            ASM mov eax,0
            ASM adc eax,0
            ASM add_cuda [bx],edx
#endif
            ASM adc eax,0
            ASM mov delay_carry,eax

#endif
#if INLINE_ASM == 3
            ASM mov ecx,rn
            ASM mov esi,i
            ASM shl esi,2
            ASM mov ebx,w0g
            ASM add_cuda ebx,esi
            ASM mov eax,[ebx]
            ASM mul DWORD PTR ndash
            ASM mov edi,eax
            ASM mov esi,mg
            ASM sub ebx,esi
            ASM sub ebx,4
            ASM push ebp
            ASM xor ebp,ebp
          m1:
            ASM mov eax,[esi]
            ASM add_cuda esi,4
            ASM mul edi
            ASM add_cuda eax,ebp
            ASM mov ebp,[esi+ebx]
            ASM adc edx,0
            ASM add_cuda ebp,eax
            ASM adc edx,0
            ASM mov [esi+ebx],ebp
            ASM dec ecx
            ASM mov ebp,edx
            ASM jnz m1

            ASM pop ebp
            ASM mov eax,delay_carry     
            ASM add_cuda [esi+ebx+4],eax
            ASM mov eax,0
            ASM adc eax,0
            ASM add_cuda [esi+ebx+4],edx
            ASM adc eax,0
            ASM mov delay_carry,eax

#endif
#if INLINE_ASM == 4
   ASM (
           "movl %0,%%ecx\n"
           "movl %1,%%esi\n"
           "shll $2,%%esi\n"
           "movl %2,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "movl (%%ebx),%%eax\n"
           "mull %3\n"
           "movl %%eax,%%edi\n"
           "movl %4,%%esi\n"
           "subl %%esi,%%ebx\n"
           "subl $4,%%ebx\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"
        "m1:\n"
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n" 
           "mull %%edi\n"
           "addl %%ebp,%%eax\n"
           "movl (%%esi,%%ebx),%%ebp\n"
           "adcl $0,%%edx\n"
           "addl %%eax,%%ebp\n" 
           "adcl $0,%%edx\n"
           "movl %%ebp,(%%esi,%%ebx)\n"
           "decl %%ecx\n"
           "movl %%edx,%%ebp\n"
           "jnz m1\n"   

           "popl %%ebp\n"
           "movl %5,%%eax\n"
           "addl %%eax,4(%%esi,%%ebx)\n"
           "movl $0,%%eax\n"
           "adcl $0,%%eax\n"
           "addl %%edx,4(%%esi,%%ebx)\n"
           "adcl $0,%%eax\n"
           "movl %%eax,%5\n"
       
        :
        :"m"(rn),"m"(i),"m"(w0g),"m"(ndash),"m"(mg),"m"(delay_carry)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif

#ifndef INLINE_ASM
/*        muldvd_cuda(w0->w[i],ndash,0,&m);    Note that after this time   */
        m=ndash*w0->w[i];
        carry=0;                       /* around the loop, w0[i]=0    */

        for (j=0;j<rn;j++)
        {
#ifdef MR_NOASM 
            dble.d=(mr_large)m*modulus->w[j]+carry+w0->w[i+j];
            w0->w[i+j]=dble.h[MR_BOT];
            carry=dble.h[MR_TOP];
#else
            muldvd2_cuda(m,modulus->w[j],&carry,&w0->w[i+j]);
#endif
        }
        w0->w[rn+i]+=delay_carry;
        if (w0->w[rn+i]<delay_carry) delay_carry=1;
        else delay_carry=0;
        w0->w[rn+i]+=carry;
        if (w0->w[rn+i]<carry) delay_carry=1; 
#endif
      }
#endif

#ifndef MR_SIMPLE_BASE
    }
    else for (i=0;i<rn;i++) 
    {
#ifdef MR_FP_ROUNDING
        imuldiv(w0->w[i],ndash,0,mr_mip->base,mr_mip->inverse_base,&m);
#else
        muldiv_cuda(w0->w[i],ndash,0,mr_mip->base,&m);
#endif
        carry=0;
        for (j=0;j<rn;j++)
        {
#ifdef MR_NOASM 
          dbled=(mr_large)m*modulus->w[j]+carry+w0->w[i+j];
#ifdef MR_FP_ROUNDING
          carry=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
#else
#ifndef MR_FP
          if (mr_mip->base==mr_mip->base2)
              carry=(mr_small)(dbled>>mr_mip->lg2b);
          else 
#endif  
              carry=(mr_small)MR_LROUND(dbled/mr_mip->base);
#endif
          w0->w[i+j]=(mr_small)(dbled-(mr_large)carry*mr_mip->base);  
#else
#ifdef MR_FP_ROUNDING
          carry=imuldiv(modulus->w[j],m,w0->w[i+j]+carry,mr_mip->base,mr_mip->inverse_base,&w0->w[i+j]);
#else
          carry=muldiv_cuda(modulus->w[j],m,w0->w[i+j]+carry,mr_mip->base,&w0->w[i+j]);
#endif
#endif
        }
        w0->w[rn+i]+=(delay_carry+carry);
        delay_carry=0;
        if (w0->w[rn+i]>=mr_mip->base)
        {
            w0->w[rn+i]-=mr_mip->base;
            delay_carry=1; 
        }
    }
#endif
    w0->w[rn2]=delay_carry;
    w0->len=rn2+1;
    mr_shift_cuda(_MIPP_ w0,(-rn),w0);
    mr_lzero_cuda(w0);
    
    if (mr_compare_cuda(w0,modulus)>=0) mr_psub_cuda(_MIPP_ w0,modulus,w0);
    copy_cuda(w0,y);
    MR_OUT
}

/* "Complex" method for ZZn2 squaring */

__device__ void nres_complex_cuda(_MIPD_ big a,big b,big r,big i)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
	MR_IN(225)

	if (mr_mip->NO_CARRY && mr_mip->qnr==-1)
	{ /* if modulus is small enough we can ignore carries, and use simple addition and subtraction */
	  /* recall that Montgomery reduction can cope as long as product is less than pR */
#ifdef MR_COMBA
#ifdef MR_COUNT_OPS
fpa+=3;
#endif
		if (mr_mip->ACTIVE)
		{
			comba_add(a,b,mr_mip->w1);
			comba_add(a,mr_mip->modulus,mr_mip->w2); /* a-b is p+a-b */
			comba_sub(mr_mip->w2,b,mr_mip->w2);
			comba_add(a,a,r);
		}
		else
		{
#endif
			mr_padd_cuda(_MIPP_ a,b,mr_mip->w1);
			mr_padd_cuda(_MIPP_ a,mr_mip->modulus,mr_mip->w2);
			mr_psub_cuda(_MIPP_ mr_mip->w2,b,mr_mip->w2);
			mr_padd_cuda(_MIPP_ a,a,r);
#ifdef MR_COMBA
		}
#endif
		nres_modmult_cuda(_MIPP_ r,b,i);
		nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,r);
	}
	else
	{
		nres_modadd_cuda(_MIPP_ a,b,mr_mip->w1);
		nres_modsub_cuda(_MIPP_ a,b,mr_mip->w2);

		if (mr_mip->qnr==-2)
			nres_modsub_cuda(_MIPP_ mr_mip->w2,b,mr_mip->w2);
     
		nres_modmult_cuda(_MIPP_ a,b,i);
		nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,r);

		if (mr_mip->qnr==-2)
			nres_modadd_cuda(_MIPP_ r,i,r);

		nres_modadd_cuda(_MIPP_ i,i,i);
	}
	MR_OUT
}

#ifndef MR_NO_LAZY_REDUCTION

/*

Lazy reduction technique for zzn2 multiplication - competitive if Reduction is more
expensive that Multiplication. This is true for pairing-based crypto. Note that
Lazy reduction can also be used with Karatsuba! Uses w1, w2, w5, and w6.

Reduction poly is X^2-D=0

(a0+a1.X).(b0+b1.X) = (a0.b0 + D.a1.b1) + (a1.b0+a0.b1).X

Karatsuba

   (a0.b0+D.a1.b1) + ((a0+a1)(b0+b1) - a0.b0 - a1.b1).X  
*/

__device__ void nres_lazy_cuda(_MIPD_ big a0,big a1,big b0,big b1,big r,big i)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    mr_mip->check=OFF;
#ifdef MR_COUNT_OPS
fpc+=3;
fpa+=5;
if (mr_mip->qnr==-2) fpa++;
#endif

#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        comba_mult(a0,b0,mr_mip->w0);
        comba_mult(a1,b1,mr_mip->w5);
    }
    else
    {
#endif
#ifdef MR_KCM
    if (mr_mip->ACTIVE)
    {
        kcm_mul(_MIPP_ a1,b1,mr_mip->w5); /* this destroys w0! */
        kcm_mul(_MIPP_ a0,b0,mr_mip->w0);
    }
    else
    { 
#endif
        MR_IN(151)
        multiply_cuda(_MIPP_ a0,b0,mr_mip->w0);
        multiply_cuda(_MIPP_ a1,b1,mr_mip->w5);
#ifdef MR_COMBA
    }
#endif
#ifdef MR_KCM
    }
#endif

	if (mr_mip->NO_CARRY && mr_mip->qnr==-1)
	{ /* if modulus is small enough we can ignore carries, and use simple addition and subtraction */
#ifdef MR_COMBA
#ifdef MR_COUNT_OPS
fpa+=2;
#endif
		if (mr_mip->ACTIVE)
		{
			comba_double_add(mr_mip->w0,mr_mip->w5,mr_mip->w6);
			comba_add(a0,a1,mr_mip->w1);
			comba_add(b0,b1,mr_mip->w2); 
		}
		else
		{
#endif
			mr_padd_cuda(_MIPP_ mr_mip->w0,mr_mip->w5,mr_mip->w6);
			mr_padd_cuda(_MIPP_ a0,a1,mr_mip->w1);
			mr_padd_cuda(_MIPP_ b0,b1,mr_mip->w2); 
#ifdef MR_COMBA
		}
#endif
	}
	else
	{
		nres_double_modadd_cuda(_MIPP_ mr_mip->w0,mr_mip->w5,mr_mip->w6);  /* w6 =  a0.b0+a1.b1 */
		if (mr_mip->qnr==-2)
          nres_double_modadd_cuda(_MIPP_ mr_mip->w5,mr_mip->w5,mr_mip->w5);
		nres_modadd_cuda(_MIPP_ a0,a1,mr_mip->w1);
		nres_modadd_cuda(_MIPP_ b0,b1,mr_mip->w2); 
    }
	nres_double_modsub_cuda(_MIPP_ mr_mip->w0,mr_mip->w5,mr_mip->w0);  /* r = a0.b0+D.a1.b1 */

#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        comba_redc(_MIPP_ mr_mip->w0,r);
        comba_mult(mr_mip->w1,mr_mip->w2,mr_mip->w0);
    }
    else
    {
#endif
#ifdef MR_KCM
    if (mr_mip->ACTIVE)
    {
        kcm_redc(_MIPP_ mr_mip->w0,r);
        kcm_mul(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w0);
    }
    else
    {
#endif
        redc_cuda(_MIPP_ mr_mip->w0,r);
        multiply_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w0);           /* w0=(a0+a1)*(b0+b1) */
#ifdef MR_COMBA
    }
#endif
#ifdef MR_KCM
    }
#endif

	if (mr_mip->NO_CARRY && mr_mip->qnr==-1)
	{
#ifdef MR_COMBA
		if (mr_mip->ACTIVE)
			comba_double_sub(mr_mip->w0,mr_mip->w6,mr_mip->w0);
		else
#endif
			mr_psub_cuda(_MIPP_ mr_mip->w0,mr_mip->w6,mr_mip->w0);
	}
	else
		nres_double_modsub_cuda(_MIPP_ mr_mip->w0,mr_mip->w6,mr_mip->w0); /* (a0+a1)*(b0+b1) - w6 */

#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        comba_redc(_MIPP_ mr_mip->w0,i);
    }
    else
    {
#endif  
#ifdef MR_KCM
    if (mr_mip->ACTIVE)
    {
        kcm_redc(_MIPP_ mr_mip->w0,i);
    }
    else
    {
#endif      
        redc_cuda(_MIPP_ mr_mip->w0,i);
        MR_OUT
#ifdef MR_COMBA
    }
#endif
#ifdef MR_KCM
    }
#endif

    mr_mip->check=ON;

}

#endif

#ifndef MR_STATIC

__device__ void nres_dotprod_cuda(_MIPD_ int n,big *x,big *y,big w)
{
    int i;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;
    MR_IN(120)
    mr_mip->check=OFF;
    zero_cuda(mr_mip->w7);
    for (i=0;i<n;i++)
    {
        multiply_cuda(_MIPP_ x[i],y[i],mr_mip->w0);
        mr_padd_cuda(_MIPP_ mr_mip->w7,mr_mip->w0,mr_mip->w7);
    }
    copy_cuda(mr_mip->pR,mr_mip->w6);
        /* w6 = p.R */
    divide_cuda(_MIPP_ mr_mip->w7,mr_mip->w6,mr_mip->w6);
    redc_cuda(_MIPP_ mr_mip->w7,w);

    mr_mip->check=ON;
    MR_OUT
}

#endif

__device__ void nres_negate_cuda(_MIPD_ big x, big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
	if (size_cuda(x)==0) 
	{
		zero_cuda(w);
		return;
	}
#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        comba_negate(_MIPP_ x,w);
        return;
    }    
    else
    {
#endif
        if (mr_mip->ERNUM) return;

        MR_IN(92)
        mr_psub_cuda(_MIPP_ mr_mip->modulus,x,w);    
        MR_OUT

#ifdef MR_COMBA
    }
#endif

}

__device__ void nres_div2_cuda(_MIPD_ big x,big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(198)
    copy_cuda(x,mr_mip->w1);
    if (remain_cuda(_MIPP_ mr_mip->w1,2)!=0)
        add_cuda(_MIPP_ mr_mip->w1,mr_mip->modulus,mr_mip->w1);
    subdiv_cuda(_MIPP_ mr_mip->w1,2,mr_mip->w1);
    copy_cuda(mr_mip->w1,w);

    MR_OUT
}

__device__ void nres_div3_cuda(_MIPD_ big x,big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(199)
    copy_cuda(x,mr_mip->w1);
    while (remain_cuda(_MIPP_ mr_mip->w1,3)!=0)
        add_cuda(_MIPP_ mr_mip->w1,mr_mip->modulus,mr_mip->w1);
    subdiv_cuda(_MIPP_ mr_mip->w1,3,mr_mip->w1);
    copy_cuda(mr_mip->w1,w);

    MR_OUT
}

__device__ void nres_div5_cuda(_MIPD_ big x,big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(208)
    copy_cuda(x,mr_mip->w1);
    while (remain_cuda(_MIPP_ mr_mip->w1,5)!=0)
        add_cuda(_MIPP_ mr_mip->w1,mr_mip->modulus,mr_mip->w1);
    subdiv_cuda(_MIPP_ mr_mip->w1,5,mr_mip->w1);
    copy_cuda(mr_mip->w1,w);

    MR_OUT
}

/* mod pR addition and subtraction */
#ifndef MR_NO_LAZY_REDUCTION

__device__ void nres_double_modadd_cuda(_MIPD_ big x,big y,big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_COMBA

    if (mr_mip->ACTIVE)
    {
        comba_double_modadd(_MIPP_ x,y,w);
        return;
    }
    else
    {
#endif 

        if (mr_mip->ERNUM) return;
        MR_IN(153)

        mr_padd_cuda(_MIPP_ x,y,w);
        if (mr_compare_cuda(w,mr_mip->pR)>=0)
            mr_psub_cuda(_MIPP_ w,mr_mip->pR,w);

        MR_OUT
#ifdef MR_COMBA
    }
#endif
}

__device__ void nres_double_modsub_cuda(_MIPD_ big x,big y,big w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_COMBA

    if (mr_mip->ACTIVE)
    {
        comba_double_modsub(_MIPP_ x,y,w);
        return;
    }
    else
    {
#endif 

        if (mr_mip->ERNUM) return;
        MR_IN(154)

        if (mr_compare_cuda(x,y)>=0)
            mr_psub_cuda(_MIPP_ x,y,w);
        else
        {
            mr_psub_cuda(_MIPP_ y,x,w);
            mr_psub_cuda(_MIPP_ mr_mip->pR,w,w);
        }

        MR_OUT
#ifdef MR_COMBA
    }
#endif
}

#endif

__device__ void nres_modadd_cuda(_MIPD_ big x,big y,big w)
{ /* modular addition */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_COUNT_OPS
fpa++; 
#endif
#ifdef MR_COMBA

    if (mr_mip->ACTIVE)
    {
        comba_modadd(_MIPP_ x,y,w);
        return;
    }
    else
    {
#endif
        if (mr_mip->ERNUM) return;

        MR_IN(90)
        mr_padd_cuda(_MIPP_ x,y,w);
        if (mr_compare_cuda(w,mr_mip->modulus)>=0) mr_psub_cuda(_MIPP_ w,mr_mip->modulus,w);

        MR_OUT
#ifdef MR_COMBA
    }
#endif
}

__device__ void nres_modsub_cuda(_MIPD_ big x,big y,big w)
{ /* modular subtraction */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_COUNT_OPS
fpa++;
#endif
#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        comba_modsub(_MIPP_ x,y,w);
        return;
    }
    else
    {
#endif
        if (mr_mip->ERNUM) return;

        MR_IN(91)

        if (mr_compare_cuda(x,y)>=0)
            mr_psub_cuda(_MIPP_ x,y,w);
        else
        {
            mr_psub_cuda(_MIPP_ y,x,w);
            mr_psub_cuda(_MIPP_ mr_mip->modulus,w,w);
        }

        MR_OUT
#ifdef MR_COMBA
    }
#endif

}

__device__ int nres_moddiv_cuda(_MIPD_ big x,big y,big w)
{ /* Modular division using n-residues w=x/y mod n */
    int gcd;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return 0;

    MR_IN(85)

    if (x==y)
    { /* Illegal parameter usage */
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        
        return 0;
    }
    redc_cuda(_MIPP_ y,mr_mip->w6);
    gcd=invmodp_cuda(_MIPP_ mr_mip->w6,mr_mip->modulus,mr_mip->w6);
   
    if (gcd!=1) zero_cuda(w); /* fails silently and returns 0 */
    else
    {
        nres_cuda(_MIPP_ mr_mip->w6,mr_mip->w6);
        nres_modmult_cuda(_MIPP_ x,mr_mip->w6,w);
    /*    mad_cuda(_MIPP_ x,mr_mip->w6,x,mr_mip->modulus,mr_mip->modulus,w); */
    }
    MR_OUT
    return gcd;
}

__device__ void nres_premult_cuda(_MIPD_ big x,int k,big w)
{ /* multiply_cuda n-residue by small ordinary integer */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    int sign=0;
    if (k==0) 
    {
        zero_cuda(w);
        return;
    }
    if (k<0)
    {
        k=-k;
        sign=1;
    }
    if (mr_mip->ERNUM) return;

    MR_IN(102)

    if (k<=6)
    {
        switch (k)
        {
        case 1: copy_cuda(x,w);
                break;
        case 2: nres_modadd_cuda(_MIPP_ x,x,w);
                break;    
        case 3:
                nres_modadd_cuda(_MIPP_ x,x,mr_mip->w0);
                nres_modadd_cuda(_MIPP_ x,mr_mip->w0,w);
                break;
        case 4:
                nres_modadd_cuda(_MIPP_ x,x,w);
                nres_modadd_cuda(_MIPP_ w,w,w);
                break;    
        case 5:
                nres_modadd_cuda(_MIPP_ x,x,mr_mip->w0);
                nres_modadd_cuda(_MIPP_ mr_mip->w0,mr_mip->w0,mr_mip->w0);
                nres_modadd_cuda(_MIPP_ x,mr_mip->w0,w);
                break;
        case 6:
                nres_modadd_cuda(_MIPP_ x,x,w);
                nres_modadd_cuda(_MIPP_ w,w,mr_mip->w0);
                nres_modadd_cuda(_MIPP_ w,mr_mip->w0,w);
                break;
        }
        if (sign==1) nres_negate_cuda(_MIPP_ w,w);
        MR_OUT
        return;
    }

    mr_pmul_cuda(_MIPP_ x,(mr_small)k,mr_mip->w0);
#ifdef MR_COMBA
#ifdef MR_SPECIAL
	comba_redc(_MIPP_ mr_mip->w0,w);
#else
	divide_cuda(_MIPP_ mr_mip->w0,mr_mip->modulus,mr_mip->modulus);
	copy_cuda(mr_mip->w0,w);
#endif
#else
    divide_cuda(_MIPP_ mr_mip->w0,mr_mip->modulus,mr_mip->modulus);
	copy_cuda(mr_mip->w0,w);
#endif 
	
    if (sign==1) nres_negate_cuda(_MIPP_ w,w);

    MR_OUT
}

__device__ void nres_modmult_cuda(_MIPD_ big x,big y,big w)
{ /* Modular multiplication using n-residues w=x*y mod n */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if ((x==NULL || x->len==0) && x==w) return;
    if ((y==NULL || y->len==0) && y==w) return;
    if (y==NULL || x==NULL || x->len==0 || y->len==0)
    {
        zero_cuda(w);
        return;
    }
#ifdef MR_COUNT_OPS
fpc++;
#endif
#ifdef MR_COMBA
    if (mr_mip->ACTIVE)
    {
        if (x==y) comba_square(x,mr_mip->w0);
        else      comba_mult(x,y,mr_mip->w0);
        comba_redc(_MIPP_ mr_mip->w0,w);
    }
    else
    {
#endif
#ifdef MR_KCM
    if (mr_mip->ACTIVE)
    {
        if (x==y) kcm_sqr(_MIPP_ x,mr_mip->w0);
        else      kcm_mul(_MIPP_ x,y,mr_mip->w0);
        kcm_redc(_MIPP_ mr_mip->w0,w);
    }
    else
    { 
#endif
#ifdef MR_PENTIUM
    if (mr_mip->ACTIVE)
    {
        if (x==y) fastmodsquare(_MIPP_ x,w);
        else      fastmodmult(_MIPP_ x,y,w);
    }
    else
    { 
#endif
        if (mr_mip->ERNUM) return;

        MR_IN(83)

        mr_mip->check=OFF;
        multiply_cuda(_MIPP_ x,y,mr_mip->w0);
        redc_cuda(_MIPP_ mr_mip->w0,w);
        mr_mip->check=ON;
        MR_OUT
#ifdef MR_COMBA
}
#endif
#ifdef MR_KCM
}
#endif
#ifdef MR_PENTIUM
}
#endif

}

/* Montgomery's trick for finding multiple   *
 * simultaneous modular inverses             *
 * Based on the observation that             *
 *           1/x = yz*(1/xyz)                *
 *           1/y = xz*(1/xyz)                *
 *           1/z = xy*(1/xyz)                *
 * Why are all of Peter Montgomery's clever  *
 * algorithms always described as "tricks" ??*/

__device__ BOOL nres_double_inverse_cuda(_MIPD_ big x,big y,big w,big z)
{ /* find y=1/x mod n and z=1/w mod n */
  /* 1/x = w/xw, and 1/w = x/xw       */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    MR_IN(145)

    nres_modmult_cuda(_MIPP_ x,w,mr_mip->w6);  /* xw */

    if (size_cuda(mr_mip->w6)==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }
    redc_cuda(_MIPP_ mr_mip->w6,mr_mip->w6);
    redc_cuda(_MIPP_ mr_mip->w6,mr_mip->w6);
    invmodp_cuda(_MIPP_ mr_mip->w6,mr_mip->modulus,mr_mip->w6);

    nres_modmult_cuda(_MIPP_ w,mr_mip->w6,mr_mip->w5);
    nres_modmult_cuda(_MIPP_ x,mr_mip->w6,z);
    copy_cuda(mr_mip->w5,y);

    MR_OUT
    return TRUE;
}

__device__ BOOL nres_multi_inverse_cuda(_MIPD_ int m,big *x,big *w)
{ /* find w[i]=1/x[i] mod n, for i=0 to m-1 *
   * x and w MUST be distinct               */
    int i;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (m==0) return TRUE;
    if (m<0) return FALSE;
    MR_IN(118)

    if (x==w)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return FALSE;
    }

    if (m==1)
    {
        copy_cuda(mr_mip->one,w[0]);
        nres_moddiv_cuda(_MIPP_ w[0],x[0],w[0]);
        MR_OUT
        return TRUE;
    }

    convert_cuda(_MIPP_ 1,w[0]);
    copy_cuda(x[0],w[1]);
    for (i=2;i<m;i++)
        nres_modmult_cuda(_MIPP_ w[i-1],x[i-1],w[i]); 

    nres_modmult_cuda(_MIPP_ w[m-1],x[m-1],mr_mip->w6);  /* y=x[0]*x[1]*x[2]....x[m-1] */
    if (size_cuda(mr_mip->w6)==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }

    redc_cuda(_MIPP_ mr_mip->w6,mr_mip->w6);
    redc_cuda(_MIPP_ mr_mip->w6,mr_mip->w6);

    invmodp_cuda(_MIPP_ mr_mip->w6,mr_mip->modulus,mr_mip->w6);

/* Now y=1/y */

    copy_cuda(x[m-1],mr_mip->w5);
    nres_modmult_cuda(_MIPP_ w[m-1],mr_mip->w6,w[m-1]);

    for (i=m-2;;i--)
    {
        if (i==0)
        {
            nres_modmult_cuda(_MIPP_ mr_mip->w5,mr_mip->w6,w[0]);
            break;
        }
        nres_modmult_cuda(_MIPP_ w[i],mr_mip->w5,w[i]);
        nres_modmult_cuda(_MIPP_ w[i],mr_mip->w6,w[i]);
        nres_modmult_cuda(_MIPP_ mr_mip->w5,x[i],mr_mip->w5);
    }

    MR_OUT 
    return TRUE;   
}

/* initialise elliptic curve */

__device__ void ecurve_init_cuda(_MIPD_ big a,big b,big p,int type)
{ /* Initialize the active ecurve    *
   * Asize indicate size_cuda of A        *
   * Bsize indicate size_cuda of B        */
    int as;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(93)

#ifndef MR_NO_SS
    mr_mip->SS=FALSE;       /* no special support for super-singular curves */ 
#endif

    prepare_monty_cuda(_MIPP_ p);

    mr_mip->Asize=size_cuda(a);
    if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
    {
        if (mr_mip->Asize>=0)
        { /* big positive number - check it isn't minus something small */
           copy_cuda(a,mr_mip->w1);
           divide_cuda(_MIPP_ mr_mip->w1,p,p);
           subtract_cuda(_MIPP_ p,mr_mip->w1,mr_mip->w1);
           as=size_cuda(mr_mip->w1);
           if (as<MR_TOOBIG) mr_mip->Asize=-as;
        }
    }
    nres_cuda(_MIPP_ a,mr_mip->A);

    mr_mip->Bsize=size_cuda(b);
    if (mr_abs(mr_mip->Bsize)==MR_TOOBIG) 
    {
        if (mr_mip->Bsize>=0)
        { /* big positive number - check it isn't minus something small */
           copy_cuda(b,mr_mip->w1);
           divide_cuda(_MIPP_ mr_mip->w1,p,p);
           subtract_cuda(_MIPP_ p,mr_mip->w1,mr_mip->w1);
           as=size_cuda(mr_mip->w1);
           if (as<MR_TOOBIG) mr_mip->Bsize=-as;
        }
    }

    nres_cuda(_MIPP_ b,mr_mip->B);
#ifdef MR_EDWARDS
    mr_mip->coord=MR_PROJECTIVE; /* only type supported for Edwards curves */
#else
#ifndef MR_AFFINE_ONLY
    if (type==MR_BEST) mr_mip->coord=MR_PROJECTIVE;
    else mr_mip->coord=type;
#else
    if (type==MR_PROJECTIVE)
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
#endif
#endif
    MR_OUT
    return;
}


#endif

#ifndef mrjack_c
#define mrjack_c

__device__ int jack_cuda(_MIPD_ big a,big n)
{ /* find jacobi symbol (a/n), for positive odd n */
    big w;
    int nm8,onm8,t;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM || size_cuda(a)==0 || size_cuda(n) <1) return 0;
    MR_IN(3)

    t=1;
    copy_cuda(n,mr_mip->w2);
    nm8=remain_cuda(_MIPP_ mr_mip->w2,8);
    if (nm8%2==0) 
    {
        MR_OUT
        return 0;
    }
    
    if (size_cuda(a)<0)
    {
        if (nm8%4==3) t=-1;
        negify_cuda(a,mr_mip->w1);
    }
    else copy_cuda(a,mr_mip->w1);

    while (size_cuda(mr_mip->w1)!=0)
    {
        while (remain_cuda(_MIPP_ mr_mip->w1,2)==0)
        {
            subdiv_cuda(_MIPP_ mr_mip->w1,2,mr_mip->w1);
            if (nm8==3 || nm8==5) t=-t; 
        }
        if (mr_compare_cuda(mr_mip->w1,mr_mip->w2)<0)
        {
            onm8=nm8;
            w=mr_mip->w1; mr_mip->w1=mr_mip->w2; mr_mip->w2=w;
            nm8=remain_cuda(_MIPP_ mr_mip->w2,8);
            if (onm8%4==3 && nm8%4==3) t=-t;
        }
        mr_psub_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w1);
        subdiv_cuda(_MIPP_ mr_mip->w1,2,mr_mip->w1);
 
        if (nm8==3 || nm8==5) t=-t; 
    }

    MR_OUT
    if (size_cuda(mr_mip->w2)==1) return t;
    return 0;
}

/*
 *   See "Efficient Algorithms for Computing the Jacobi Symbol"
 *   Eikenberry & Sorenson
 *
 *   Its turns out this is slower than the binary method above for reasonable sizes
 *   of parameters (and takes up a lot more space!)


#ifdef MR_FP
#include <math.h>
#endif


static void rfind(mr_small u,mr_small v,mr_small k,mr_small sk,mr_utype *a,mr_utype *b)
{
    mr_utype x2,y2,r;
    mr_small w,q,x1,y1,sr;
#ifdef MR_FP
    mr_small dres;
#endif

    w=invers(v,k);
    w=smul(u,w,k);
    
    x1=k; x2=0;
    y1=w; y2=1;

// NOTE: x1 and y1 are always +ve. x2 and y2 are always small 

    while (y1>=sk)
    {
#ifndef MR_NOFULLWIDTH
        if (x1==0) q=muldvm_cuda((mr_small)1,(mr_small)0,y1,&sr);
        else 
#endif
        q=MR_DIV(x1,y1);
        r= x1-q*y1; x1=y1; y1=r;
        sr=x2-q*y2; x2=y2; y2=sr;
    }
    if (y2>=0) { *a=y2;  *b=0-y1; }
    else       { *a=-y2; *b=y1;  }
}

int jack_cuda(_MIPD_ big U,big V)
{ // find jacobi symbol for U wrt V. Only defined for 
  // positive V, V odd. Otherwise returns 0           
    int i,e,r,m,t,v8,u4;
    mr_utype a,b;
    mr_small u,v,d,g_cuda,k,sk,s;
#ifdef MR_FP
    mr_small dres;
#endif
    big w;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifdef MR_FP_ROUNDING
    mr_large ik,id;
#endif
    if (mr_mip->ERNUM || size_cuda(U)==0 || size_cuda(V) <1) return 0;
    copy_cuda(U,mr_mip->w1);
    copy_cuda(V,mr_mip->w2);
    a=0;
    MR_IN(3)

    if (remain_cuda(_MIPP_ mr_mip->w2,2)==0)
    { // V is even 
        MR_OUT
        return 0;
    }

    if (mr_mip->base!=0)
    {
        k=1;
        for (m=1;;m++)
        {
           k*=2;
           if (k==MAXBASE) break;
        }    
        if (m%2==1) {m--; k=MR_DIV(k,2);}
#ifdef MR_FP_ROUNDING
        ik=mr_invert(k);
#endif
    }
    else
    {
        m=MIRACL;
        k=0;
    }
    r=m/2;
    sk=1;
    for (i=0;i<r;i++) sk*=2;

    t=1;
    v8=remain_cuda(_MIPP_ mr_mip->w2,8); 

    while (!mr_mip->ERNUM && size_cuda(mr_mip->w1)!=0)
    {
        if (size_cuda(mr_mip->w1)<0)
        {
            negify_cuda(mr_mip->w1,mr_mip->w1);
            if (v8%4==3) t=-t;
        }

        do { // oddify 

#ifndef MR_ALWAYS_BINARY
            if (mr_mip->base==mr_mip->base2) 
            {
#endif
                 if (mr_mip->base==k) u=mr_mip->w1->w[0];
                 else                 u=MR_REMAIN(mr_mip->w1->w[0],k); 
#ifndef MR_ALWAYS_BINARY
            }

#ifdef MR_FP_ROUNDING
            else u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,ik,mr_mip->w3);
#else
            else u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,mr_mip->w3);
#endif

#endif
            if (u==0) {s=k; e=0;}
            else
            {
                s=1; e=0;
                while (MR_REMAIN(u,2)==0) {s*=2; e++; u=MR_DIV(u,2);}
            }
            if (s==mr_mip->base) mr_shift_cuda(_MIPP_ mr_mip->w1,-1,mr_mip->w1);
#ifdef MR_FP_ROUNDING
            else if (s>1) 
            { 
                mr_sdiv_cuda(_MIPP_ mr_mip->w1,s,mr_invert(s),mr_mip->w1);
            }
#else
            else if (s>1) mr_sdiv_cuda(_MIPP_ mr_mip->w1,s,mr_mip->w1);
#endif
        } while (u==0);
        if (e%2!=0 && (v8==3 || v8==5)) t=-t;
        if (mr_compare_cuda(mr_mip->w1,mr_mip->w2)<0)
        {
            if (mr_mip->base==mr_mip->base2) u4=(int)MR_REMAIN(mr_mip->w1->w[0],4);
            else                             u4=remain_cuda(_MIPP_ mr_mip->w1,4);
            if (v8%4==3 && u4==3) t=-t; 
            w=mr_mip->w1; mr_mip->w1=mr_mip->w2; mr_mip->w2=w;
        }

#ifndef MR_ALWAYS_BINARY
        if (mr_mip->base==mr_mip->base2)
        {
#endif
            if (k==mr_mip->base)   
            {
                u=mr_mip->w1->w[0];
                v=mr_mip->w2->w[0];
            }
            else
            {
                u=MR_REMAIN(mr_mip->w1->w[0],k);
                v=MR_REMAIN(mr_mip->w2->w[0],k);
            }
#ifndef MR_ALWAYS_BINARY
        }
        else
        {
#ifdef MR_FP_ROUNDING
            u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,ik,mr_mip->w3);
            v=mr_sdiv_cuda(_MIPP_ mr_mip->w2,k,ik,mr_mip->w3);
#else
            u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,mr_mip->w3);
            v=mr_sdiv_cuda(_MIPP_ mr_mip->w2,k,mr_mip->w3);
#endif
        }
#endif
        rfind(u,v,k,sk,&a,&b);
        if (a>1)
        {
#ifdef MR_FP_ROUNDING
            d=mr_sdiv_cuda(_MIPP_ mr_mip->w2,a,mr_invert(a),mr_mip->w3);
#else
            d=mr_sdiv_cuda(_MIPP_ mr_mip->w2,a,mr_mip->w3);
#endif
            d=sgcd(d,a);
            a=MR_DIV(a,d); 
        }
        else d=1;

        if (d>1) 
        {
#ifdef MR_FP_ROUNDING
            id=mr_invert(d);
            mr_sdiv_cuda(_MIPP_ mr_mip->w2,d,id,mr_mip->w2);
            u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,d,id,mr_mip->w3);
#else
            mr_sdiv_cuda(_MIPP_ mr_mip->w2,d,mr_mip->w2);
            u=mr_sdiv_cuda(_MIPP_ mr_mip->w1,d,mr_mip->w3);
#endif
        }
        else u=0;   

        g_cuda=a;
        if (mr_mip->base==mr_mip->base2) v8=(int)MR_REMAIN(mr_mip->w2->w[0],8);
        else                             v8=remain_cuda(_MIPP_ mr_mip->w2,8);
        while (MR_REMAIN(g_cuda,2)==0)
        {
            g_cuda=MR_DIV(g_cuda,2);
            if (v8==3 || v8==5) t=-t;
        }
        if (MR_REMAIN(g_cuda,4)==3 && v8%4==3) t=-t;
#ifdef MR_FP_ROUNDING
        v=mr_sdiv_cuda(_MIPP_ mr_mip->w2,g_cuda,mr_invert(g_cuda),mr_mip->w3);
#else
        v=mr_sdiv_cuda(_MIPP_ mr_mip->w2,g_cuda,mr_mip->w3);
#endif
        t*=jac(v,g_cuda)*jac(u,d);
        if (t==0) 
        {
            MR_OUT
            return 0;
        }

// printf("a= %I64d b=%I64d %d\n",a,b,(int)b); 

        if (a>1) mr_pmul_cuda(_MIPP_ mr_mip->w1,a,mr_mip->w1);
        if (b>=0)
            mr_pmul_cuda(_MIPP_ mr_mip->w2,b,mr_mip->w3);
        else
        {
            b=-b;
            mr_pmul_cuda(_MIPP_ mr_mip->w2,b,mr_mip->w3);
            negify_cuda(mr_mip->w3,mr_mip->w3);
        }
       // premult_cuda(_MIPP_ mr_mip->w2,(int)b,mr_mip->w3); <- nasty bug - potential loss of precision in b 
        add_cuda(_MIPP_ mr_mip->w1,mr_mip->w3,mr_mip->w1);
        if (k==mr_mip->base) mr_shift_cuda(_MIPP_ mr_mip->w1,-1,mr_mip->w1);
#ifdef MR_FP_ROUNDING
        else                 mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,ik,mr_mip->w1);
#else
        else                 mr_sdiv_cuda(_MIPP_ mr_mip->w1,k,mr_mip->w1);
#endif
    }
    MR_OUT
    if (size_cuda(mr_mip->w2)==1) return t;
    return 0; 
} 

*/


#endif

#ifndef mrsroot_c
#define mrsroot_c

__device__ BOOL nres_sqroot_cuda(_MIPD_ big x,big w)
{ /* w=sqrt(x) mod p. This depends on p being prime! */
    int t,js;
   
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    copy_cuda(x,w);
    if (size_cuda(w)==0) return TRUE; 

    MR_IN(100)

    redc_cuda(_MIPP_ w,w);   /* get it back into normal form */

    if (size_cuda(w)==1) /* square root of 1 is 1 */
    {
        nres_cuda(_MIPP_ w,w);
        MR_OUT
        return TRUE;
    }

    if (size_cuda(w)==4) /* square root of 4 is 2 */
    {
        convert_cuda(_MIPP_ 2,w);
        nres_cuda(_MIPP_ w,w);
        MR_OUT
        return TRUE;
    }

    if (jack_cuda(_MIPP_ w,mr_mip->modulus)!=1) 
    { /* Jacobi test */ 
        zero_cuda(w);
        MR_OUT
        return FALSE;
    }

    js=mr_mip->pmod8%4-2;     /* 1 mod 4 or 3 mod 4 prime? */

    incr_cuda(_MIPP_ mr_mip->modulus,js,mr_mip->w10);
    subdiv_cuda(_MIPP_ mr_mip->w10,4,mr_mip->w10);    /* (p+/-1)/4 */

    if (js==1)
    { /* 3 mod 4 primes - do a quick and dirty sqrt(x)=x^(p+1)/4 mod p */
        nres_cuda(_MIPP_ w,mr_mip->w2);
        copy_cuda(mr_mip->one,w);
        forever
        { /* Simple Right-to-Left exponentiation */

            if (mr_mip->user!=NULL) (*mr_mip->user)();
            if (subdiv_cuda(_MIPP_ mr_mip->w10,2,mr_mip->w10)!=0)
                nres_modmult_cuda(_MIPP_ w,mr_mip->w2,w);
            if (mr_mip->ERNUM || size_cuda(mr_mip->w10)==0) break;
            nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w2);
        }
 
 /*     nres_moddiv_cuda(_MIPP_ mr_mip->one,w,mr_mip->w11); 
        nres_modadd_cuda(_MIPP_ mr_mip->w11,w,mr_mip->w3);  
        nres_lucas_cuda(_MIPP_ mr_mip->w3,mr_mip->w10,w,w);
        nres_modadd_cuda(_MIPP_ mr_mip->w11,mr_mip->one,mr_mip->w11); 
        nres_moddiv_cuda(_MIPP_ w,mr_mip->w11,w); */
    } 
    else
    { /* 1 mod 4 primes */
        for (t=1; ;t++)
        { /* t=1.5 on average */
            if (t==1) copy_cuda(w,mr_mip->w4);
            else
            {
                premult_cuda(_MIPP_ w,t,mr_mip->w4);
                divide_cuda(_MIPP_ mr_mip->w4,mr_mip->modulus,mr_mip->modulus);
                premult_cuda(_MIPP_ mr_mip->w4,t,mr_mip->w4);
                divide_cuda(_MIPP_ mr_mip->w4,mr_mip->modulus,mr_mip->modulus);
            }

            decr_cuda(_MIPP_ mr_mip->w4,4,mr_mip->w1);
            if (jack_cuda(_MIPP_ mr_mip->w1,mr_mip->modulus)==js) break;
            if (mr_mip->ERNUM) break;
        }
    
        decr_cuda(_MIPP_ mr_mip->w4,2,mr_mip->w3);
        nres_cuda(_MIPP_ mr_mip->w3,mr_mip->w3);
        nres_lucas_cuda(_MIPP_ mr_mip->w3,mr_mip->w10,w,w); /* heavy lifting done here */
        if (t!=1)
        {
            convert_cuda(_MIPP_ t,mr_mip->w11);
            nres_cuda(_MIPP_ mr_mip->w11,mr_mip->w11);
            nres_moddiv_cuda(_MIPP_ w,mr_mip->w11,w);
        }
    }
    
    MR_OUT
    return TRUE;
}

__device__ BOOL sqroot_cuda(_MIPD_ big x,big p,big w)
{ /* w = sqrt(x) mod p */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(101)

    if (subdivisible_cuda(_MIPP_ p,2))
    { /* p must be odd */
        zero_cuda(w);
        MR_OUT
        return FALSE;
    }

    prepare_monty_cuda(_MIPP_ p);
    nres_cuda(_MIPP_ x,w);
    if (nres_sqroot_cuda(_MIPP_ w,w))
    {
        redc_cuda(_MIPP_ w,w);
        MR_OUT
        return TRUE;
    }

    zero_cuda(w);
    MR_OUT
    return FALSE;
}


#endif

#ifndef mrlucas_c
#define mrlucas_c

__device__ void nres_lucas_cuda(_MIPD_ big p,big r,big vp,big v)
{
    int i,nb;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(107)

    if (size_cuda(r)==0) 
    {
        zero_cuda(vp);
        convert_cuda(_MIPP_ 2,v);
        nres_cuda(_MIPP_ v,v);
        MR_OUT
        return;
    }
    if (size_cuda(r)==1 || size_cuda(r)==(-1))
    { /* note - sign of r doesn't matter */
        convert_cuda(_MIPP_ 2,vp);
        nres_cuda(_MIPP_ vp,vp);
        copy_cuda(p,v);
        MR_OUT
        return;
    }

    copy_cuda(p,mr_mip->w3);
    
    convert_cuda(_MIPP_ 2,mr_mip->w4);
    nres_cuda(_MIPP_ mr_mip->w4,mr_mip->w4);     /* w4=2 */

    copy_cuda(mr_mip->w4,mr_mip->w8);
    copy_cuda(mr_mip->w3,mr_mip->w9);

    copy_cuda(r,mr_mip->w1);
    insign_cuda(PLUS,mr_mip->w1);         
    decr_cuda(_MIPP_ mr_mip->w1,1,mr_mip->w1);

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        nb=logb2_cuda(_MIPP_ mr_mip->w1);
        for (i=nb-1;i>=0;i--)
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();

            if (mr_testbit_cuda(_MIPP_ mr_mip->w1,i))
            {
                nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w9,mr_mip->w8);
                nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,mr_mip->w8);
                nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w9,mr_mip->w9);
                nres_modsub_cuda(_MIPP_ mr_mip->w9,mr_mip->w4,mr_mip->w9);

            }
            else
            {
                nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w8,mr_mip->w9);
                nres_modsub_cuda(_MIPP_ mr_mip->w9,mr_mip->w3,mr_mip->w9);
                nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w8);
                nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w4,mr_mip->w8);
            }  
        }

#ifndef MR_ALWAYS_BINARY
    }
    else
    {
        expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w1)-1,mr_mip->w2);                                                                                                   

        while (!mr_mip->ERNUM && size_cuda(mr_mip->w2)!=0)
        { /* use binary method */
            if (mr_compare_cuda(mr_mip->w1,mr_mip->w2)>=0)
            { /* vp=v*vp-p, v=v*v-2 */ 
                nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w9,mr_mip->w8);
                nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,mr_mip->w8);
                nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w9,mr_mip->w9);
                nres_modsub_cuda(_MIPP_ mr_mip->w9,mr_mip->w4,mr_mip->w9);
                subtract_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w1);
            }
            else
            { /* v=v*vp-p, vp=vp*vp-2 */
                nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w8,mr_mip->w9);
                nres_modsub_cuda(_MIPP_ mr_mip->w9,mr_mip->w3,mr_mip->w9);
                nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w8);
                nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w4,mr_mip->w8);
            }
            subdiv_cuda(_MIPP_ mr_mip->w2,2,mr_mip->w2);
        }
    }
#endif

    copy_cuda(mr_mip->w9,v);
    if (v!=vp) copy_cuda(mr_mip->w8,vp);
    MR_OUT

}

__device__ void lucas_cuda(_MIPD_ big p,big r,big n,big vp,big v)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(108)
    prepare_monty_cuda(_MIPP_ n);
    nres_cuda(_MIPP_ p,mr_mip->w3);
    nres_lucas_cuda(_MIPP_ mr_mip->w3,r,mr_mip->w8,mr_mip->w9);
    redc_cuda(_MIPP_ mr_mip->w9,v);
    if (v!=vp) redc_cuda(_MIPP_ mr_mip->w8,vp);
    MR_OUT
}

#include<iostream>
#include<cstdio>
#include<cmath>
#include<complex>
#include<cstdlib>

const double pi=acos(-1);

void FFT_cuda(Comp *a, int *r, int v, int n)
{
    for(int i=0;i<n;i++) 
        if(i<r[i])
        {
            Comp tmp=a[i];
            a[i]=a[r[i]];
            a[r[i]] = tmp;
        }
    for(int i=1;i<n;i*=2)
	{
		Comp wn;
        wn.r=cos(pi/i);
        wn.i=v*sin(pi/i);
		int p=i*2;
		for(int j=0;j<n;j+=p)
		{
			Comp w;
            w.r=1;
            w.i = 0;
            for (int k = 0; k < i; k++)
            {
				Comp x=a[j+k],y=w*a[i+j+k];
				a[j+k]=x+y;a[i+j+k]=x-y;
				w=w*wn;
			}
		}
	}
}

#endif

#ifndef mrarth0_c
#define mrarth0_c

__device__ void mr_padd_cuda(_MIPD_ big x,big y,big z)
{ /*  add_cuda two  big numbers, z=x+y where *
   *  x and y are positive              */
    int i,lx,ly,lz,la;
    mr_small carry,psum;
    mr_small *gx,*gy,*gz; 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    lx = (int)x->len;
    ly = (int)y->len;
    
    if (ly>lx)
    {
        lz=ly;
        la=lx;
        if (x!=z) copy_cuda(y,z); 
        else la=ly;  
    }
    else
    {
        lz=lx;
        la=ly;
        if (y!=z) copy_cuda(x,z);
        else la=lx;
    }
    carry=0;
    z->len=lz;
    gx=x->w; gy=y->w; gz=z->w;
    if (lz<mr_mip->nib || !mr_mip->check) z->len++;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0) 
    {
#endif
        for (i=0;i<la;i++)
        { /* add_cuda by columns to length of the smaller number */
            psum=gx[i]+gy[i]+carry;
            if (psum>gx[i]) carry=0;
            else if (psum<gx[i]) carry=1;
            gz[i]=psum;
        }
        for (;i<lz && carry>0;i++ )
        { /* add_cuda by columns to the length of larger number (if there is a carry) */
            psum=gx[i]+gy[i]+carry;
            if (psum>gx[i]) carry=0;
            else if (psum<gx[i]) carry=1;
            gz[i]=psum;
        }
        if (carry)
        { /* carry left over - possible overflow */
            if (mr_mip->check && i>=mr_mip->nib)
            {
                mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
                return;
            }
            gz[i]=carry;
        }
#ifndef MR_SIMPLE_BASE
    }
    else
    {
        for (i=0;i<la;i++)
        { /* add_cuda by columns */
            psum=gx[i]+gy[i]+carry;
            carry=0;
            if (psum>=mr_mip->base)
            { /* set carry */
                carry=1;
                psum-=mr_mip->base;
            }
            gz[i]=psum;
        }
        for (;i<lz && carry>0;i++)
        {
            psum=gx[i]+gy[i]+carry;
            carry=0;
            if (psum>=mr_mip->base)
            { /* set carry */
                carry=1;
                psum-=mr_mip->base;
            }
            gz[i]=psum;
        }
        if (carry)
        { /* carry left over - possible overflow */
            if (mr_mip->check && i>=mr_mip->nib)
            {
                mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
                return;
            }
            gz[i]=carry;
        }
    }
#endif
    if (gz[z->len-1]==0) z->len--;

}

__device__ void mr_psub_cuda(_MIPD_ big x,big y,big z)
{  /*  subtract_cuda two big numbers z=x-y      *
    *  where x and y are positive and x>y  */
    int i,lx,ly;
    mr_small borrow,pdiff;
    mr_small *gx,*gy,*gz;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    lx = (int)x->len;
    ly = (int)y->len;
    if (ly>lx)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NEG_RESULT);
        return;
    }
    if (y!=z) copy_cuda(x,z);
    else ly=lx;
    z->len=lx;
    gx=x->w; gy=y->w; gz=z->w;
    borrow=0;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif    
        for (i=0;i<ly || borrow>0;i++)
        { /* subtract_cuda by columns */
            if (i>lx)
            {
                mr_berror_cuda(_MIPP_ MR_ERR_NEG_RESULT);
                return;
            }
            pdiff=gx[i]-gy[i]-borrow;
            if (pdiff<gx[i]) borrow=0;
            else if (pdiff>gx[i]) borrow=1;
            gz[i]=pdiff;
        }
#ifndef MR_SIMPLE_BASE
    }
    else for (i=0;i<ly || borrow>0;i++)
    { /* subtract_cuda by columns */
        if (i>lx)
        {
            mr_berror_cuda(_MIPP_ MR_ERR_NEG_RESULT);
            return;
        }
        pdiff=gy[i]+borrow;
        borrow=0;
        if (gx[i]>=pdiff) pdiff=gx[i]-pdiff;
        else
        { /* set borrow */
            pdiff=mr_mip->base+gx[i]-pdiff;
            borrow=1;
        }
        gz[i]=pdiff;
    }
#endif
    mr_lzero_cuda(z);
}

__device__ static void mr_select_cuda(_MIPD_ big x,int d,big y,big z)
{ /* perform required add_cuda or subtract_cuda operation */
    int sx,sy,sz,jf,xgty;
#ifdef MR_FLASH
    if (mr_notint_cuda(x) || mr_notint_cuda(y))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
        return;
    }
#endif
    sx=exsign_cuda(x);
    sy=exsign_cuda(y);
    sz=0;
    x->len&=MR_OBITS;  /* force operands to be positive */
    y->len&=MR_OBITS;
    xgty=mr_compare_cuda(x,y);
    jf=(1+sx)+(1+d*sy)/2;
    switch (jf)
    { /* branch according to signs of operands */
    case 0:
        if (xgty>=0)
            mr_padd_cuda(_MIPP_ x,y,z);
        else
            mr_padd_cuda(_MIPP_ y,x,z);
        sz=MINUS;
        break;
    case 1:
        if (xgty<=0)
        {
            mr_psub_cuda(_MIPP_ y,x,z);
            sz=PLUS;
        }
        else
        {
            mr_psub_cuda(_MIPP_ x,y,z);
            sz=MINUS;
        }
        break;
    case 2:
        if (xgty>=0)
        {
            mr_psub_cuda(_MIPP_ x,y,z);
            sz=PLUS;
        }
        else
        {
            mr_psub_cuda(_MIPP_ y,x,z);
            sz=MINUS;
        }
        break;
    case 3:
        if (xgty>=0)
            mr_padd_cuda(_MIPP_ x,y,z);
        else
            mr_padd_cuda(_MIPP_ y,x,z);
        sz=PLUS;
        break;
    }
    if (sz<0) z->len^=MR_MSBIT;         /* set sign of result         */
    if (x!=z && sx<0) x->len^=MR_MSBIT; /* restore signs to operands  */
    if (y!=z && y!=x && sy<0) y->len^=MR_MSBIT;
}

__device__ void add_cuda(_MIPD_ big x,big y,big z)
{  /* add_cuda two signed big numbers together z=x+y */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(27)

    mr_select_cuda(_MIPP_ x,PLUS,y,z);

    MR_OUT
}

__device__ void subtract_cuda(_MIPD_ big x,big y,big z)
{ /* subtract_cuda two big signed numbers z=x-y */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(28)

    mr_select_cuda(_MIPP_ x,MINUS,y,z);

    MR_OUT
}

__device__ void incr_cuda(_MIPD_ big x,int n,big z)
{  /* add_cuda int to big number: z=x+n */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(7)

    convert_cuda(_MIPP_ n,mr_mip->w0);
    mr_select_cuda(_MIPP_ x,PLUS,mr_mip->w0,z);

    MR_OUT
}

__device__ void decr_cuda(_MIPD_ big x,int n,big z)
{  /* subtract_cuda int from big number: z=x-n */   
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(8)

    convert_cuda(_MIPP_ n,mr_mip->w0);
    mr_select_cuda(_MIPP_ x,MINUS,mr_mip->w0,z);

    MR_OUT
}



#endif

#ifndef mrarth1_c
#define mrarth1_c

#ifdef MR_FP
#include <math.h>
#endif

#ifdef MR_WIN64
#include <intrin.h>
#endif

#ifdef MR_FP_ROUNDING
#ifdef __GNUC__
#include <ieeefp.h>
#endif

/* Invert n and set FP rounding. 
 * Set to round up
 * Calculate 1/n
 * set to round down (towards zero_cuda)
 * If rounding cannot be controlled, this function returns 0.0 */

__device__ mr_large mr_invert_cuda(mr_small n)
{
    mr_large inn;
    int up=  0x1BFF; 

#ifdef _MSC_VER
  #ifdef MR_NOASM
#define NO_EXTENDED
  #endif 
#endif

#ifdef NO_EXTENDED
    int down=0x1EFF;
#else
    int down=0x1FFF;
#endif

#ifdef __TURBOC__
    asm
    {
        fldcw WORD PTR up
        fld1
        fld QWORD PTR n;
        fdiv
        fstp TBYTE PTR inn;
        fldcw WORD PTR down;
    }
    return inn;   
#endif
#ifdef _MSC_VER
    _asm
    {
        fldcw WORD PTR up
        fld1
        fld QWORD PTR n;
        fdiv
        fstp QWORD  PTR inn;
        fldcw WORD PTR down;
    }
    return inn;   
#endif
#ifdef __GNUC__
#ifdef i386
    __asm__ __volatile__ (
    "fldcw %2\n"
    "fld1\n"
    "fldl %1\n"
    "fdivrp\n"
    "fstpt %0\n"
    "fldcw %3\n"
    : "=m"(inn)
    : "m"(n),"m"(up),"m"(down)
    : "memory"
    );
    return inn;   
#else
    fpsetround(FP_RP);
    inn=(mr_large)1.0/n;
    fpsetround(FP_RZ);
    return inn;
#endif
#endif
    return 0.0L;   
}

#endif

__device__ void mr_pmul_cuda(_MIPD_ big x,mr_small sn,big z)
{ 
    int m,xl;
    mr_lentype sx;
    mr_small carry,*xg,*zg;

#ifdef MR_ITANIUM
    mr_small tm;
#endif
#ifdef MR_WIN64
    mr_small tm;
#endif
#ifdef MR_NOASM
    union doubleword dble;
    mr_large dbled;
    mr_large ldres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (x!=z)
    {
        zero_cuda(z);
        if (sn==0) return;
    }
    else if (sn==0)
    {
        zero_cuda(z);
        return;
    }
    m=0;
    carry=0;
    sx=x->len&MR_MSBIT;
    xl=(int)(x->len&MR_OBITS);

#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0) 
    {
#endif
#ifndef MR_NOFULLWIDTH
        xg=x->w; zg=z->w;
/* inline 8086 assembly - substitutes for loop below */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
        ASM cld
        ASM mov cx,xl
        ASM or cx,cx
        ASM je out1
#ifdef MR_LMM
        ASM push ds
        ASM push es
        ASM les di,DWORD PTR zg
        ASM lds si,DWORD PTR xg
#else
        ASM mov ax,ds
        ASM mov es,ax
        ASM mov di,zg
        ASM mov si,xg
#endif
        ASM mov bx,sn
        ASM push bp
        ASM xor bp,bp
    tcl1:
        ASM lodsw
        ASM mul bx
        ASM add_cuda ax,bp
        ASM adc dx,0
        ASM stosw
        ASM mov bp,dx
        ASM loop tcl1

        ASM mov ax,bp
        ASM pop bp
#ifdef MR_LMM
        ASM pop es
        ASM pop ds
#endif
        ASM mov carry,ax
     out1: 
#endif
#if INLINE_ASM == 2
        ASM cld
        ASM mov cx,xl
        ASM or cx,cx
        ASM je out1
#ifdef MR_LMM
        ASM push ds
        ASM push es
        ASM les di,DWORD PTR zg
        ASM lds si,DWORD PTR xg
#else
        ASM mov ax,ds
        ASM mov es,ax
        ASM mov di,zg
        ASM mov si,xg
#endif
        ASM mov ebx,sn
        ASM push ebp
        ASM xor ebp,ebp
    tcl1:
        ASM lodsd
        ASM mul ebx
        ASM add_cuda eax,ebp
        ASM adc edx,0
        ASM stosd
        ASM mov ebp,edx
        ASM loop tcl1

        ASM mov eax,ebp
        ASM pop ebp
#ifdef MR_LMM
        ASM pop es
        ASM pop ds
#endif
        ASM mov carry,eax
     out1: 
#endif
#if INLINE_ASM == 3
        ASM mov ecx,xl
        ASM or ecx,ecx
        ASM je out1
        ASM mov ebx,sn
        ASM mov edi,zg
        ASM mov esi,xg
        ASM push ebp
        ASM xor ebp,ebp
    tcl1:
        ASM mov eax,[esi]
        ASM add_cuda esi,4
        ASM mul ebx
        ASM add_cuda eax,ebp
        ASM adc edx,0
        ASM mov [edi],eax
        ASM add_cuda edi,4
        ASM mov ebp,edx
        ASM dec ecx
        ASM jnz tcl1

        ASM mov eax,ebp
        ASM pop ebp
        ASM mov carry,eax
     out1: 
#endif
#if INLINE_ASM == 4

        ASM (
           "movl %4,%%ecx\n"
           "orl  %%ecx,%%ecx\n"
           "je 1f\n"
           "movl %3,%%ebx\n"
           "movl %1,%%edi\n"
           "movl %2,%%esi\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"  
        "0:\n"  
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n"
           "mull %%ebx\n"
           "addl %%ebp,%%eax\n"
           "adcl $0,%%edx\n"
           "movl %%eax,(%%edi)\n"
           "addl $4,%%edi\n"
           "movl %%edx,%%ebp\n"
           "decl %%ecx\n"
           "jnz 0b\n"
 
           "movl %%ebp,%%eax\n"
           "popl %%ebp\n"
           "movl %%eax,%0\n"
        "1:"  
        :"=m"(carry)
        :"m"(zg),"m"(xg),"m"(sn),"m"(xl)
        :"eax","edi","esi","ebx","ecx","edx","memory"
        );

#endif
#endif
#ifndef INLINE_ASM
        for (m=0;m<xl;m++)
#ifdef MR_NOASM
        {
            dble.d=(mr_large)x->w[m]*sn+carry;
            carry=dble.h[MR_TOP];
            z->w[m]=dble.h[MR_BOT];
        }
#else
            carry=muldvd_cuda(x->w[m],sn,carry,&z->w[m]);
#endif
#endif
        if (carry>0)
        {
            m=xl;
            if (m>=mr_mip->nib && mr_mip->check)
            {
                mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
                return;
            }
            z->w[m]=carry;
            z->len=m+1;
        }
        else z->len=xl;
#endif
#ifndef MR_SIMPLE_BASE
    }
    else while (m<xl || carry>0)
    { /* multiply_cuda each digit of x by n */ 
    
        if (m>mr_mip->nib && mr_mip->check)
        {
            mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
            return;
        }
#ifdef MR_NOASM
        dbled=(mr_large)x->w[m]*sn+carry;
 #ifdef MR_FP_ROUNDING
        carry=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
 #else
  #ifndef MR_FP
        if (mr_mip->base==mr_mip->base2)
          carry=(mr_small)(dbled>>mr_mip->lg2b);
        else 
  #endif  
          carry=(mr_small)MR_LROUND(dbled/mr_mip->base);
 #endif
        z->w[m]=(mr_small)(dbled-(mr_large)carry*mr_mip->base);
#else
 #ifdef MR_FP_ROUNDING
        carry=imuldiv(x->w[m],sn,carry,mr_mip->base,mr_mip->inverse_base,&z->w[m]);
 #else
        carry=muldiv_cuda(x->w[m],sn,carry,mr_mip->base,&z->w[m]);
 #endif
#endif

        m++;
        z->len=m;
    }
#endif
    if (z->len!=0) z->len|=sx;
}

__device__ void premult_cuda(_MIPD_ big x,int n,big z)
{ /* premultiply a big number by an int z=x.n */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(9)


#ifdef MR_FLASH
    if (mr_notint_cuda(x))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
        MR_OUT
        return;
    }
#endif
    if (n==0)  /* test for some special cases  */
    {
        zero_cuda(z);
        MR_OUT
        return;
    }
    if (n==1)
    {
        copy_cuda(x,z);
        MR_OUT
        return;
    }
    if (n<0)
    {
        n=(-n);
        mr_pmul_cuda(_MIPP_ x,(mr_small)n,z);
        if (z->len!=0) z->len^=MR_MSBIT;
    }
    else mr_pmul_cuda(_MIPP_ x,(mr_small)n,z);
    MR_OUT
}

#ifdef MR_FP_ROUNDING
__device__ mr_small mr_sdiv_cuda(_MIPD_ big x,mr_small sn,mr_large isn,big z)
#else
__device__ mr_small mr_sdiv_cuda(_MIPD_ big x,mr_small sn,big z)
#endif
{
    int i,xl;
    mr_small sr,*xg,*zg;
#ifdef MR_NOASM
    union doubleword dble;
    mr_large dbled;
    mr_large ldres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    sr=0;
    xl=(int)(x->len&MR_OBITS);
    if (x!=z) zero_cuda(z);
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0) 
    {
#endif
#ifndef MR_NOFULLWIDTH
        xg=x->w; zg=z->w;
/* inline - substitutes for loop below */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
        ASM std
        ASM mov cx,xl
        ASM or cx,cx
        ASM je out2
        ASM mov bx,cx
        ASM shl bx,1
        ASM sub bx,2
#ifdef MR_LMM
        ASM push ds
        ASM push es
        ASM les di,DWORD PTR zg
        ASM lds si,DWORD PTR xg
#else
        ASM mov ax,ds
        ASM mov es,ax
        ASM mov di,zg
        ASM mov si,xg
#endif
        ASM add_cuda si,bx
        ASM add_cuda di,bx
        ASM mov bx,sn
        ASM push bp
        ASM xor bp,bp
    tcl2:
        ASM mov dx,bp
        ASM lodsw
        ASM div bx
        ASM mov bp,dx
        ASM stosw
        ASM loop tcl2

        ASM mov ax,bp
        ASM pop bp
#ifdef MR_LMM
        ASM pop es
        ASM pop ds
#endif
        ASM mov sr,ax
     out2:
        ASM cld
#endif
#if INLINE_ASM == 2
        ASM std
        ASM mov cx,xl
        ASM or cx,cx
        ASM je out2
        ASM mov bx,cx
        ASM shl bx,2
        ASM sub bx,4
#ifdef MR_LMM
        ASM push ds
        ASM push es
        ASM les di,DWORD PTR zg
        ASM lds si,DWORD PTR xg
#else
        ASM mov ax,ds
        ASM mov es,ax
        ASM mov di, zg
        ASM mov si, xg
#endif
        ASM add_cuda si,bx
        ASM add_cuda di,bx
        ASM mov ebx,sn
        ASM push ebp
        ASM xor ebp,ebp
    tcl2:
        ASM mov edx,ebp
        ASM lodsd
        ASM div ebx
        ASM mov ebp,edx
        ASM stosd
        ASM loop tcl2

        ASM mov eax,ebp
        ASM pop ebp
#ifdef MR_LMM
        ASM pop es
        ASM pop ds
#endif
        ASM mov sr,eax
     out2: 
        ASM cld
#endif
#if INLINE_ASM == 3
        ASM mov ecx,xl
        ASM or ecx,ecx
        ASM je out2
        ASM mov ebx,ecx
        ASM shl ebx,2
        ASM mov esi, xg
        ASM add_cuda esi,ebx
        ASM mov edi, zg
        ASM add_cuda edi,ebx
        ASM mov ebx,sn
        ASM push ebp
        ASM xor ebp,ebp
    tcl2:
        ASM sub esi,4
        ASM mov edx,ebp
        ASM mov eax,[esi]
        ASM div ebx
        ASM sub edi,4
        ASM mov ebp,edx
        ASM mov [edi],eax
        ASM dec ecx
        ASM jnz tcl2

        ASM mov eax,ebp
        ASM pop ebp
        ASM mov sr,eax
     out2:
        ASM nop
#endif
#if INLINE_ASM == 4

        ASM (
           "movl %4,%%ecx\n"
           "orl  %%ecx,%%ecx\n"
           "je 3f\n"
           "movl %%ecx,%%ebx\n"
           "shll $2,%%ebx\n"
           "movl %2,%%esi\n"
           "addl %%ebx,%%esi\n"
           "movl %1,%%edi\n"
           "addl %%ebx,%%edi\n"
           "movl %3,%%ebx\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"  
         "2:\n"  
           "subl $4,%%esi\n"
           "movl %%ebp,%%edx\n"
           "movl (%%esi),%%eax\n"
           "divl %%ebx\n"
           "subl $4,%%edi\n"
           "movl %%edx,%%ebp\n"
           "movl %%eax,(%%edi)\n"
           "decl %%ecx\n"
           "jnz 2b\n"
 
           "movl %%ebp,%%eax\n"
           "popl %%ebp\n"
           "movl %%eax,%0\n"
        "3:"
           "nop"  
        :"=m"(sr)
        :"m"(zg),"m"(xg),"m"(sn),"m"(xl)
        :"eax","edi","esi","ebx","ecx","edx","memory"
        );
#endif
#endif
#ifndef INLINE_ASM
        for (i=xl-1;i>=0;i--)
        {
#ifdef MR_NOASM
            dble.h[MR_BOT]=x->w[i];
            dble.h[MR_TOP]=sr;
            z->w[i]=(mr_small)(dble.d/sn);
            sr=(mr_small)(dble.d-(mr_large)z->w[i]*sn);
#else
            z->w[i]=muldvm_cuda(sr,x->w[i],sn,&sr);
#endif
        }
#endif
#endif
#ifndef MR_SIMPLE_BASE
    }
    else for (i=xl-1;i>=0;i--)
    { /* divide_cuda each digit of x by n */
#ifdef MR_NOASM
        dbled=(mr_large)sr*mr_mip->base+x->w[i];
#ifdef MR_FP_ROUNDING
        z->w[i]=(mr_small)MR_LROUND(dbled*isn);
#else
        z->w[i]=(mr_small)MR_LROUND(dbled/sn);
#endif
        sr=(mr_small)(dbled-(mr_large)z->w[i]*sn);
#else
#ifdef MR_FP_ROUNDING
        z->w[i]=imuldiv(sr,mr_mip->base,x->w[i],sn,isn,&sr);
#else
        z->w[i]=muldiv_cuda(sr,mr_mip->base,x->w[i],sn,&sr);
#endif
#endif
    }
#endif
    z->len=x->len;
    mr_lzero_cuda(z);
    return sr;
}
         
__device__ int subdiv_cuda(_MIPD_ big x,int n,big z)
{  /*  subdivide a big number by an int   z=x/n  *
    *  returns int remainder                     */ 
    mr_lentype sx;
#ifdef MR_FP_ROUNDING
    mr_large in;
#endif
    int r,i,msb;
    mr_small lsb;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return 0;

    MR_IN(10)
#ifdef MR_FLASH
    if (mr_notint_cuda(x)) mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
#endif
    if (n==0) mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
    if (mr_mip->ERNUM) 
    {
        MR_OUT
        return 0;
    }

    if (x->len==0)
    {
        zero_cuda(z);
        MR_OUT
        return 0;
    }
    if (n==1) /* special case */
    {
        copy_cuda(x,z);
        MR_OUT
        return 0;
    }
    sx=(x->len&MR_MSBIT);
    if (n==2 && mr_mip->base==0)
    { /* fast division by 2 using shifting */
#ifndef MR_NOFULLWIDTH

/* I don't want this code upsetting the compiler ... */
/* mr_mip->base==0 can't happen with MR_NOFULLWIDTH  */

        copy_cuda(x,z);
        msb=(int)(z->len&MR_OBITS)-1;
        r=(int)z->w[0]&1;
        for (i=0;;i++)
        {
            z->w[i]>>=1;
            if (i==msb) 
            {
                if (z->w[i]==0) mr_lzero_cuda(z);
                break;
            }
            lsb=z->w[i+1]&1;
            z->w[i]|=(lsb<<(MIRACL-1));
        }

        MR_OUT
        if (sx==0) return r;
        else       return (-r);
#endif
    }

#ifdef MR_FP_ROUNDING
    in=mr_invert(n);
#endif
    if (n<0)
    {
        n=(-n);
#ifdef MR_FP_ROUNDING
        r=(int)mr_sdiv_cuda(_MIPP_ x,(mr_small)n,in,z);
#else
        r=(int)mr_sdiv_cuda(_MIPP_ x,(mr_small)n,z);
#endif
        if (z->len!=0) z->len^=MR_MSBIT;
    }
#ifdef MR_FP_ROUNDING
    else r=(int)mr_sdiv_cuda(_MIPP_ x,(mr_small)n,in,z);
#else
    else r=(int)mr_sdiv_cuda(_MIPP_ x,(mr_small)n,z);
#endif
    MR_OUT
    if (sx==0) return r;
    else       return (-r);
}

__device__ int remain_cuda(_MIPD_ big x,int n)
{ /* return integer remainder when x divided by n */
    int r;
    mr_lentype sx;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(88);

    sx=(x->len&MR_MSBIT);

    if (n==2 && MR_REMAIN(mr_mip->base,2)==0)
    { /* fast odd/even check if base is even */
        MR_OUT
        if ((int)MR_REMAIN(x->w[0],2)==0) return 0;
        else
        {
            if (sx==0) return 1;
            else       return (-1);
        } 
    }
    if (n==8 && MR_REMAIN(mr_mip->base,8)==0)
    { /* fast check */
        MR_OUT
        r=(int)MR_REMAIN(x->w[0],8);
        if (sx!=0) r=-r;
        return r;
    }
    
    copy_cuda(x,mr_mip->w0);
    r=subdiv_cuda(_MIPP_ mr_mip->w0,n,mr_mip->w0);
    MR_OUT
    return r;
}

__device__ BOOL subdivisible_cuda(_MIPD_ big x,int n)
{
    if (remain_cuda(_MIPP_ x,n)==0) return TRUE;
    else                return FALSE;
}

__device__ int hamming_cuda(_MIPD_ big x)
{
    int h;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return 0;
    MR_IN(148);
    h=0;
    copy_cuda(x,mr_mip->w1);
    absol_cuda(mr_mip->w1,mr_mip->w1);
    while (size_cuda(mr_mip->w1)!=0)
        h+=subdiv_cuda(_MIPP_ mr_mip->w1,2,mr_mip->w1);
    
    MR_OUT
    return h;
}

__device__ void bytes_to_big_cuda(_MIPD_ int len,const char *ptr,big x)
{ /* convert_cuda len bytes into a big           *
   * The first byte is the Most significant */
    int i,j,m,n,r;
    unsigned int dig;
    unsigned char ch;
    mr_small wrd;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(140);

    zero_cuda(x);

    if (len<=0)
    {
        MR_OUT
        return;
    }
/* remove leading zeros.. */

    while (*ptr==0) 
    {
        ptr++; len--;
        if (len==0) 
        {
            MR_OUT
            return;
        } 
    }

#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    { /* pack bytes directly into big */
#endif
#ifndef MR_NOFULLWIDTH
        m=MIRACL/8;  
        n=len/m;

        r=len%m;
		wrd=(mr_small)0;  
        if (r!=0)
        {
            n++; 
            for (j=0;j<r;j++) {wrd<<=8; wrd|=MR_TOBYTE(*ptr++); }
        }
        x->len=n;
        if (n>mr_mip->nib && mr_mip->check)
        {
            mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
            MR_OUT
            return;
        }
        if (r!=0) 
        {
            n--;
            x->w[n]=wrd;
        }

        for (i=n-1;i>=0;i--)
        {
            for (j=0;j<m;j++) { wrd<<=8; wrd|=MR_TOBYTE(*ptr++); }
            x->w[i]=wrd;
        }
        mr_lzero_cuda(x);     /* needed */
#endif
#ifndef MR_SIMPLE_BASE
    }
    else
    {
        for (i=0;i<len;i++)
        {
            if (mr_mip->ERNUM) break;
#if MIRACL==8
            mr_shift_cuda(_MIPP_ x,1,x);
#else
            premult_cuda(_MIPP_ x,256,x);
#endif
            ch=MR_TOBYTE(ptr[i]);
            dig=ch;  
            incr_cuda(_MIPP_ x,(int)dig,x);
        }
    }
#endif
    MR_OUT
} 

__device__ int big_to_bytes_cuda(_MIPD_ int max,big x,char *ptr,BOOL justify)
{ /* convert_cuda positive big into octet string */
    int i,j,r,m,n,len,start;
    unsigned int dig;
    unsigned char ch;
    mr_small wrd;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM || max<0) return 0;

	if (max==0 && justify) return 0; 
	if (size_cuda(x)==0)
	{
		if (justify)
		{
			for (i=0;i<max;i++) ptr[i]=0;
			return max;
		}
		return 0;
	}
     
    MR_IN(141);

    mr_lzero_cuda(x);        /* should not be needed.... */
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
        m=MIRACL/8;
        n=(int)(x->len&MR_OBITS);
        n--;
        len=n*m;
        wrd=x->w[n]; /* most significant */
        r=0;
        while (wrd!=(mr_small)0) { r++; wrd>>=8; len++;}
        r%=m;

        if (max>0 && len>max)
        {
            mr_berror_cuda(_MIPP_ MR_ERR_TOO_BIG);
            MR_OUT
            return 0; 
        }

        if (justify)
        {
            start=max-len;
            for (i=0;i<start;i++) ptr[i]=0; 
        }
        else start=0;
        
        if (r!=0)
        {
            wrd=x->w[n--];
            for (i=r-1;i>=0;i--)
            {
                ptr[start+i]=(char)(wrd&0xFF);
                wrd>>=8;
            }  
        }

        for (i=r;i<len;i+=m)
        {
            wrd=x->w[n--];
            for (j=m-1;j>=0;j--)
            {
                ptr[start+i+j]=(char)(wrd&0xFF);
                wrd>>=8;
            }
        }
#endif
#ifndef MR_SIMPLE_BASE
    }
    else
    {
        copy_cuda(x,mr_mip->w1);
        for (len=0;;len++)
        {
            if (mr_mip->ERNUM) break;

            if (size_cuda(mr_mip->w1)==0)
            {
                if (justify)
                {
                   if (len==max) break;
                }
                else break;
            }

            if (max>0 && len>=max)
            {
                mr_berror_cuda(_MIPP_ MR_ERR_TOO_BIG);
                MR_OUT
                return 0; 
            }
#if MIRACL==8
            ch=mr_mip->w1->w[0];
            mr_shift_cuda(_MIPP_ mr_mip->w1,-1,mr_mip->w1);
#else
            dig=(unsigned int)subdiv_cuda(_MIPP_ mr_mip->w1,256,mr_mip->w1);
            ch=MR_TOBYTE(dig);
#endif
            for (i=len;i>0;i--) ptr[i]=ptr[i-1];
            ptr[0]=MR_TOBYTE(ch);
        }
    }
#endif
    MR_OUT
    if (justify) return max;
    else         return len;
}

#ifndef MR_NO_ECC_MULTIADD

/* Solinas's Joint Sparse Form */

__device__ void mr_jsf_cuda(_MIPD_ big k0,big k1,big u0p,big u0m,big u1p,big u1m)
{
    int j,u0,u1,d0,d1,l0,l1;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;   

    MR_IN(191)

    d0=d1=0;

    convert_cuda(_MIPP_ 1,mr_mip->w1);
    copy_cuda(k0,mr_mip->w2);
    copy_cuda(k1,mr_mip->w3);
    zero_cuda(u0p); zero_cuda(u0m); zero_cuda(u1p); zero_cuda(u1m);

    j=0;
    while (!mr_mip->ERNUM)
    {
        if (size_cuda(mr_mip->w2)==0 && d0==0 && size_cuda(mr_mip->w3)==0 && d1==0) break;
        l0=remain_cuda(_MIPP_ mr_mip->w2,8);
        l0=(l0+d0)&0x7;
        l1=remain_cuda(_MIPP_ mr_mip->w3,8);
        l1=(l1+d1)&0x7;

        if (l0%2==0) u0=0;
        else
        {
            u0=2-(l0%4);
            if ((l0==3 || l0==5) && l1%4==2) u0=-u0;
        }
        if (l1%2==0) u1=0;
        else
        {
            u1=2-(l1%4);
            if ((l1==3 || l1==5) && l0%4==2) u1=-u1;
        }
#ifndef MR_ALWAYS_BINARY
        if (mr_mip->base==mr_mip->base2)
        {
#endif
            if (u0>0) mr_addbit_cuda(_MIPP_ u0p,j);
            if (u0<0) mr_addbit_cuda(_MIPP_ u0m,j);
            if (u1>0) mr_addbit_cuda(_MIPP_ u1p,j);
            if (u1<0) mr_addbit_cuda(_MIPP_ u1m,j);

#ifndef MR_ALWAYS_BINARY
        }
        else
        {
            if (u0>0) add_cuda(_MIPP_ u0p,mr_mip->w1,u0p);
            if (u0<0) add_cuda(_MIPP_ u0m,mr_mip->w1,u0m);
            if (u1>0) add_cuda(_MIPP_ u1p,mr_mip->w1,u1p);
            if (u1<0) add_cuda(_MIPP_ u1m,mr_mip->w1,u1m);
        }
#endif
      
        if (d0+d0==1+u0) d0=1-d0;
        if (d1+d1==1+u1) d1=1-d1;

        subdiv_cuda(_MIPP_ mr_mip->w2,2,mr_mip->w2);
        subdiv_cuda(_MIPP_ mr_mip->w3,2,mr_mip->w3);

#ifndef MR_ALWAYS_BINARY
        if (mr_mip->base==mr_mip->base2)
#endif
            j++;
#ifndef MR_ALWAYS_BINARY
        else
            premult_cuda(_MIPP_ mr_mip->w1,2,mr_mip->w1);
#endif        
    }
    MR_OUT
    return;
}

#endif


#endif

#ifndef mrarth2_c
#define mrarth2_c

#ifdef MR_FP
#include <math.h>
#endif

#ifdef MR_WIN64
#include <intrin.h>
#endif


/* If a number has more than this number of digits, then squaring is faster */

#define SQR_FASTER_THRESHOLD 5

__device__ mr_small normalise_cuda(_MIPD_ big x,big y)
{ /* normalise_cuda divisor */
    mr_small norm,r;
#ifdef MR_FP
    mr_small dres;
#endif
    int len;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(4)

    if (x!=y) copy_cuda(x,y);
    len=(int)(y->len&MR_OBITS);
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
        if ((r=y->w[len-1]+1)==0) norm=1;
#ifdef MR_NOASM
        else norm=(mr_small)(((mr_large)1 << MIRACL)/r);
#else
        else norm=muldvm_cuda((mr_small)1,(mr_small)0,r,&r);
#endif
        if (norm!=1) mr_pmul_cuda(_MIPP_ y,norm,y);
#endif
#ifndef MR_SIMPLE_BASE
    }
    else
    {
        norm=MR_DIV(mr_mip->base,(mr_small)(y->w[len-1]+1));   
        if (norm!=1) mr_pmul_cuda(_MIPP_ y,norm,y);
    }
#endif
    MR_OUT
    return norm;
}

__device__ void multiply_cuda(_MIPD_ big x,big y,big z)
{  /*  multiply_cuda two big numbers: z=x.y  */
    int i,xl,yl,j,ti;
    mr_small carry,*xg,*yg,*w0g;

#ifdef MR_ITANIUM
    mr_small tm;
#endif
#ifdef MR_WIN64
    mr_small tm,tr;
#endif
    mr_lentype sz;
    big w0;
#ifdef MR_NOASM
    union doubleword dble;
    mr_large dbled;
    mr_large ldres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    if (y->len==0 || x->len==0) 
    {
        zero_cuda(z);
        return;
    }
    if (x!=mr_mip->w5 && y!=mr_mip->w5 && z==mr_mip->w5) w0=mr_mip->w5;
    else w0=mr_mip->w0;    /* local pointer */

    MR_IN(5)

#ifdef MR_FLASH
    if (mr_notint_cuda(x) || mr_notint_cuda(y))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
        MR_OUT
        return;
    }
#endif
    sz=((x->len&MR_MSBIT)^(y->len&MR_MSBIT));
    xl=(int)(x->len&MR_OBITS);
    yl=(int)(y->len&MR_OBITS);
    zero_cuda(w0);
    if (mr_mip->check && xl+yl>mr_mip->nib)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
        MR_OUT
        return;
    }
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
        xg=x->w; yg=y->w; w0g=w0->w; 
        if (x==y && xl>SQR_FASTER_THRESHOLD)    
                             /* extra hassle make it not    */
                             /* worth it for small numbers */
        { /* fast squaring */
            for (i=0;i<xl-1;i++)
            {  /* long multiplication */
           /* inline - substitutes for loop below */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
                ASM cld
                ASM mov dx,i
                ASM mov cx,xl
                ASM sub cx,dx
                ASM dec cx
                ASM shl dx,1
#ifdef MR_LMM
                ASM push ds
                ASM push es
                ASM les bx,DWORD PTR w0g
                ASM lds si,DWORD PTR xg
                ASM add_cuda si,dx
                ASM mov di,[si]
#else
                ASM mov bx,w0g
                ASM mov si,xg
                ASM add_cuda si,dx
                ASM mov di,[si]
#endif
                ASM add_cuda bx,dx
                ASM add_cuda bx,dx 
                ASM add_cuda bx,2
                ASM add_cuda si,2
                ASM push bp
                ASM xor bp,bp
              tcl4:
                ASM lodsw
                ASM mul di
                ASM add_cuda ax,bp
                ASM adc dx,0
#ifdef MR_LMM
                ASM add_cuda es:[bx],ax
#else
                ASM add_cuda [bx],ax 
#endif
                ASM adc dx,0
                ASM inc bx
                ASM inc bx 
                ASM mov bp,dx
                ASM loop tcl4

#ifdef MR_LMM
                ASM mov es:[bx],bp
                ASM pop bp
                ASM pop es
                ASM pop ds
#else
                ASM mov [bx],bp
                ASM pop bp
#endif
#endif
#if INLINE_ASM == 2
                ASM cld
                ASM mov dx,i
                ASM mov cx,xl
                ASM sub cx,dx
                ASM dec cx
                ASM shl dx,2
#ifdef MR_LMM
                ASM push ds
                ASM push es
                ASM les bx,DWORD PTR w0g
                ASM lds si,DWORD PTR xg
                ASM add_cuda si,dx
                ASM mov edi,[si]
#else
                ASM mov bx,w0g
                ASM mov si,xg
                ASM add_cuda si,dx
                ASM mov edi,[si]
#endif
                ASM add_cuda bx,dx
                ASM add_cuda bx,dx
                ASM add_cuda bx,4
                ASM add_cuda si,4   
                ASM push ebp
                ASM xor ebp,ebp
              tcl4:
                ASM lodsd
                ASM mul edi
                ASM add_cuda eax,ebp
                ASM adc edx,0
#ifdef MR_LMM
                ASM add_cuda es:[bx],eax
#else
                ASM add_cuda [bx],eax
#endif
                ASM adc edx,0
                ASM add_cuda bx,4
                ASM mov ebp,edx
                ASM loop tcl4

#ifdef MR_LMM
                ASM mov es:[bx],ebp
                ASM pop ebp
                ASM pop es
                ASM pop ds
#else
                ASM mov [bx],ebp
                ASM pop ebp
#endif
#endif
#if INLINE_ASM == 3
                ASM mov esi,i
                ASM mov ecx,xl
                ASM sub ecx,esi
                ASM dec ecx
                ASM shl esi,2
                ASM mov edx, xg
                ASM mov ebx,edx
                ASM add_cuda ebx,esi
                ASM mov edi,[ebx]
                ASM mov ebx,w0g
                ASM add_cuda ebx,esi
                ASM add_cuda esi,edx
                ASM sub ebx,edx
                ASM add_cuda esi,4  
                ASM sub ebx,4
                ASM push ebp
                ASM xor ebp,ebp
              tcl4:
                ASM mov eax,[esi]   /* optimized for Pentium */
                ASM add_cuda esi,4
                ASM mul edi
                ASM add_cuda eax,ebp
                ASM mov ebp,[esi+ebx]
                ASM adc edx,0
                ASM add_cuda ebp,eax
                ASM adc edx,0
                ASM mov [esi+ebx],ebp
                ASM dec ecx
                ASM mov ebp,edx
                ASM jnz tcl4

                ASM mov [esi+ebx+4],ebp
                ASM pop ebp
#endif
#if INLINE_ASM == 4
   ASM (
           "movl %0,%%esi\n"
           "movl %1,%%ecx\n"
           "subl %%esi,%%ecx\n"
           "decl %%ecx\n"
           "shll $2,%%esi\n"
           "movl %2,%%edx\n"
           "movl %%edx,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "movl (%%ebx),%%edi\n"
           "movl %3,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "addl %%edx,%%esi\n"
           "subl %%edx,%%ebx\n"
           "addl $4,%%esi\n"
           "subl $4,%%ebx\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"
         "tcl4:\n"
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n"
           "mull %%edi\n"
           "addl %%ebp,%%eax\n"
           "movl (%%esi,%%ebx),%%ebp\n"
           "adcl $0,%%edx\n"
           "addl %%eax,%%ebp\n"
           "adcl $0,%%edx\n"
           "movl %%ebp,(%%esi,%%ebx)\n"
           "decl %%ecx\n"
           "movl %%edx,%%ebp\n"
           "jnz tcl4\n"

           "movl %%ebp,4(%%esi,%%ebx)\n"
           "popl %%ebp\n"
           
        :
        :"m"(i),"m"(xl),"m"(xg),"m"(w0g)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif
#endif
#ifndef INLINE_ASM
                carry=0;
                for (j=i+1;j<xl;j++)
                { /* Only do above the diagonal */
#ifdef MR_NOASM
                    dble.d=(mr_large)x->w[i]*x->w[j]+carry+w0->w[i+j];
                    w0->w[i+j]=dble.h[MR_BOT];
                    carry=dble.h[MR_TOP];
#else
                    muldvd2_cuda(x->w[i],x->w[j],&carry,&w0->w[i+j]);
#endif
                }
                w0->w[xl+i]=carry;
#endif
            }
#ifdef INLINE_ASM
#if INLINE_ASM == 1
            ASM mov cx,xl
            ASM shl cx,1
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g  
#else
            ASM mov bx,w0g
#endif
          tcl5:
#ifdef MR_LMM
            ASM rcl WORD PTR es:[bx],1
#else
            ASM rcl WORD PTR [bx],1
#endif
            ASM inc bx
            ASM inc bx
            ASM loop tcl5

            ASM cld
            ASM mov cx,xl
#ifdef MR_LMM
            ASM les di,DWORD PTR w0g
            ASM lds si,DWORD PTR xg
#else
            ASM mov di,w0g
            ASM mov si,xg
#endif
       
            ASM xor bx,bx
          tcl7:
            ASM lodsw
            ASM mul ax
            ASM add_cuda ax,bx
            ASM adc dx,0
#ifdef MR_LMM
            ASM add_cuda es:[di],ax
#else
            ASM add_cuda [di],ax
#endif
            ASM adc dx,0
            ASM xor bx,bx
            ASM inc di
            ASM inc di
#ifdef MR_LMM
            ASM add_cuda es:[di],dx
#else
            ASM add_cuda [di],dx
#endif
            ASM adc bx,0
            ASM inc di
            ASM inc di
            ASM loop tcl7
#ifdef MR_LMM
            ASM pop es
            ASM pop ds
#endif
#endif
#if INLINE_ASM == 2
            ASM mov cx,xl
            ASM shl cx,1
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g
#else
            ASM mov bx,w0g
#endif
          tcl5:
#ifdef MR_LMM
            ASM rcl DWORD PTR es:[bx],1
#else
            ASM rcl DWORD PTR [bx],1
#endif
            ASM inc bx
            ASM inc bx
            ASM inc bx
            ASM inc bx
            ASM loop tcl5

            ASM cld
            ASM mov cx,xl
#ifdef MR_LMM
            ASM les di,DWORD PTR w0g
            ASM lds si,DWORD PTR xg
#else
            ASM mov di,w0g
            ASM mov si,xg
#endif
            ASM xor ebx,ebx
          tcl7:
            ASM lodsd
            ASM mul eax
            ASM add_cuda eax,ebx
            ASM adc edx,0
#ifdef MR_LMM
            ASM add_cuda es:[di],eax
#else
            ASM add_cuda [di],eax
#endif
            ASM adc edx,0
            ASM xor ebx,ebx
            ASM add_cuda di,4
#ifdef MR_LMM
            ASM add_cuda es:[di],edx
#else
            ASM add_cuda [di],edx
#endif
            ASM adc ebx,0
            ASM add_cuda di,4
            ASM loop tcl7
#ifdef MR_LMM
            ASM pop es
            ASM pop ds
#endif
#endif
#if INLINE_ASM == 3
            ASM mov ecx,xl
            ASM shl ecx,1
            ASM mov edi,w0g
          tcl5:
            ASM rcl DWORD PTR [edi],1
            ASM inc edi
            ASM inc edi
            ASM inc edi
            ASM inc edi
            ASM loop tcl5

            ASM mov ecx,xl
            ASM mov esi,xg
            ASM mov edi,w0g
            ASM xor ebx,ebx
          tcl7:
            ASM mov eax,[esi]
            ASM add_cuda esi,4
            ASM mul eax
            ASM add_cuda eax,ebx
            ASM adc edx,0
            ASM add_cuda [edi],eax
            ASM adc edx,0
            ASM xor ebx,ebx
            ASM add_cuda edi,4
            ASM add_cuda [edi],edx
            ASM adc ebx,0
            ASM add_cuda edi,4
            ASM dec ecx
            ASM jnz tcl7
#endif
#if INLINE_ASM == 4
   ASM (
           "movl %0,%%ecx\n"
           "shll $1,%%ecx\n"
           "movl %1,%%edi\n"
         "tcl5:\n"
           "rcll $1,(%%edi)\n"
           "incl %%edi\n"
           "incl %%edi\n"
           "incl %%edi\n"
           "incl %%edi\n"
           "loop tcl5\n"

           "movl %0,%%ecx\n"
           "movl %2,%%esi\n"
           "movl %1,%%edi\n"
           "xorl %%ebx,%%ebx\n"
         "tcl7:\n"
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n"
           "mull %%eax\n"
           "addl %%ebx,%%eax\n"
           "adcl $0,%%edx\n"
           "addl %%eax,(%%edi)\n"
           "adcl $0,%%edx\n"
           "xorl %%ebx,%%ebx\n"
           "addl $4,%%edi\n"
           "addl %%edx,(%%edi)\n"
           "adcl $0,%%ebx\n"
           "addl $4,%%edi\n"
           "decl %%ecx\n"
           "jnz tcl7\n"                       
        :
        :"m"(xl),"m"(w0g),"m"(xg)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif
#endif
#ifndef INLINE_ASM
            w0->len=xl+xl-1;
            mr_padd_cuda(_MIPP_ w0,w0,w0);     /* double it */
            carry=0;
            for (i=0;i<xl;i++)
            { /* add_cuda in squared elements */
                ti=i+i;
#ifdef MR_NOASM               
                dble.d=(mr_large)x->w[i]*x->w[i]+carry+w0->w[ti];
                w0->w[ti]=dble.h[MR_BOT];
                carry=dble.h[MR_TOP];
#else
                muldvd2_cuda(x->w[i],x->w[i],&carry,&w0->w[ti]);
#endif
                w0->w[ti+1]+=carry;
                if (w0->w[ti+1]<carry) carry=1;
                else                   carry=0;
            }
#endif
        }
        else for (i=0;i<xl;i++)
        { /* long multiplication */
       /* inline - substitutes for loop below */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
            ASM cld
            ASM mov cx,yl
            ASM mov dx,i
            ASM shl dx,1
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g
            ASM add_cuda bx,dx
            ASM lds si,DWORD PTR xg
            ASM add_cuda si,dx
            ASM mov di,[si]
            ASM lds si,DWORD PTR yg
#else
            ASM mov bx,w0g
            ASM add_cuda bx,dx
            ASM mov si,xg
            ASM add_cuda si,dx
            ASM mov di,[si]
            ASM mov si,yg
#endif
            ASM push bp
            ASM xor bp,bp
          tcl6:
            ASM lodsw
            ASM mul di
            ASM add_cuda ax,bp
            ASM adc dx,0
#ifdef MR_LMM
            ASM add_cuda es:[bx],ax
#else
            ASM add_cuda [bx],ax
#endif
            ASM inc bx
            ASM inc bx
            ASM adc dx,0
            ASM mov bp,dx
            ASM loop tcl6

#ifdef MR_LMM
            ASM mov es:[bx],bp
            ASM pop bp
            ASM pop es
            ASM pop ds
#else
            ASM mov [bx],bp
            ASM pop bp
#endif
#endif
#if INLINE_ASM == 2
            ASM cld
            ASM mov cx,yl
            ASM mov dx,i
            ASM shl dx,2
#ifdef MR_LMM
            ASM push ds
            ASM push es
            ASM les bx,DWORD PTR w0g
            ASM add_cuda bx,dx
            ASM lds si,DWORD PTR xg
            ASM add_cuda si,dx
            ASM mov edi,[si]
            ASM lds si,DWORD PTR yg
#else           
            ASM mov bx,w0g
            ASM add_cuda bx,dx
            ASM mov si,xg
            ASM add_cuda si,dx
            ASM mov edi,[si]
            ASM mov si,yg
#endif
            ASM push ebp
            ASM xor ebp,ebp
          tcl6:
            ASM lodsd
            ASM mul edi
            ASM add_cuda eax,ebp
            ASM adc edx,0
#ifdef MR_LMM
            ASM add_cuda es:[bx],eax
#else
            ASM add_cuda [bx],eax
#endif
            ASM adc edx,0
            ASM add_cuda bx,4
            ASM mov ebp,edx
            ASM loop tcl6

#ifdef MR_LMM
            ASM mov es:[bx],ebp
            ASM pop ebp
            ASM pop es
            ASM pop ds
#else
            ASM mov [bx],ebp
            ASM pop ebp
#endif
#endif
#if INLINE_ASM == 3
            ASM mov ecx,yl
            ASM mov esi,i
            ASM shl esi,2
            ASM mov ebx,xg
            ASM add_cuda ebx,esi
            ASM mov edi,[ebx]
            ASM mov ebx,w0g
            ASM add_cuda ebx,esi
            ASM mov esi,yg
            ASM sub ebx,esi
            ASM sub ebx,4
            ASM push ebp
            ASM xor ebp,ebp
          tcl6:
            ASM mov eax,[esi]
            ASM add_cuda esi,4
            ASM mul edi
            ASM add_cuda eax,ebp
            ASM mov ebp,[esi+ebx]
            ASM adc edx,0
            ASM add_cuda ebp,eax
            ASM adc edx,0
            ASM mov [esi+ebx],ebp
            ASM dec ecx
            ASM mov ebp,edx
            ASM jnz tcl6

            ASM mov [esi+ebx+4],ebp
            ASM pop ebp
#endif
#if INLINE_ASM == 4
   ASM (
           "movl %0,%%ecx\n"
           "movl %1,%%esi\n"
           "shll $2,%%esi\n"
           "movl %2,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "movl (%%ebx),%%edi\n"
           "movl %3,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "movl %4,%%esi\n"
           "subl %%esi,%%ebx\n"
           "subl $4,%%ebx\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"
         "tcl6:\n"
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n" 
           "mull %%edi\n"
           "addl %%ebp,%%eax\n"
           "movl (%%esi,%%ebx),%%ebp\n"
           "adcl $0,%%edx\n"
           "addl %%eax,%%ebp\n" 
           "adcl $0,%%edx\n"
           "movl %%ebp,(%%esi,%%ebx)\n"
           "decl %%ecx\n"
           "movl %%edx,%%ebp\n"
           "jnz tcl6\n"   

           "movl %%ebp,4(%%esi,%%ebx)\n"
           "popl %%ebp\n"
        :
        :"m"(yl),"m"(i),"m"(xg),"m"(w0g),"m"(yg)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif
#endif
#ifndef INLINE_ASM
            carry=0;
            for (j=0;j<yl;j++)
            { /* multiply_cuda each digit of y by x[i] */
#ifdef MR_NOASM 
                dble.d=(mr_large)x->w[i]*y->w[j]+carry+w0->w[i+j];
                w0->w[i+j]=dble.h[MR_BOT];
                carry=dble.h[MR_TOP];
#else
                muldvd2_cuda(x->w[i],y->w[j],&carry,&w0->w[i+j]);
#endif
            }
            w0->w[yl+i]=carry;
#endif
        }
#endif
#ifndef MR_SIMPLE_BASE
    }
    else
    {
        if (x==y && xl>SQR_FASTER_THRESHOLD)
        { /* squaring can be done nearly twice as fast */
            for (i=0;i<xl-1;i++)
            { /* long multiplication */
                carry=0;
                for (j=i+1;j<xl;j++)
                { /* Only do above the diagonal */
#ifdef MR_NOASM
                   dbled=(mr_large)x->w[i]*x->w[j]+w0->w[i+j]+carry;
  #ifdef MR_FP_ROUNDING
                   carry=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base); 
  #else
    #ifndef MR_FP
                   if (mr_mip->base==mr_mip->base2)
                       carry=(mr_small)(dbled>>mr_mip->lg2b);
                   else
    #endif
                       carry=(mr_small)MR_LROUND(dbled/mr_mip->base);  
  #endif
                   w0->w[i+j]=(mr_small)(dbled-(mr_large)carry*mr_mip->base);
#else

  #ifdef MR_FP_ROUNDING
              carry=imuldiv(x->w[i],x->w[j],w0->w[i+j]+carry,mr_mip->base,mr_mip->inverse_base,&w0->w[i+j]);
  #else
              carry=muldiv_cuda(x->w[i],x->w[j],w0->w[i+j]+carry,mr_mip->base,&w0->w[i+j]); 
  #endif
#endif
                }
                w0->w[xl+i]=carry;
            }
            w0->len=xl+xl-1;
            mr_padd_cuda(_MIPP_ w0,w0,w0);     /* double it */
            carry=0;
            for (i=0;i<xl;i++)
            { /* add_cuda in squared elements */
                ti=i+i;
#ifdef MR_NOASM
                dbled=(mr_large)x->w[i]*x->w[i]+w0->w[ti]+carry;
#ifdef MR_FP_ROUNDING
                carry=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
#else
#ifndef MR_FP
                if (mr_mip->base==mr_mip->base2)
                    carry=(mr_small)(dbled>>mr_mip->lg2b);
                else
#endif
                    carry=(mr_small)MR_LROUND(dbled/mr_mip->base); 
#endif
                w0->w[ti]=(mr_small)(dbled-(mr_large)carry*mr_mip->base);
#else

#ifdef MR_FP_ROUNDING
                carry=imuldiv(x->w[i],x->w[i],w0->w[ti]+carry,mr_mip->base,mr_mip->inverse_base,&w0->w[ti]);
#else
                carry=muldiv_cuda(x->w[i],x->w[i],w0->w[ti]+carry,mr_mip->base,&w0->w[ti]);
#endif

#endif
                w0->w[ti+1]+=carry;
                carry=0;
                if (w0->w[ti+1]>=mr_mip->base)
                {
                    carry=1;
                    w0->w[ti+1]-=mr_mip->base;
                }
            }
        }
        else for (i=0;i<xl;i++)
        { /* long multiplication */
            carry=0; 
            for (j=0;j<yl;j++)
            { /* multiply_cuda each digit of y by x[i] */
#ifdef MR_NOASM
                dbled=(mr_large)x->w[i]*y->w[j]+w0->w[i+j]+carry;

#ifdef MR_FP_ROUNDING
                carry=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
#else
#ifndef MR_FP
                if (mr_mip->base==mr_mip->base2)
                    carry=(mr_small)(dbled>>mr_mip->lg2b);
                else 
#endif  
                    carry=(mr_small)MR_LROUND(dbled/mr_mip->base);
#endif
                w0->w[i+j]=(mr_small)(dbled-(mr_large)carry*mr_mip->base);
#else

#ifdef MR_FP_ROUNDING
                carry=imuldiv(x->w[i],y->w[j],w0->w[i+j]+carry,mr_mip->base,mr_mip->inverse_base,&w0->w[i+j]);
#else
                carry=muldiv_cuda(x->w[i],y->w[j],w0->w[i+j]+carry,mr_mip->base,&w0->w[i+j]);
#endif

#endif
            }
            w0->w[yl+i]=carry;
        }
    }
#endif
    w0->len=(sz|(xl+yl)); /* set length and sign of result */

    mr_lzero_cuda(w0);
    copy_cuda(w0,z);
    MR_OUT
}

__device__ void divide_cuda(_MIPD_ big x,big y,big z)
{  /*  divide_cuda two big numbers  z=x/y : x=x mod y  *
    *  returns quotient only if  divide_cuda(x,y,x)    *
    *  returns remainder only if divide_cuda(x,y,y)    */
    mr_small carry,attemp,ldy,sdy,ra,r,d,tst,psum;
#ifdef MR_FP
    mr_small dres;
#endif
    mr_lentype sx,sy,sz;
    mr_small borrow,dig,*w0g,*yg;
    int i,k,m,x0,y0,w00;
    big w0;

#ifdef MR_ITANIUM
    mr_small tm;
#endif
#ifdef MR_WIN64
    mr_small tm;
#endif
#ifdef MR_NOASM
    union doubleword dble;
    mr_large dbled;
    mr_large ldres;
#endif
    BOOL check;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    w0=mr_mip->w0;

    MR_IN(6)

    if (x==y) mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
#ifdef MR_FLASH
    if (mr_notint_cuda(x) || mr_notint_cuda(y)) mr_berror_cuda(_MIPP_ MR_ERR_INT_OP);
#endif
    if (y->len==0) mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
    if (mr_mip->ERNUM)
    {
        MR_OUT
        return;
    }
    sx=(x->len&MR_MSBIT);   /*  extract signs ... */
    sy=(y->len&MR_MSBIT);
    sz=(sx^sy);
    x->len&=MR_OBITS;   /* ... and force operands to positive  */
    y->len&=MR_OBITS;
    x0=(int)x->len;
    y0=(int)y->len;
    copy_cuda(x,w0);
    w00=(int)w0->len;
    if (mr_mip->check && (w00-y0+1>mr_mip->nib))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_OVERFLOW);
        MR_OUT
        return;
    }
    d=0;
    if (x0==y0)
    {
        if (x0==1) /* special case - x and y are both mr_smalls */
        { 
            d=MR_DIV(w0->w[0],y->w[0]);
            w0->w[0]=MR_REMAIN(w0->w[0],y->w[0]);
            mr_lzero_cuda(w0);
        }
        else if (MR_DIV(w0->w[x0-1],4)<y->w[x0-1])
        while (mr_compare_cuda(w0,y)>=0)
        {  /* mr_small quotient - so do up to four subtracts instead */
            mr_psub_cuda(_MIPP_ w0,y,w0);
            d++;
        }
    }
    if (mr_compare_cuda(w0,y)<0)
    {  /*  x less than y - so x becomes remainder */
        if (x!=z)  /* testing parameters */
        {
            copy_cuda(w0,x);
            if (x->len!=0) x->len|=sx;
        }
        if (y!=z)
        {
            zero_cuda(z);
            z->w[0]=d;
            if (d>0) z->len=(sz|1);
        }
        y->len|=sy;
        MR_OUT
        return;
    }

    if (y0==1)
    {  /* y is int - so use subdiv_cuda instead */
#ifdef MR_FP_ROUNDING
        r=mr_sdiv_cuda(_MIPP_ w0,y->w[0],mr_invert(y->w[0]),w0);
#else
        r=mr_sdiv_cuda(_MIPP_ w0,y->w[0],w0);
#endif
        if (y!=z)
        {
            copy_cuda(w0,z);
            z->len|=sz;
        }
        if (x!=z)
        {
            zero_cuda(x);
            x->w[0]=r;
            if (r>0) x->len=(sx|1);
        }
        y->len|=sy;
        MR_OUT
        return;
    }
    if (y!=z) zero_cuda(z);
    d=normalise_cuda(_MIPP_ y,y);
    check=mr_mip->check;
    mr_mip->check=OFF;
#ifndef MR_SIMPLE_BASE
    if (mr_mip->base==0)
    {
#endif
#ifndef MR_NOFULLWIDTH
        if (d!=1) mr_pmul_cuda(_MIPP_ w0,d,w0);
        ldy=y->w[y0-1];
        sdy=y->w[y0-2];
        w0g=w0->w; yg=y->w;
        for (k=w00-1;k>=y0-1;k--)
        {  /* long division */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
#ifdef MR_LMM
            ASM push ds
            ASM lds bx,DWORD PTR w0g
#else
            ASM mov bx,w0g
#endif
            ASM mov si,k
            ASM shl si,1
            ASM add_cuda bx,si
            ASM mov dx,[bx+2]
            ASM mov ax,[bx]
            ASM cmp dx,ldy
            ASM jne tcl8
            ASM mov di,0xffff
            ASM mov si,ax
            ASM add_cuda si,ldy
            ASM jc tcl12
            ASM jmp tcl10
          tcl8:
            ASM div WORD PTR ldy
            ASM mov di,ax
            ASM mov si,dx
          tcl10:
            ASM mov ax,sdy
            ASM mul di
            ASM cmp dx,si
            ASM jb tcl12
            ASM jne tcl11
            ASM cmp ax,[bx-2]
            ASM jbe tcl12
          tcl11:
            ASM dec di
            ASM add_cuda si,ldy
            ASM jnc tcl10
          tcl12:
            ASM mov attemp,di
#ifdef MR_LMM
            ASM pop ds
#endif
#endif
/* NOTE push and pop of esi/edi should not be necessary - Borland C bug *
 * These pushes are needed here even if register variables are disabled */
#if INLINE_ASM == 2
            ASM push esi
            ASM push edi
#ifdef MR_LMM
            ASM push ds
            ASM lds bx,DWORD PTR w0g
#else
            ASM mov bx,w0g
#endif
            ASM mov si,k
            ASM shl si,2
            ASM add_cuda bx,si
            ASM mov edx,[bx+4]
            ASM mov eax,[bx]
            ASM cmp edx,ldy
            ASM jne tcl8
            ASM mov edi,0xffffffff
            ASM mov esi,eax
            ASM add_cuda esi,ldy
            ASM jc tcl12
            ASM jmp tcl10
          tcl8:
            ASM div DWORD PTR ldy
            ASM mov edi,eax
            ASM mov esi,edx
          tcl10:
            ASM mov eax,sdy
            ASM mul edi
            ASM cmp edx,esi
            ASM jb tcl12
            ASM jne tcl11
            ASM cmp eax,[bx-4]
            ASM jbe tcl12
          tcl11:
            ASM dec edi
            ASM add_cuda esi,ldy
            ASM jnc tcl10
          tcl12:
            ASM mov attemp,edi
#ifdef MR_LMM
            ASM pop ds
#endif
            ASM pop edi
            ASM pop esi
#endif  
#if INLINE_ASM == 3
            ASM push esi
            ASM push edi
            ASM mov ebx,w0g
            ASM mov esi,k
            ASM shl esi,2
            ASM add_cuda ebx,esi
            ASM mov edx,[ebx+4]
            ASM mov eax,[ebx]
            ASM cmp edx,ldy
            ASM jne tcl8
            ASM mov edi,0xffffffff
            ASM mov esi,eax
            ASM add_cuda esi,ldy
            ASM jc tcl12
            ASM jmp tcl10
          tcl8:
            ASM div DWORD PTR ldy
            ASM mov edi,eax
            ASM mov esi,edx
          tcl10:
            ASM mov eax,sdy
            ASM mul edi
            ASM cmp edx,esi
            ASM jb tcl12
            ASM jne tcl11
            ASM cmp eax,[ebx-4]
            ASM jbe tcl12
          tcl11:
            ASM dec edi
            ASM add_cuda esi,ldy
            ASM jnc tcl10
          tcl12:
            ASM mov attemp,edi
            ASM pop edi
            ASM pop esi
#endif       
#if INLINE_ASM == 4
   ASM (
           "movl %1,%%ebx\n"
           "movl %2,%%esi\n"
           "shll $2,%%esi\n"
           "addl %%esi,%%ebx\n"
           "movl 4(%%ebx),%%edx\n"
           "movl (%%ebx),%%eax\n"
           "cmpl %3,%%edx\n"
           "jne tcl8\n"
           "movl $0xffffffff,%%edi\n"
           "movl %%eax,%%esi\n"
           "addl %3,%%esi\n"
           "jc tcl12\n"
           "jmp tcl10\n"
         "tcl8:\n"
           "divl %3\n"
           "movl %%eax,%%edi\n"
           "movl %%edx,%%esi\n"
         "tcl10:\n"
           "movl %4,%%eax\n"
           "mull %%edi\n"
           "cmpl %%esi,%%edx\n"
           "jb tcl12\n"
           "jne tcl11\n"
           "cmpl -4(%%ebx),%%eax\n"
           "jbe tcl12\n"
         "tcl11:\n"
           "decl %%edi\n"
           "addl %3,%%esi\n"
           "jnc tcl10\n"
         "tcl12:\n"
           "movl %%edi,%0\n"
        :"=m"(attemp)
        :"m"(w0g),"m"(k),"m"(ldy),"m"(sdy)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif
#endif
#ifndef INLINE_ASM
            carry=0;
            if (w0->w[k+1]==ldy) /* guess next quotient digit */
            {
                attemp=(mr_small)(-1);
                ra=ldy+w0->w[k];
                if (ra<ldy) carry=1;
            }
#ifdef MR_NOASM
            else
            {
                dble.h[MR_BOT]=w0->w[k];
                dble.h[MR_TOP]=w0->w[k+1];
                attemp=(mr_small)(dble.d/ldy);
                ra=(mr_small)(dble.d-(mr_large)attemp*ldy);
            }
#else
            else attemp=muldvm_cuda(w0->w[k+1],w0->w[k],ldy,&ra);
#endif
            while (carry==0)
            {
#ifdef MR_NOASM
                dble.d=(mr_large)attemp*sdy;
                r=dble.h[MR_BOT];
                tst=dble.h[MR_TOP];
#else
                tst=muldvd_cuda(sdy,attemp,(mr_small)0,&r);
#endif
                if (tst< ra || (tst==ra && r<=w0->w[k-1])) break;
                attemp--;  /* refine guess */
                ra+=ldy;
                if (ra<ldy) carry=1;
            }
#endif    
            m=k-y0+1;
            if (attemp>0)
            { /* do partial subtraction */
                borrow=0;
    /*  inline - substitutes for loop below */
#ifdef INLINE_ASM
#if INLINE_ASM == 1
                ASM cld
                ASM mov cx,y0
                ASM mov si,m
                ASM shl si,1
                ASM mov di,attemp
#ifdef MR_LMM
                ASM push ds
                ASM push es
                ASM les bx,DWORD PTR w0g
                ASM add_cuda bx,si
                ASM sub bx,2
                ASM lds si,DWORD PTR yg
#else
                ASM mov bx,w0g
                ASM add_cuda bx,si
                ASM sub bx,2
                ASM mov si,yg
#endif
                ASM push bp
                ASM xor bp,bp

             tcl3:
                ASM lodsw
                ASM mul di
                ASM add_cuda ax,bp
                ASM adc dx,0
                ASM inc bx
                ASM inc bx
#ifdef MR_LMM
                ASM sub es:[bx],ax
#else
                ASM sub [bx],ax
#endif              
                ASM adc dx,0
                ASM mov bp,dx
                ASM loop tcl3

                ASM mov ax,bp
                ASM pop bp
#ifdef MR_LMM
                ASM pop es
                ASM pop ds
#endif
                ASM mov borrow,ax
#endif
/* NOTE push and pop of esi/edi should not be necessary - Borland C bug *
 * These pushes are needed here even if register variables are disabled */
#if INLINE_ASM == 2
                ASM push esi
                ASM push edi
                ASM cld
                ASM mov cx,y0
                ASM mov si,m
                ASM shl si,2
                ASM mov edi,attemp
#ifdef MR_LMM
                ASM push ds
                ASM push es
                ASM les bx,DWORD PTR w0g
                ASM add_cuda bx,si
                ASM sub bx,4
                ASM lds si,DWORD PTR yg
#else
                ASM mov bx,w0g
                ASM add_cuda bx,si
                ASM sub bx,4
                ASM mov si,yg
#endif
                ASM push ebp
                ASM xor ebp,ebp

             tcl3:
                ASM lodsd
                ASM mul edi
                ASM add_cuda eax,ebp
                ASM adc edx,0
                ASM add_cuda bx,4
#ifdef MR_LMM
                ASM sub es:[bx],eax
#else
                ASM sub [bx],eax
#endif
                ASM adc edx,0
                ASM mov ebp,edx
                ASM loop tcl3

                ASM mov eax,ebp
                ASM pop ebp
#ifdef MR_LMM
                ASM pop es
                ASM pop ds
#endif
                ASM mov borrow,eax
                ASM pop edi
                ASM pop esi
#endif
#if INLINE_ASM == 3
                ASM push esi
                ASM push edi
                ASM mov ecx,y0
                ASM mov esi,m
                ASM shl esi,2
                ASM mov edi,attemp
                ASM mov ebx,w0g
                ASM add_cuda ebx,esi
                ASM mov esi,yg
                ASM sub ebx,esi
                ASM sub ebx,4
                ASM push ebp
                ASM xor ebp,ebp

             tcl3:
                ASM mov eax,[esi]
                ASM add_cuda esi,4
                ASM mul edi
                ASM add_cuda eax,ebp
                ASM mov ebp,[esi+ebx]
                ASM adc edx,0
                ASM sub ebp,eax
                ASM adc edx,0
                ASM mov [esi+ebx],ebp
                ASM dec ecx
                ASM mov ebp,edx
                ASM jnz tcl3

                ASM mov eax,ebp
                ASM pop ebp
                ASM mov borrow,eax
                ASM pop edi
                ASM pop esi
#endif
#if INLINE_ASM == 4
   ASM (
           "movl %1,%%ecx\n"
           "movl %2,%%esi\n"
           "shll $2,%%esi\n"
           "movl %3,%%edi\n"
           "movl %4,%%ebx\n"
           "addl %%esi,%%ebx\n"
           "movl %5,%%esi\n"
           "subl %%esi,%%ebx\n"
           "subl $4,%%ebx\n"
           "pushl %%ebp\n"
           "xorl %%ebp,%%ebp\n"
         "tcl3:\n"
           "movl (%%esi),%%eax\n"
           "addl $4,%%esi\n"
           "mull %%edi\n"
           "addl %%ebp,%%eax\n"
           "movl (%%esi,%%ebx),%%ebp\n"
           "adcl $0,%%edx\n"
           "subl %%eax,%%ebp\n"
           "adcl $0,%%edx\n"
           "movl %%ebp,(%%esi,%%ebx)\n"
           "decl %%ecx\n"
           "movl %%edx,%%ebp\n"
           "jnz tcl3\n"
    
           "movl %%ebp,%%eax\n"
           "popl %%ebp\n"
           "movl %%eax,%0\n"
 
        :"=m"(borrow)
        :"m"(y0),"m"(m),"m"(attemp),"m"(w0g),"m"(yg)
        :"eax","edi","esi","ebx","ecx","edx","memory"
       );
#endif
#endif
#ifndef INLINE_ASM
                for (i=0;i<y0;i++)
                {
#ifdef MR_NOASM
                    dble.d=(mr_large)attemp*y->w[i]+borrow;
                    dig=dble.h[MR_BOT];
                    borrow=dble.h[MR_TOP];
#else
                  borrow=muldvd_cuda(attemp,y->w[i],borrow,&dig);
#endif
                  if (w0->w[m+i]<dig) borrow++;
                  w0->w[m+i]-=dig;
                }
#endif

                if (w0->w[k+1]<borrow)
                {  /* whoops! - over did it */
                    w0->w[k+1]=0;
                    carry=0;
                    for (i=0;i<y0;i++)
                    {  /* compensate for error ... */
                        psum=w0->w[m+i]+y->w[i]+carry;
                        if (psum>y->w[i]) carry=0;
                        if (psum<y->w[i]) carry=1;
                        w0->w[m+i]=psum;
                    }
                    attemp--;  /* ... and adjust guess */
                }
                else w0->w[k+1]-=borrow;
            }
            if (k==w00-1 && attemp==0) w00--;
            else if (y!=z) z->w[m]=attemp;
        }
#endif
#ifndef MR_SIMPLE_BASE
    }
    else
    {   /* have to do it the hard way */
        if (d!=1) mr_pmul_cuda(_MIPP_ w0,d,w0);
        ldy=y->w[y0-1];
        sdy=y->w[y0-2];

        for (k=w00-1;k>=y0-1;k--)
        {  /* long division */


            if (w0->w[k+1]==ldy) /* guess next quotient digit */
            {
                attemp=mr_mip->base-1;
                ra=ldy+w0->w[k];
            }
#ifdef MR_NOASM
            else 
            {
                dbled=(mr_large)w0->w[k+1]*mr_mip->base+w0->w[k];
                attemp=(mr_small)MR_LROUND(dbled/ldy);
                ra=(mr_small)(dbled-(mr_large)attemp*ldy);
            }
#else
            else attemp=muldiv_cuda(w0->w[k+1],mr_mip->base,w0->w[k],ldy,&ra);
#endif
            while (ra<mr_mip->base)
            {
#ifdef MR_NOASM
                dbled=(mr_large)sdy*attemp;
#ifdef MR_FP_ROUNDING
                tst=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
#else
#ifndef MR_FP
                if (mr_mip->base==mr_mip->base2)
                    tst=(mr_small)(dbled>>mr_mip->lg2b);
                else 
#endif  
                    tst=(mr_small)MR_LROUND(dbled/mr_mip->base);
#endif
                r=(mr_small)(dbled-(mr_large)tst*mr_mip->base);
#else
#ifdef MR_FP_ROUNDING
                tst=imuldiv(sdy,attemp,(mr_small)0,mr_mip->base,mr_mip->inverse_base,&r); 
#else
                tst=muldiv_cuda(sdy,attemp,(mr_small)0,mr_mip->base,&r); 
#endif
#endif
                if (tst< ra || (tst==ra && r<=w0->w[k-1])) break;
                attemp--;  /* refine guess */
                ra+=ldy;
            }    
            m=k-y0+1;
            if (attemp>0)
            { /* do partial subtraction */
                borrow=0;
                for (i=0;i<y0;i++)
                {
#ifdef MR_NOASM
                  dbled=(mr_large)attemp*y->w[i]+borrow;
#ifdef MR_FP_ROUNDING
                  borrow=(mr_small)MR_LROUND(dbled*mr_mip->inverse_base);
#else
#ifndef MR_FP
                  if (mr_mip->base==mr_mip->base2)
                      borrow=(mr_small)(dbled>>mr_mip->lg2b);
                  else 
#endif  
                      borrow=(mr_small)MR_LROUND(dbled/mr_mip->base);
#endif
                  dig=(mr_small)(dbled-(mr_large)borrow*mr_mip->base);
#else
#ifdef MR_FP_ROUNDING
                  borrow=imuldiv(attemp,y->w[i],borrow,mr_mip->base,mr_mip->inverse_base,&dig);
#else
                  borrow=muldiv_cuda(attemp,y->w[i],borrow,mr_mip->base,&dig);
#endif
#endif
                  if (w0->w[m+i]<dig)
                  { /* set borrow */
                      borrow++;
                      w0->w[m+i]+=(mr_mip->base-dig);
                  }
                  else w0->w[m+i]-=dig;
                }
                if (w0->w[k+1]<borrow)
                {  /* whoops! - over did it */
                    w0->w[k+1]=0;
                    carry=0;
                    for (i=0;i<y0;i++)
                    {  /* compensate for error ... */
                        psum=w0->w[m+i]+y->w[i]+carry;
                        carry=0;
                        if (psum>=mr_mip->base)
                        {
                            carry=1;
                            psum-=mr_mip->base;
                        }
                        w0->w[m+i]=psum;
                    }
                    attemp--;  /* ... and adjust guess */
                }
                else
                    w0->w[k+1]-=borrow;
            }
            if (k==w00-1 && attemp==0) w00--;
            else if (y!=z) z->w[m]=attemp;
        }
    }
#endif
    if (y!=z) z->len=((w00-y0+1)|sz); /* set sign and length of result */

    w0->len=y0;

    mr_lzero_cuda(y);
    mr_lzero_cuda(z);

    if (x!=z)
    {
        mr_lzero_cuda(w0);
#ifdef MR_FP_ROUNDING
        if (d!=1) mr_sdiv_cuda(_MIPP_ w0,d,mr_invert(d),x);
#else
        if (d!=1) mr_sdiv_cuda(_MIPP_ w0,d,x);
#endif
        else copy_cuda(w0,x);
        if (x->len!=0) x->len|=sx;
    }
#ifdef MR_FP_ROUNDING
    if (d!=1) mr_sdiv_cuda(_MIPP_ y,d,mr_invert(d),y);
#else
    if (d!=1) mr_sdiv_cuda(_MIPP_ y,d,y);
#endif
    y->len|=sy;
    mr_mip->check=check;

    MR_OUT
}

__device__ BOOL divisible_cuda(_MIPD_ big x,big y)
{ /* returns y|x, that is TRUE if y divides x exactly */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(87)

    copy_cuda (x,mr_mip->w0);
    divide_cuda(_MIPP_ mr_mip->w0,y,y);

    MR_OUT
    if (size_cuda(mr_mip->w0)==0) return TRUE;
    else                    return FALSE;
}     

__device__ void mad_cuda(_MIPD_ big x,big y,big z,big w,big q,big r)
{ /* Multiply, Add and Divide; q=(x*y+z)/w remainder r   *
   * returns remainder only if w=q, quotient only if q=r *
   * add_cuda done only if x, y and z are distinct.           */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    BOOL check;
    if (mr_mip->ERNUM) return;

    MR_IN(24)
    if (w==r)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return;
    }
    check=mr_mip->check;
    mr_mip->check=OFF;           /* turn off some error checks */

    multiply_cuda(_MIPP_ x,y,mr_mip->w0);
    if (x!=z && y!=z) add_cuda(_MIPP_ mr_mip->w0,z,mr_mip->w0);

    divide_cuda(_MIPP_ mr_mip->w0,w,q);
    if (q!=r) copy_cuda(mr_mip->w0,r);
    mr_mip->check=check;
    MR_OUT
}

#endif

#ifndef mrmuldv_c
#define mrmuldv_c

__device__ mr_small muldiv_cuda(mr_small a, mr_small b, mr_small c, mr_small m, mr_small *rp)
{
    __int128_t t = (__int128_t)a * b + c;
    *rp = (mr_small)(t % m);
    return (mr_small)(t / m);
}
__device__ mr_small muldvm_cuda(mr_small a, mr_small c, mr_small m, mr_small *rp)
{
    __int128_t t = (__int128_t)a<<64 | c;
    *rp = (mr_small)(t % m);
    return (mr_small)(t / m);
}
__device__ mr_small muldvd_cuda(mr_small a, mr_small b, mr_small c, mr_small *rp)
{
    __int128_t t = (__int128_t)a * b + c;
    *rp = (mr_small)t;
    return (mr_small)(t >> 64);
}
__device__ void muldvd2_cuda(mr_small a, mr_small b, mr_small *c, mr_small *rp)
{
    __int128_t t = (__int128_t)a * b + *c + *rp;
    *rp = (mr_small)t;
    *c = (mr_small)(t >> 64);
}

#endif

#ifndef mrxgcd_c
#define mrxgcd_c

#ifdef MR_FP
#include <math.h>
#endif

#ifdef MR_COUNT_OPS
extern int fpx; 
#endif

#ifndef MR_USE_BINARY_XGCD

#ifdef mr_dltype

__device__ static mr_small qdiv_cuda(mr_large u,mr_large v)
{ /* fast division - small quotient expected.  */
    mr_large lq,x=u;
#ifdef MR_FP
    mr_small dres;
#endif
    x-=v;
    if (x<v) return 1;
    x-=v;
    if (x<v) return 2;
    x-=v;
    if (x<v) return 3;
    x-=v;
    if (x<v) return 4;
    x-=v;
    if (x<v) return 5;
    x-=v;
    if (x<v) return 6;
    x-=v;
    if (x<v) return 7;
    x-=v;
    if (x<v) return 8;

/* do it the hard way! */

    lq=8+MR_DIV(x,v);
    if (lq>=MAXBASE) return 0;
    return (mr_small)lq;
}

#else

__device__ static mr_small qdiv_cuda(mr_small u,mr_small v)
{ /* fast division - small quotient expected */
    mr_small x=u;
    x-=v;
    if (x<v) return 1;
    x-=v;
    if (x<v) return 2;

    return MR_DIV(u,v);
}

#endif

__device__ int xgcd_cuda(_MIPD_ big x,big y,big xd,big yd,big z)
{ /* greatest common divisor by Euclids method  *
   * extended to also calculate xd and yd where *
   *      z = x.xd + y.yd = gcd(x,y)            *
   * if xd, yd not distinct, only xd calculated *
   * z only returned if distinct from xd and yd *
   * xd will always be positive, yd negative    */

    int s,n,iter;
    mr_small r,a,b,c,d;
    mr_small q,m,sr;
#ifdef MR_FP
    mr_small dres;
#endif

#ifdef mr_dltype
    union doubleword uu,vv;
    mr_large u,v,lr;
#else
    mr_small u,v,lr;
#endif

    BOOL last,dplus=TRUE;
    big t;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return 0;

    MR_IN(30)

#ifdef MR_COUNT_OPS
    fpx++; 
#endif
  
    copy_cuda(x,mr_mip->w1);
    copy_cuda(y,mr_mip->w2);
    s=exsign_cuda(mr_mip->w1);
    insign_cuda(PLUS,mr_mip->w1);
    insign_cuda(PLUS,mr_mip->w2);
    convert_cuda(_MIPP_ 1,mr_mip->w3);
    zero_cuda(mr_mip->w4);
    last=FALSE;
    a=b=c=d=0;
    iter=0;

    while (size_cuda(mr_mip->w2)!=0)
    {
        if (b==0)
        { /* update mr_mip->w1 and mr_mip->w2 */

            divide_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w5);
            t=mr_mip->w1,mr_mip->w1=mr_mip->w2,mr_mip->w2=t;    /* swap(mr_mip->w1,mr_mip->w2) */
            multiply_cuda(_MIPP_ mr_mip->w4,mr_mip->w5,mr_mip->w0);
            add_cuda(_MIPP_ mr_mip->w3,mr_mip->w0,mr_mip->w3);
            t=mr_mip->w3,mr_mip->w3=mr_mip->w4,mr_mip->w4=t;    /* swap(xd,yd) */
            iter++;

        }
        else
        {

 /* printf("a= %I64u b= %I64u c= %I64u  d= %I64u \n",a,b,c,d);   */

            mr_pmul_cuda(_MIPP_ mr_mip->w1,c,mr_mip->w5);   /* c*w1 */
            mr_pmul_cuda(_MIPP_ mr_mip->w1,a,mr_mip->w1);   /* a*w1 */
            mr_pmul_cuda(_MIPP_ mr_mip->w2,b,mr_mip->w0);   /* b*w2 */
            mr_pmul_cuda(_MIPP_ mr_mip->w2,d,mr_mip->w2);   /* d*w2 */

            if (!dplus)
            {
                mr_psub_cuda(_MIPP_ mr_mip->w0,mr_mip->w1,mr_mip->w1); /* b*w2-a*w1 */
                mr_psub_cuda(_MIPP_ mr_mip->w5,mr_mip->w2,mr_mip->w2); /* c*w1-d*w2 */
            }
            else
            {
                mr_psub_cuda(_MIPP_ mr_mip->w1,mr_mip->w0,mr_mip->w1); /* a*w1-b*w2 */
                mr_psub_cuda(_MIPP_ mr_mip->w2,mr_mip->w5,mr_mip->w2); /* d*w2-c*w1 */
            }
            mr_pmul_cuda(_MIPP_ mr_mip->w3,c,mr_mip->w5);
            mr_pmul_cuda(_MIPP_ mr_mip->w3,a,mr_mip->w3);
            mr_pmul_cuda(_MIPP_ mr_mip->w4,b,mr_mip->w0);
            mr_pmul_cuda(_MIPP_ mr_mip->w4,d,mr_mip->w4);
    
            if (a==0) copy_cuda(mr_mip->w0,mr_mip->w3);
            else      mr_padd_cuda(_MIPP_ mr_mip->w3,mr_mip->w0,mr_mip->w3);
            mr_padd_cuda(_MIPP_ mr_mip->w4,mr_mip->w5,mr_mip->w4);
        }
        if (mr_mip->ERNUM || size_cuda(mr_mip->w2)==0) break;


        n=(int)mr_mip->w1->len;
        if (n==1)
        {
            last=TRUE;
            u=mr_mip->w1->w[0];
            v=mr_mip->w2->w[0];
        }
        else
        {
            m=mr_mip->w1->w[n-1]+1;
#ifndef MR_SIMPLE_BASE
            if (mr_mip->base==0)
            {
#endif
#ifndef MR_NOFULLWIDTH
#ifdef mr_dltype
 /* use double length type if available */
                if (n>2 && m!=0)
                { /* squeeze out as much significance as possible */
                    uu.h[MR_TOP]=muldvm_cuda(mr_mip->w1->w[n-1],mr_mip->w1->w[n-2],m,&sr);
                    uu.h[MR_BOT]=muldvm_cuda(sr,mr_mip->w1->w[n-3],m,&sr);
                    vv.h[MR_TOP]=muldvm_cuda(mr_mip->w2->w[n-1],mr_mip->w2->w[n-2],m,&sr);
                    vv.h[MR_BOT]=muldvm_cuda(sr,mr_mip->w2->w[n-3],m,&sr);
                }
                else
                {
                    uu.h[MR_TOP]=mr_mip->w1->w[n-1];
                    uu.h[MR_BOT]=mr_mip->w1->w[n-2];
                    vv.h[MR_TOP]=mr_mip->w2->w[n-1];
                    vv.h[MR_BOT]=mr_mip->w2->w[n-2];
                    if (n==2) last=TRUE;
                }

                u=uu.d;
                v=vv.d;
#else
                if (m==0)
                {
                    u=mr_mip->w1->w[n-1];
                    v=mr_mip->w2->w[n-1];   
                }
                else
                {
                    u=muldvm_cuda(mr_mip->w1->w[n-1],mr_mip->w1->w[n-2],m,&sr);
                    v=muldvm_cuda(mr_mip->w2->w[n-1],mr_mip->w2->w[n-2],m,&sr);
                }
#endif
#endif
#ifndef MR_SIMPLE_BASE
            }
            else
            {
#ifdef mr_dltype
                if (n>2)
                { /* squeeze out as much significance as possible */
                    u=muldiv_cuda(mr_mip->w1->w[n-1],mr_mip->base,mr_mip->w1->w[n-2],m,&sr);
                    u=u*mr_mip->base+muldiv_cuda(sr,mr_mip->base,mr_mip->w1->w[n-3],m,&sr);
                    v=muldiv_cuda(mr_mip->w2->w[n-1],mr_mip->base,mr_mip->w2->w[n-2],m,&sr);
                    v=v*mr_mip->base+muldiv_cuda(sr,mr_mip->base,mr_mip->w2->w[n-3],m,&sr);
                }
                else
                {
                    u=(mr_large)mr_mip->base*mr_mip->w1->w[n-1]+mr_mip->w1->w[n-2];
                    v=(mr_large)mr_mip->base*mr_mip->w2->w[n-1]+mr_mip->w2->w[n-2];
                    last=TRUE;
                }
#else
                u=muldiv_cuda(mr_mip->w1->w[n-1],mr_mip->base,mr_mip->w1->w[n-2],m,&sr);
                v=muldiv_cuda(mr_mip->w2->w[n-1],mr_mip->base,mr_mip->w2->w[n-2],m,&sr);
#endif
            }
#endif
        }

        dplus=TRUE;
        a=1; b=0; c=0; d=1;

        forever
        { /* work only with most significant piece */
            if (last)
            {
                if (v==0) break;
                q=qdiv_cuda(u,v);
                if (q==0) break;
            }
            else
            {
                if (dplus)
                { 
                    if ((mr_small)(v-c)==0 || (mr_small)(v+d)==0) break;

                    q=qdiv_cuda(u+a,v-c);

                    if (q==0) break;

                    if (q!=qdiv_cuda(u-b,v+d)) break;
                }
                else 
                {
                    if ((mr_small)(v+c)==0 || (mr_small)(v-d)==0) break;
                    q=qdiv_cuda(u-a,v+c);
                    if (q==0) break;
                    if (q!=qdiv_cuda(u+b,v-d)) break;
                }
            }

            if (q==1)
            {
                if ((mr_small)(b+d) >= MAXBASE) break; 
                r=a+c;  a=c; c=r;
                r=b+d;  b=d; d=r;
                lr=u-v; u=v; v=lr;      
            }
            else
            { 
                if (q>=MR_DIV(MAXBASE-b,d)) break;
                r=a+q*c;  a=c; c=r;
                r=b+q*d;  b=d; d=r;
                lr=u-q*v; u=v; v=lr;
            }
            iter++;
            dplus=!dplus;
        }
        iter%=2;

    }

    if (s==MINUS) iter++;
    if (iter%2==1) subtract_cuda(_MIPP_ y,mr_mip->w3,mr_mip->w3);

    if (xd!=yd)
    {
        negify_cuda(x,mr_mip->w2);
        mad_cuda(_MIPP_ mr_mip->w2,mr_mip->w3,mr_mip->w1,y,mr_mip->w4,mr_mip->w4);
        copy_cuda(mr_mip->w4,yd);
    }
    copy_cuda(mr_mip->w3,xd);
    if (z!=xd && z!=yd) copy_cuda(mr_mip->w1,z);

    MR_OUT
    return (size_cuda(mr_mip->w1));
}

__device__ int invmodp_cuda(_MIPD_ big x,big y,big z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    int gcd;

    MR_IN(213);
    gcd=xgcd_cuda(_MIPP_ x,y,z,z,z);
    MR_OUT
    return gcd;
}

#else

/* much  smaller, much slower binary inversion algorithm */
/* fails silently if a is not co-prime to p   */

/* experimental! At least 3 times slower than standard method.. */

__device__ int invmodp_cuda(_MIPD_ big a,big p,big z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    big u,v,x1,x2;

    MR_IN(213);

    u=mr_mip->w1; v=mr_mip->w2; x1=mr_mip->w3; x2=mr_mip->w4;
    copy_cuda(a,u);    
    copy_cuda(p,v);    
    convert_cuda(_MIPP_ 1,x1); 
    zero_cuda(x2);      
   
    while (size_cuda(u)!=1 && size_cuda(v)!=1)
    {
        while (remain_cuda(_MIPP_ u,2)==0)
        {
            subdiv_cuda(_MIPP_ u,2,u);
            if (remain_cuda(_MIPP_ x1,2)!=0) add_cuda(_MIPP_ x1,p,x1);
            subdiv_cuda(_MIPP_ x1,2,x1);
        }
        while (remain_cuda(_MIPP_ v,2)==0)
        {
            subdiv_cuda(_MIPP_ v,2,v);
            if (remain_cuda(_MIPP_ x2,2)!=0) add_cuda(_MIPP_ x2,p,x2);
            subdiv_cuda(_MIPP_ x2,2,x2);
        }
        if (mr_compare_cuda(u,v)>=0)
        {
            mr_psub_cuda(_MIPP_ u,v,u);
            subtract_cuda(_MIPP_ x1,x2,x1);
        }
        else
        {
            mr_psub_cuda(_MIPP_ v,u,v);
            subtract_cuda(_MIPP_ x2,x1,x2);
        }
    }
    if (size_cuda(u)==1) copy_cuda(x1,z);
    else            copy_cuda(x2,z);

    if (size_cuda(z)<0) add_cuda(_MIPP_ z,p,z);

    MR_OUT
    return 1; /* note - no checking that gcd=1 */
}

#endif

#ifndef MR_STATIC

/* Montgomery's method for multiple 
   simultaneous modular inversions */

__device__ BOOL double_inverse_cuda(_MIPD_ big n,big x,big y,big w,big z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(146)

    mad_cuda(_MIPP_ x,w,w,n,n,mr_mip->w6);
    if (size_cuda(mr_mip->w6)==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }
    invmodp_cuda(_MIPP_ mr_mip->w6,n,mr_mip->w6);

    mad_cuda(_MIPP_ w,mr_mip->w6,w,n,n,y);
    mad_cuda(_MIPP_ x,mr_mip->w6,x,n,n,z);

    MR_OUT 
    return TRUE;   
}

__device__ BOOL multi_inverse_cuda(_MIPD_ int m,big *x,big n,big *w)
{ /* find w[i]=1/x[i] mod n, for i=0 to m-1 *
   * x and w MUST be distinct               */
    int i;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (m==0) return TRUE;
    if (m<0) return FALSE;

    MR_IN(25)

    if (x==w)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return FALSE;
    }
    if (m==1)
    {
        invmodp_cuda(_MIPP_ x[0],n,w[0]);
        MR_OUT
        return TRUE;
    }

    convert_cuda(_MIPP_ 1,w[0]);
    copy_cuda(x[0],w[1]);
    for (i=2;i<m;i++)
        mad_cuda(_MIPP_ w[i-1],x[i-1],x[i-1],n,n,w[i]); 

    mad_cuda(_MIPP_ w[m-1],x[m-1],x[m-1],n,n,mr_mip->w6);  /* y=x[0]*x[1]*x[2]....x[m-1] */
    if (size_cuda(mr_mip->w6)==0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }

    invmodp_cuda(_MIPP_ mr_mip->w6,n,mr_mip->w6);

/* Now y=1/y */

    copy_cuda(x[m-1],mr_mip->w5);
    mad_cuda(_MIPP_ w[m-1],mr_mip->w6,mr_mip->w6,n,n,w[m-1]);

    for (i=m-2;;i--)
    {
        if (i==0)
        {
            mad_cuda(_MIPP_ mr_mip->w5,mr_mip->w6,mr_mip->w6,n,n,w[0]);
            break;
        }
        mad_cuda(_MIPP_ w[i],mr_mip->w5,w[i],n,n,w[i]);
        mad_cuda(_MIPP_ w[i],mr_mip->w6,w[i],n,n,w[i]);
        mad_cuda(_MIPP_ mr_mip->w5,x[i],x[i],n,n,mr_mip->w5);
    }

    MR_OUT 
    return TRUE;   
}

#endif


#endif

#ifndef mrbits_c
#define mrbits_c

#ifdef MR_FP
#include <math.h>
#endif

__device__ int logb2_cuda(_MIPD_ big x)
{ /* returns number of bits in x */
    int xl,lg2;
    mr_small top;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM || size_cuda(x)==0) return 0;

    MR_IN(49)


#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        xl=(int)(x->len&MR_OBITS);
        lg2=mr_mip->lg2b*(xl-1);
        top=x->w[xl-1];
        while (top>=1)
        {
            lg2++;
            top/=2;
        }

#ifndef MR_ALWAYS_BINARY
    }
    else 
    {
        copy_cuda(x,mr_mip->w0);
        insign_cuda(PLUS,mr_mip->w0);
        lg2=0;
        while (mr_mip->w0->len>1)
        {
#ifdef MR_FP_ROUNDING
            mr_sdiv_cuda(_MIPP_ mr_mip->w0,mr_mip->base2,mr_invert(mr_mip->base2),mr_mip->w0);
#else
            mr_sdiv_cuda(_MIPP_ mr_mip->w0,mr_mip->base2,mr_mip->w0);
#endif
            lg2+=mr_mip->lg2b;
        }

        while (mr_mip->w0->w[0]>=1)
        {
            lg2++;
            mr_mip->w0->w[0]/=2;
        }
    }
#endif
    MR_OUT
    return lg2;
}

__device__ void sftbit_cuda(_MIPD_ big x,int n,big z)
{ /* shift x by n bits */
    int m;
    mr_small sm;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    copy_cuda(x,z);
    if (n==0) return;

    MR_IN(47)

    m=mr_abs(n);
    sm=mr_shiftbits_cuda((mr_small)1,m%mr_mip->lg2b);
    if (n>0)
    { /* shift left */

#ifndef MR_ALWAYS_BINARY
        if (mr_mip->base==mr_mip->base2)
        {
#endif
            mr_shift_cuda(_MIPP_ z,n/mr_mip->lg2b,z);
            mr_pmul_cuda(_MIPP_ z,sm,z);
#ifndef MR_ALWAYS_BINARY
        }
        else
        {
            expb2_cuda(_MIPP_ m,mr_mip->w1);
            multiply_cuda(_MIPP_ z,mr_mip->w1,z);
        }
#endif
    }
    else
    { /* shift right */

#ifndef MR_ALWAYS_BINARY
        if (mr_mip->base==mr_mip->base2)
        {
#endif
            mr_shift_cuda(_MIPP_ z,n/mr_mip->lg2b,z);
#ifdef MR_FP_ROUNDING
            mr_sdiv_cuda(_MIPP_ z,sm,mr_invert(sm),z);
#else
            mr_sdiv_cuda(_MIPP_ z,sm,z);
#endif

#ifndef MR_ALWAYS_BINARY
        }
        else
        {
            expb2_cuda(_MIPP_ m,mr_mip->w1);
            divide_cuda(_MIPP_ z,mr_mip->w1,z);
        }
#endif
    }
    MR_OUT
}

__device__ void expb2_cuda(_MIPD_ int n,big x)
{ /* sets x=2^n */
    int r,p;
#ifndef MR_ALWAYS_BINARY
    int i;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    convert_cuda(_MIPP_ 1,x);
    if (n==0) return;

    MR_IN(149)

    if (n<0)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NEG_POWER);
        MR_OUT
        return;
    }
    r=n/mr_mip->lg2b;
    p=n%mr_mip->lg2b;

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        mr_shift_cuda(_MIPP_ x,r,x);
        x->w[x->len-1]=mr_shiftbits_cuda(x->w[x->len-1],p);
#ifndef MR_ALWAYS_BINARY
    }
    else
    {
        for (i=1;i<=r;i++)
            mr_pmul_cuda(_MIPP_ x,mr_mip->base2,x);
        mr_pmul_cuda(_MIPP_ x,mr_shiftbits_cuda((mr_small)1,p),x);
    }
#endif
    MR_OUT
}

#ifndef MR_NO_RAND

__device__ void bigbits_cuda(_MIPD_ int n,big x)
{ /* sets x as random < 2^n */
    mr_small r;
    mr_lentype wlen;
#ifdef MR_FP
    mr_small dres;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zero_cuda(x);
    if (mr_mip->ERNUM || n<=0) return;
    
    MR_IN(150)

    expb2_cuda(_MIPP_ n,mr_mip->w1);
    wlen=mr_mip->w1->len;
    do
    {
        r=brand_cuda(_MIPPO_ );
        if (mr_mip->base==0) x->w[x->len++]=r;
        else                 x->w[x->len++]=MR_REMAIN(r,mr_mip->base);
    } while (x->len<wlen);
#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif

    x->w[wlen-1]=MR_REMAIN(x->w[wlen-1],mr_mip->w1->w[wlen-1]);
    mr_lzero_cuda(x);

#ifndef MR_ALWAYS_BINARY
    }
    else
    {
        divide_cuda(_MIPP_ x,mr_mip->w1,mr_mip->w1);
    }
#endif

    MR_OUT
}

#endif


#endif

#ifndef mrcurve_c
#define mrcurve_c

#ifdef MR_STATIC
#include <string.h>
#endif

#ifndef MR_EDWARDS

__device__ static void epoint_getrhs_cuda(_MIPD_ big x,big y)
{ /* x and y must be different */

  /* find x^3+Ax+B */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    nres_modmult_cuda(_MIPP_ x,x,y);

    nres_modmult_cuda(_MIPP_ y,x,y);
    if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
        nres_modmult_cuda(_MIPP_ x,mr_mip->A,mr_mip->w1);
    else
        nres_premult_cuda(_MIPP_ x,mr_mip->Asize,mr_mip->w1);
    nres_modadd_cuda(_MIPP_ y,mr_mip->w1,y);
    if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)
        nres_modadd_cuda(_MIPP_ y,mr_mip->B,y);
    else
    {
        convert_cuda(_MIPP_ mr_mip->Bsize,mr_mip->w1);
        nres_cuda(_MIPP_ mr_mip->w1,mr_mip->w1);
        nres_modadd_cuda(_MIPP_ y,mr_mip->w1,y);
    }
}

#ifndef MR_NOSUPPORT_COMPRESSION

__device__ BOOL epoint_x_cuda(_MIPD_ big x)
{ /* test if x is associated with a point on the   *
   * currently active curve                        */
    int j;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(147)
    
    if (x==NULL) return FALSE;

    nres_cuda(_MIPP_ x,mr_mip->w2);
    epoint_getrhs_cuda(_MIPP_ mr_mip->w2,mr_mip->w3);

    if (size_cuda(mr_mip->w3)==0)
    {
        MR_OUT
        return TRUE;
    }

    redc_cuda(_MIPP_ mr_mip->w3,mr_mip->w4);
    j=jack_cuda(_MIPP_ mr_mip->w4,mr_mip->modulus);

    MR_OUT
    if (j==1) return TRUE;
    return FALSE;
}

#endif

__device__ BOOL epoint_set_cuda(_MIPD_ big x,big y,int cb,epoint *p)
{ /* initialise a point on active ecurve            *
   * if x or y == NULL, set to point at infinity    *
   * if x==y, a y co-ordinate is calculated - if    *
   * possible - and cb suggests LSB 0/1  of y       *
   * (which "decompresses" y). Otherwise, check     *
   * validity of given (x,y) point, ignoring cb.    *
   * Returns TRUE for valid point, otherwise FALSE. */
  
    BOOL valid;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(97)

    if (x==NULL || y==NULL)
    {
        copy_cuda(mr_mip->one,p->X);
        copy_cuda(mr_mip->one,p->Y);
        p->marker=MR_EPOINT_INFINITY;
        MR_OUT
        return TRUE;
    }

/* find x^3+Ax+B */

    nres_cuda(_MIPP_ x,p->X);

    epoint_getrhs_cuda(_MIPP_ p->X,mr_mip->w3);

    valid=FALSE;

    if (x!=y)
    { /* compare with y^2 */
        nres_cuda(_MIPP_ y,p->Y);
        nres_modmult_cuda(_MIPP_ p->Y,p->Y,mr_mip->w1);
        
        if (mr_compare_cuda(mr_mip->w1,mr_mip->w3)==0) valid=TRUE;
    }
    else
    { /* no y supplied - calculate one. Find square root */
#ifndef MR_NOSUPPORT_COMPRESSION

        valid=nres_sqroot_cuda(_MIPP_ mr_mip->w3,p->Y);
    /* check LSB - have we got the right root? */
        redc_cuda(_MIPP_ p->Y,mr_mip->w1);
        if (remain_cuda(_MIPP_ mr_mip->w1,2)!=cb) 
            mr_psub_cuda(_MIPP_ mr_mip->modulus,p->Y,p->Y);

#else
    mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
    MR_OUT
    return FALSE;
#endif
    } 
    if (valid)
    {
        p->marker=MR_EPOINT_NORMALIZED;
        MR_OUT
        return TRUE;
    }

    MR_OUT
    return FALSE;
}

#ifndef MR_STATIC

__device__ void epoint_getxyz_cuda(_MIPD_ epoint *p,big x,big y,big z)
{ /* get (x,y,z) coordinates */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(143)
    convert_cuda(_MIPP_ 1,mr_mip->w1);
    if (p->marker==MR_EPOINT_INFINITY)
    {
#ifndef MR_AFFINE_ONLY
        if (mr_mip->coord==MR_AFFINE)
        { /* (0,1) or (0,0) = O */
#endif
            if (x!=NULL) zero_cuda(x);
            if (mr_mip->Bsize==0)
            {
                if (y!=NULL) copy_cuda(mr_mip->w1,y);
            }
            else
            {
                if (y!=NULL) zero_cuda(y);
            }
#ifndef MR_AFFINE_ONLY 
        }
        if (mr_mip->coord==MR_PROJECTIVE)
        { /* (1,1,0) = O */
            if (x!=NULL) copy_cuda(mr_mip->w1,x);
            if (y!=NULL) copy_cuda(mr_mip->w1,y);
        }
#endif
        if (z!=NULL) zero_cuda(z);
        MR_OUT
        return;
    }
    if (x!=NULL) redc_cuda(_MIPP_ p->X,x);
    if (y!=NULL) redc_cuda(_MIPP_ p->Y,y);
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    {
#endif
        if (z!=NULL) zero_cuda(z);
#ifndef MR_AFFINE_ONLY
    }

    if (mr_mip->coord==MR_PROJECTIVE)
    {
        if (z!=NULL) 
        {
            if (p->marker!=MR_EPOINT_GENERAL) copy_cuda(mr_mip->w1,z);
            else redc_cuda(_MIPP_ p->Z,z);
        }
    }
#endif
    MR_OUT
    return;
}

#endif

__device__ int epoint_get_cuda(_MIPD_ epoint* p,big x,big y)
{ /* Get point co-ordinates in affine, normal form       *
   * (converted from projective, Montgomery form)        *
   * if x==y, supplies x only. Return value is Least     *
   * Significant Bit of y (useful for point compression) */

    int lsb;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (p->marker==MR_EPOINT_INFINITY)
    {
        zero_cuda(x);
        zero_cuda(y);
        return 0;
    }
    if (mr_mip->ERNUM) return 0;

    MR_IN(98)

    if (!epoint_norm_cuda(_MIPP_ p)) 
    { /* not possible ! */
        MR_OUT
        return (-1);
    }

    redc_cuda(_MIPP_ p->X,x);
    redc_cuda(_MIPP_ p->Y,mr_mip->w1);

    if (x!=y) copy_cuda(mr_mip->w1,y);
    lsb=remain_cuda(_MIPP_ mr_mip->w1,2); 
    MR_OUT
    return lsb;
}

__device__ BOOL epoint_norm_cuda(_MIPD_ epoint *p)
{ /* normalise_cuda a point */
    
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_AFFINE_ONLY

    if (mr_mip->coord==MR_AFFINE) return TRUE;
    if (p->marker!=MR_EPOINT_GENERAL) return TRUE;

    if (mr_mip->ERNUM) return FALSE;

    MR_IN(117)

    copy_cuda(mr_mip->one,mr_mip->w8);

    if (nres_moddiv_cuda(_MIPP_ mr_mip->w8,p->Z,mr_mip->w8)>1) /* 1/Z  */
    {
        epoint_set_cuda(_MIPP_ NULL,NULL,0,p);
        mr_berror_cuda(_MIPP_ MR_ERR_COMPOSITE_MODULUS); 
        MR_OUT
        return FALSE;
    }
    
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w1);/* 1/ZZ */
    nres_modmult_cuda(_MIPP_ p->X,mr_mip->w1,p->X);            /* X/ZZ */
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w8,mr_mip->w1);/* 1/ZZZ */
    nres_modmult_cuda(_MIPP_ p->Y,mr_mip->w1,p->Y);            /* Y/ZZZ */

    copy_cuda(mr_mip->one,p->Z);
   
    p->marker=MR_EPOINT_NORMALIZED;
    MR_OUT

#endif

    return TRUE;
}

__device__ BOOL epoint_multi_norm_cuda(_MIPD_ int m,big *work,epoint **p)
{ /* Normalise an array of points of length m<MR_MAX_M_T_S - requires a workspace array of length m */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
#ifndef MR_AFFINE_ONLY
    int i;
	BOOL inf=FALSE;
    big w[MR_MAX_M_T_S];
    if (mr_mip->coord==MR_AFFINE) return TRUE;
    if (mr_mip->ERNUM) return FALSE;   
    if (m>MR_MAX_M_T_S) return FALSE;

    MR_IN(190)

    for (i=0;i<m;i++)
    {
        if (p[i]->marker==MR_EPOINT_NORMALIZED) w[i]=mr_mip->one;
        else w[i]=p[i]->Z;
		if (p[i]->marker==MR_EPOINT_INFINITY) {inf=TRUE; break;} /* whoops, one of them is point at infinity */
    }

	if (inf)
	{
		for (i=0;i<m;i++) epoint_norm_cuda(_MIPP_ p[i]);
		MR_OUT
		return TRUE;
	}  

    if (!nres_multi_inverse_cuda(_MIPP_ m,w,work)) 
    {
       MR_OUT
       return FALSE;
    }

    for (i=0;i<m;i++)
    {
        copy_cuda(mr_mip->one,p[i]->Z);
        p[i]->marker=MR_EPOINT_NORMALIZED;
        nres_modmult_cuda(_MIPP_ work[i],work[i],mr_mip->w1);
        nres_modmult_cuda(_MIPP_ p[i]->X,mr_mip->w1,p[i]->X);    /* X/ZZ */
        nres_modmult_cuda(_MIPP_ mr_mip->w1,work[i],mr_mip->w1);
        nres_modmult_cuda(_MIPP_ p[i]->Y,mr_mip->w1,p[i]->Y);    /* Y/ZZZ */
    }    
    MR_OUT
#endif
    return TRUE;   
}

/* adds b+=a, d+=c, and slopes in s1 and s2 */

#ifndef MR_NO_ECC_MULTIADD
#ifndef MR_STATIC

__device__ void ecurve_double_add_cuda(_MIPD_ epoint *a,epoint*b,epoint *c,epoint *d,big *s1,big *s2)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;    

    MR_IN(144);

#ifndef MR_AFFINE_ONLY

    if (mr_mip->coord==MR_AFFINE)
    {
#endif
        if (a->marker==MR_EPOINT_INFINITY || size_cuda(a->Y)==0)
        {
            *s1=NULL;
            ecurve_add_cuda(_MIPP_ c,d);
            *s2=mr_mip->w8;
            MR_OUT
            return;
        }
        if (b->marker==MR_EPOINT_INFINITY || size_cuda(b->Y)==0)
        {
            *s1=NULL;
            epoint_copy_cuda(a,b);
            ecurve_add_cuda(_MIPP_ c,d);
            *s2=mr_mip->w8;
            MR_OUT
            return;
        }
        if (c->marker==MR_EPOINT_INFINITY || size_cuda(c->Y)==0)
        {
            ecurve_add_cuda(_MIPP_ a,b);
            *s1=mr_mip->w8;
            *s2=NULL;
            MR_OUT
            return;
        }
        if (d->marker==MR_EPOINT_INFINITY || size_cuda(d->Y)==0)
        {
            epoint_copy_cuda(c,d);
            ecurve_add_cuda(_MIPP_ a,b);
            *s1=mr_mip->w8;
            *s2=NULL;
            MR_OUT
            return;
        }

        if (a==b || (mr_compare_cuda(a->X,b->X)==0 && mr_compare_cuda(a->Y,b->Y)==0))
        {
            nres_modmult_cuda(_MIPP_ a->X,a->X,mr_mip->w8);
            nres_premult_cuda(_MIPP_ mr_mip->w8,3,mr_mip->w8); /* 3x^2 */
            if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
                nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->A,mr_mip->w8);
            else
            {
                convert_cuda(_MIPP_ mr_mip->Asize,mr_mip->w2);
                nres_cuda(_MIPP_ mr_mip->w2,mr_mip->w2);
                nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w2,mr_mip->w8);
            }
            nres_premult_cuda(_MIPP_ a->Y,2,mr_mip->w10);
        }
        else
        {
            if (mr_compare_cuda(a->X,b->X)==0)
            {
                epoint_set_cuda(_MIPP_ NULL,NULL,0,b);
                *s1=NULL;
                ecurve_add_cuda(_MIPP_ c,d);
                *s2=mr_mip->w8;
                MR_OUT
                return;
            }
            nres_modsub_cuda(_MIPP_ a->Y,b->Y,mr_mip->w8);
            nres_modsub_cuda(_MIPP_ a->X,b->X,mr_mip->w10);
        }

        if (c==d || (mr_compare_cuda(c->X,d->X)==0 && mr_compare_cuda(c->Y,d->Y)==0))
        {
            nres_modmult_cuda(_MIPP_ c->X,c->X,mr_mip->w9);
            nres_premult_cuda(_MIPP_ mr_mip->w9,3,mr_mip->w9); /* 3x^2 */
            if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
                nres_modadd_cuda(_MIPP_ mr_mip->w9,mr_mip->A,mr_mip->w9);
            else
            {
                convert_cuda(_MIPP_ mr_mip->Asize,mr_mip->w2);
                nres_cuda(_MIPP_ mr_mip->w2,mr_mip->w2);
                nres_modadd_cuda(_MIPP_ mr_mip->w9,mr_mip->w2,mr_mip->w9);
            }
            nres_premult_cuda(_MIPP_ c->Y,2,mr_mip->w11);
        }
        else
        {
            if (mr_compare_cuda(c->X,d->X)==0)
            {
                epoint_set_cuda(_MIPP_ NULL,NULL,0,d);
                *s2=NULL;
                ecurve_add_cuda(_MIPP_ a,b);
                *s1=mr_mip->w8;
                MR_OUT
                return;
            }
            nres_modsub_cuda(_MIPP_ c->Y,d->Y,mr_mip->w9);
            nres_modsub_cuda(_MIPP_ c->X,d->X,mr_mip->w11);
        }

        nres_double_inverse_cuda(_MIPP_ mr_mip->w10,mr_mip->w10,mr_mip->w11,mr_mip->w11);
        nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w10,mr_mip->w8);
        nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w11,mr_mip->w9);

        nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w2); /* m^2 */
        nres_modsub_cuda(_MIPP_ mr_mip->w2,a->X,mr_mip->w1);
        nres_modsub_cuda(_MIPP_ mr_mip->w1,b->X,mr_mip->w1);

        nres_modsub_cuda(_MIPP_ b->X,mr_mip->w1,mr_mip->w2);
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w8,mr_mip->w2);
        nres_modsub_cuda(_MIPP_ mr_mip->w2,b->Y,b->Y);
        copy_cuda(mr_mip->w1,b->X);
        b->marker=MR_EPOINT_GENERAL;

        nres_modmult_cuda(_MIPP_ mr_mip->w9,mr_mip->w9,mr_mip->w2); /* m^2 */
        nres_modsub_cuda(_MIPP_ mr_mip->w2,c->X,mr_mip->w1);
        nres_modsub_cuda(_MIPP_ mr_mip->w1,d->X,mr_mip->w1);

        nres_modsub_cuda(_MIPP_ d->X,mr_mip->w1,mr_mip->w2);
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w9,mr_mip->w2);
        nres_modsub_cuda(_MIPP_ mr_mip->w2,d->Y,d->Y);
        copy_cuda(mr_mip->w1,d->X);
        d->marker=MR_EPOINT_GENERAL;

        *s1=mr_mip->w8;
        *s2=mr_mip->w9;
#ifndef MR_AFFINE_ONLY
    }
    else
    { /* no speed-up */
        ecurve_add_cuda(_MIPP_ a,b);
        copy_cuda(mr_mip->w8,mr_mip->w9);
        *s1=mr_mip->w9;
        ecurve_add_cuda(_MIPP_ c,d);
        *s2=mr_mip->w8;
    }
#endif
    MR_OUT
}

__device__ void ecurve_multi_add_cuda(_MIPD_ int m,epoint **x,epoint**w)
{ /* adds m points together simultaneously, w[i]+=x[i] */
    int i,*flag;
    big *A,*B,*C;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;    

    MR_IN(122)
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    { /* this can be done faster */
#endif
        A=(big *)mr_alloc_cuda(_MIPP_ m,sizeof(big));
        B=(big *)mr_alloc_cuda(_MIPP_ m,sizeof(big));
        C=(big *)mr_alloc_cuda(_MIPP_ m,sizeof(big));
        flag=(int *)mr_alloc_cuda(_MIPP_ m,sizeof(int));

        copy_cuda(mr_mip->one,mr_mip->w3);

        for (i=0;i<m;i++)
        {
            A[i]=mirvar_cuda(_MIPP_ 0);
            B[i]=mirvar_cuda(_MIPP_ 0);
            C[i]=mirvar_cuda(_MIPP_ 0);
            flag[i]=0;
            if (mr_compare_cuda(x[i]->X,w[i]->X)==0 && mr_compare_cuda(x[i]->Y,w[i]->Y)==0) 
            { /* doubling */
                if (x[i]->marker==MR_EPOINT_INFINITY || size_cuda(x[i]->Y)==0)
                {
                    flag[i]=1;       /* result is infinity */
                    copy_cuda(mr_mip->w3,B[i]);
                    continue;    
                }
                nres_modmult_cuda(_MIPP_ x[i]->X,x[i]->X,A[i]);
                nres_premult_cuda(_MIPP_ A[i],3,A[i]);  /* 3*x^2 */
                if (mr_abs(mr_mip->Asize) == MR_TOOBIG)
                    nres_modadd_cuda(_MIPP_ A[i],mr_mip->A,A[i]);
                else
                {
                    convert_cuda(_MIPP_ mr_mip->Asize,mr_mip->w2);
                    nres_cuda(_MIPP_ mr_mip->w2,mr_mip->w2);
                    nres_modadd_cuda(_MIPP_ A[i],mr_mip->w2,A[i]);
                }                                       /* 3*x^2+A */
                nres_premult_cuda(_MIPP_ x[i]->Y,2,B[i]);
            }
            else
            {
                if (x[i]->marker==MR_EPOINT_INFINITY)
                {
                    flag[i]=2;              /* w[i] unchanged */
                    copy_cuda(mr_mip->w3,B[i]);
                    continue;
                }
                if (w[i]->marker==MR_EPOINT_INFINITY)
                {
                    flag[i]=3;              /* w[i] = x[i] */
                    copy_cuda(mr_mip->w3,B[i]);
                    continue;
                }
                nres_modsub_cuda(_MIPP_ x[i]->X,w[i]->X,B[i]);
                if (size_cuda(B[i])==0)
                { /* point at infinity */
                    flag[i]=1;       /* result is infinity */
                    copy_cuda(mr_mip->w3,B[i]);
                    continue;    
                }
                nres_modsub_cuda(_MIPP_ x[i]->Y,w[i]->Y,A[i]);
            }   
        }
        nres_multi_inverse_cuda(_MIPP_ m,B,C);  /* only one inversion needed */
        for (i=0;i<m;i++)
        {
            if (flag[i]==1)
            { /* point at infinity */
                epoint_set_cuda(_MIPP_ NULL,NULL,0,w[i]);
                continue;
            }
            if (flag[i]==2)
            {
                continue;
            }
            if (flag[i]==3)
            {
                epoint_copy_cuda(x[i],w[i]);
                continue;
            }
            nres_modmult_cuda(_MIPP_ A[i],C[i],mr_mip->w8);

            nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w2); /* m^2 */
            nres_modsub_cuda(_MIPP_ mr_mip->w2,x[i]->X,mr_mip->w1);
            nres_modsub_cuda(_MIPP_ mr_mip->w1,w[i]->X,mr_mip->w1);
       
            nres_modsub_cuda(_MIPP_ w[i]->X,mr_mip->w1,mr_mip->w2);
            nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w8,mr_mip->w2);
            nres_modsub_cuda(_MIPP_ mr_mip->w2,w[i]->Y,w[i]->Y);
            copy_cuda(mr_mip->w1,w[i]->X);
            w[i]->marker=MR_EPOINT_NORMALIZED;

            mr_free_cuda(C[i]);
            mr_free_cuda(B[i]);
            mr_free_cuda(A[i]);
        }
        mr_free_cuda(flag);
        mr_free_cuda(C); mr_free_cuda(B); mr_free_cuda(A);
#ifndef MR_AFFINE_ONLY
    }
    else
    { /* no speed-up */
        for (i=0;i<m;i++) ecurve_add_cuda(_MIPP_ x[i],w[i]);
    }
#endif
    MR_OUT  
}

#endif
#endif

__device__ void ecurve_double_cuda(_MIPD_ epoint *p)
{ /* double epoint on active ecurve */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    if (p->marker==MR_EPOINT_INFINITY) 
    { /* 2 times infinity == infinity ! */
        return;
    }

#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    { /* 2 sqrs, 1 mul, 1 div */
#endif
        if (size_cuda(p->Y)==0) 
        { /* set to point at infinity */
            epoint_set_cuda(_MIPP_ NULL,NULL,0,p);
            return;
        }
 
        nres_modmult_cuda(_MIPP_ p->X,p->X,mr_mip->w8);    /* w8=x^2   */
        nres_premult_cuda(_MIPP_ mr_mip->w8,3,mr_mip->w8); /* w8=3*x^2 */
        if (mr_abs(mr_mip->Asize) == MR_TOOBIG)
            nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->A,mr_mip->w8);
        else
        {
            convert_cuda(_MIPP_ mr_mip->Asize,mr_mip->w2);
            nres_cuda(_MIPP_ mr_mip->w2,mr_mip->w2);
            nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w2,mr_mip->w8);
        }                                     /* w8=3*x^2+A */
        nres_premult_cuda(_MIPP_ p->Y,2,mr_mip->w6);      /* w6=2y */
        if (nres_moddiv_cuda(_MIPP_ mr_mip->w8,mr_mip->w6,mr_mip->w8)>1) 
        {
            epoint_set_cuda(_MIPP_ NULL,NULL,0,p);
            mr_berror_cuda(_MIPP_ MR_ERR_COMPOSITE_MODULUS); 
            return;
        }

/* w8 is slope m on exit */

        nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w2); /* w2=m^2 */
        nres_premult_cuda(_MIPP_ p->X,2,mr_mip->w1);
        nres_modsub_cuda(_MIPP_ mr_mip->w2,mr_mip->w1,mr_mip->w1); /* w1=m^2-2x */
        
        nres_modsub_cuda(_MIPP_ p->X,mr_mip->w1,mr_mip->w2);
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w8,mr_mip->w2);
        nres_modsub_cuda(_MIPP_ mr_mip->w2,p->Y,p->Y);
        copy_cuda(mr_mip->w1,p->X);
        
        return;    
#ifndef MR_AFFINE_ONLY
    }

    if (size_cuda(p->Y)==0)
    { /* set to point at infinity */
        epoint_set_cuda(_MIPP_ NULL,NULL,0,p);
        return;
    }
 
    convert_cuda(_MIPP_ 1,mr_mip->w1);
    if (mr_abs(mr_mip->Asize) < MR_TOOBIG)
    {
        if (mr_mip->Asize!=0)
        {
            if (p->marker==MR_EPOINT_NORMALIZED)
				nres_cuda(_MIPP_ mr_mip->w1,mr_mip->w6);
            else nres_modmult_cuda(_MIPP_ p->Z,p->Z,mr_mip->w6);
        }

        if (mr_mip->Asize==(-3))
        { /* a is -3. Goody. 4 sqrs, 4 muls */
            nres_modsub_cuda(_MIPP_ p->X,mr_mip->w6,mr_mip->w3);
            nres_modadd_cuda(_MIPP_ p->X,mr_mip->w6,mr_mip->w8);
            nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->w8,mr_mip->w3);
            nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w3,mr_mip->w8);
            nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,mr_mip->w8);
        }
        else
        { /* a is small */
            if (mr_mip->Asize!=0)
            { /* a is non zero_cuda! */
                nres_modmult_cuda(_MIPP_ mr_mip->w6,mr_mip->w6,mr_mip->w3);
                nres_premult_cuda(_MIPP_ mr_mip->w3,mr_mip->Asize,mr_mip->w3);
            }
            nres_modmult_cuda(_MIPP_ p->X,p->X,mr_mip->w1);
            nres_modadd_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w8);
            nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w1,mr_mip->w8);
            if (mr_mip->Asize!=0) nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,mr_mip->w8);
        }
    }
    else
    { /* a is not special */
        if (p->marker==MR_EPOINT_NORMALIZED) nres_cuda(_MIPP_ mr_mip->w1,mr_mip->w6);
        else nres_modmult_cuda(_MIPP_ p->Z,p->Z,mr_mip->w6);

        nres_modmult_cuda(_MIPP_ mr_mip->w6,mr_mip->w6,mr_mip->w3);
        nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->A,mr_mip->w3);
        nres_modmult_cuda(_MIPP_ p->X,p->X,mr_mip->w1);
        nres_modadd_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w8);
        nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w1,mr_mip->w8);
        nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,mr_mip->w8);        
    }

/* w8 contains numerator of slope 3x^2+A.z^4  *
 * denominator is now placed in Z             */

    nres_modmult_cuda(_MIPP_ p->Y,p->Y,mr_mip->w2);
    nres_modmult_cuda(_MIPP_ p->X,mr_mip->w2,mr_mip->w3);
    nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w3,mr_mip->w3);
    nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w3,mr_mip->w3);
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,p->X);
    nres_modsub_cuda(_MIPP_ p->X,mr_mip->w3,p->X);
    nres_modsub_cuda(_MIPP_ p->X,mr_mip->w3,p->X);
    
    if (p->marker==MR_EPOINT_NORMALIZED)
        copy_cuda(p->Y,p->Z);
    else nres_modmult_cuda(_MIPP_ p->Z,p->Y,p->Z);
    nres_modadd_cuda(_MIPP_ p->Z,p->Z,p->Z);

    nres_modadd_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w7);
    nres_modmult_cuda(_MIPP_ mr_mip->w7,mr_mip->w7,mr_mip->w2);
    nres_modadd_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w2);
    nres_modsub_cuda(_MIPP_ mr_mip->w3,p->X,mr_mip->w3);
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w3,p->Y);
    nres_modsub_cuda(_MIPP_ p->Y,mr_mip->w2,p->Y);

/* alternative method
    nres_modadd_cuda(_MIPP_ p->Y,p->Y,mr_mip->w2);  

    if (p->marker==MR_EPOINT_NORMALIZED)
        copy_cuda(mr_mip->w2,p->Z);

    else nres_modmult_cuda(_MIPP_ mr_mip->w2,p->Z,p->Z);

    nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w2); 
    nres_modmult_cuda(_MIPP_ p->X,mr_mip->w2,mr_mip->w3);
    nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w3,p->X);
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w1);
    nres_modsub_cuda(_MIPP_ mr_mip->w1,p->X,p->X);
    nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w2);

    if (remain_cuda(_MIPP_ mr_mip->w2,2)!=0)
        mr_padd_cuda(_MIPP_ mr_mip->w2,mr_mip->modulus,mr_mip->w2);
    subdiv_cuda(_MIPP_ mr_mip->w2,2,mr_mip->w2);

    nres_modsub_cuda(_MIPP_ mr_mip->w3,p->X,mr_mip->w3);
    nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->w8,mr_mip->w3);
    nres_modsub_cuda(_MIPP_ mr_mip->w3,mr_mip->w2,p->Y);
*/

/* 

Observe that when finished w8 contains the line_cuda slope, w7 has 2y^2 and w6 has z^2 
This is useful for calculating line_cuda functions in pairings  

*/

    p->marker=MR_EPOINT_GENERAL;
    return;
#endif
}
   
__device__ static BOOL ecurve_padd_cuda(_MIPD_ epoint *p,epoint *pa)
{ /* primitive add_cuda two epoints on the active ecurve - pa+=p;   *
   * note that if p is normalized, its Z coordinate isn't used */
 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    {  /* 1 sqr, 1 mul, 1 div */
#endif
        nres_modsub_cuda(_MIPP_ p->Y,pa->Y,mr_mip->w8);
        nres_modsub_cuda(_MIPP_ p->X,pa->X,mr_mip->w6);
        if (size_cuda(mr_mip->w6)==0) 
        { /* divide_cuda by 0 */
            if (size_cuda(mr_mip->w8)==0) 
            { /* should have doubled ! */
                return FALSE; 
            }
            else
            { /* point at infinity */
                epoint_set_cuda(_MIPP_ NULL,NULL,0,pa);
                return TRUE;
            }
        }
        if (nres_moddiv_cuda(_MIPP_ mr_mip->w8,mr_mip->w6,mr_mip->w8)>1)
        {
            epoint_set_cuda(_MIPP_ NULL,NULL,0,pa);
            mr_berror_cuda(_MIPP_ MR_ERR_COMPOSITE_MODULUS);
            return TRUE;
        }

        nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w2); /* w2=m^2 */
        nres_modsub_cuda(_MIPP_ mr_mip->w2,p->X,mr_mip->w1); /* w1=m^2-x1-x2 */
        nres_modsub_cuda(_MIPP_ mr_mip->w1,pa->X,mr_mip->w1);
        

        nres_modsub_cuda(_MIPP_ pa->X,mr_mip->w1,mr_mip->w2);
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w8,mr_mip->w2);
        nres_modsub_cuda(_MIPP_ mr_mip->w2,pa->Y,pa->Y);
        copy_cuda(mr_mip->w1,pa->X);

        pa->marker=MR_EPOINT_NORMALIZED;
        return TRUE;
#ifndef MR_AFFINE_ONLY
    }

    if (p->marker!=MR_EPOINT_NORMALIZED)    
    {
        nres_modmult_cuda(_MIPP_ p->Z,p->Z,mr_mip->w6);
        nres_modmult_cuda(_MIPP_ pa->X,mr_mip->w6,mr_mip->w1);
        nres_modmult_cuda(_MIPP_ mr_mip->w6,p->Z,mr_mip->w6);
        nres_modmult_cuda(_MIPP_ pa->Y,mr_mip->w6,mr_mip->w8);
    }
    else
    {
        copy_cuda(pa->X,mr_mip->w1);
        copy_cuda(pa->Y,mr_mip->w8);
    }
    if (pa->marker==MR_EPOINT_NORMALIZED)
        copy_cuda(mr_mip->one,mr_mip->w6);
    else nres_modmult_cuda(_MIPP_ pa->Z,pa->Z,mr_mip->w6);

    nres_modmult_cuda(_MIPP_ p->X,mr_mip->w6,mr_mip->w4);
    if (pa->marker!=MR_EPOINT_NORMALIZED) 
        nres_modmult_cuda(_MIPP_ mr_mip->w6,pa->Z,mr_mip->w6);
    nres_modmult_cuda(_MIPP_ p->Y,mr_mip->w6,mr_mip->w5);
    nres_modsub_cuda(_MIPP_ mr_mip->w1,mr_mip->w4,mr_mip->w1);
    nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w5,mr_mip->w8);

/* w8 contains the numerator of the slope */

    if (size_cuda(mr_mip->w1)==0)
    {
        if (size_cuda(mr_mip->w8)==0)
        { /* should have doubled ! */
           return FALSE; 
        }
        else
        { /* point at infinity */
            epoint_set_cuda(_MIPP_ NULL,NULL,0,pa);
            return TRUE;
        }
    }
    nres_modadd_cuda(_MIPP_ mr_mip->w4,mr_mip->w4,mr_mip->w6);
    nres_modadd_cuda(_MIPP_ mr_mip->w1,mr_mip->w6,mr_mip->w4);
    nres_modadd_cuda(_MIPP_ mr_mip->w5,mr_mip->w5,mr_mip->w6);
    nres_modadd_cuda(_MIPP_ mr_mip->w8,mr_mip->w6,mr_mip->w5);
    
    if (p->marker!=MR_EPOINT_NORMALIZED)
    { 
        if (pa->marker!=MR_EPOINT_NORMALIZED) 
            nres_modmult_cuda(_MIPP_ pa->Z,p->Z,mr_mip->w3);
        else
            copy_cuda(p->Z,mr_mip->w3);
        nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->w1,pa->Z);
    }
    else
    {
        if (pa->marker!=MR_EPOINT_NORMALIZED)
            nres_modmult_cuda(_MIPP_ pa->Z,mr_mip->w1,pa->Z);
        else
            copy_cuda(mr_mip->w1,pa->Z);
    }
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w6);
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w6,mr_mip->w1);
    nres_modmult_cuda(_MIPP_ mr_mip->w6,mr_mip->w4,mr_mip->w6);
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w8,mr_mip->w4);

    nres_modsub_cuda(_MIPP_ mr_mip->w4,mr_mip->w6,pa->X);
    nres_modsub_cuda(_MIPP_ mr_mip->w6,pa->X,mr_mip->w6);
    nres_modsub_cuda(_MIPP_ mr_mip->w6,pa->X,mr_mip->w6);
    nres_modmult_cuda(_MIPP_ mr_mip->w8,mr_mip->w6,mr_mip->w2);
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w5,mr_mip->w1);
    nres_modsub_cuda(_MIPP_ mr_mip->w2,mr_mip->w1,mr_mip->w5);

/* divide_cuda by 2 */

    nres_div2_cuda(_MIPP_ mr_mip->w5,pa->Y);

    pa->marker=MR_EPOINT_GENERAL;
    return TRUE;      
#endif
}

__device__ void epoint_copy_cuda(epoint *a,epoint *b)
{   
    if (a==b || b==NULL) return;

    copy_cuda(a->X,b->X);
    copy_cuda(a->Y,b->Y);
#ifndef MR_AFFINE_ONLY
    if (a->marker==MR_EPOINT_GENERAL) copy_cuda(a->Z,b->Z);
#endif
    b->marker=a->marker;
    return;
}

__device__ BOOL epoint_comp_cuda(_MIPD_ epoint *a,epoint *b)
{
    BOOL result;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;
    if (a==b) return TRUE;
    if (a->marker==MR_EPOINT_INFINITY)
    {
        if (b->marker==MR_EPOINT_INFINITY) return TRUE;
        else return FALSE;
    }
    if (b->marker==MR_EPOINT_INFINITY)
        return FALSE;
    
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    {
#endif
        if (mr_compare_cuda(a->X,b->X)==0 && mr_compare_cuda(a->Y,b->Y)==0) result=TRUE;
        else result=FALSE;
        return result;
#ifndef MR_AFFINE_ONLY
    }

    if (mr_mip->coord==MR_PROJECTIVE)
    {
        MR_IN(105)
        if (a->marker!=MR_EPOINT_GENERAL) 
            copy_cuda(mr_mip->one,mr_mip->w1);
        else copy_cuda(a->Z,mr_mip->w1);

        if (b->marker!=MR_EPOINT_GENERAL) 
            copy_cuda(mr_mip->one,mr_mip->w2);
        else copy_cuda(b->Z,mr_mip->w2);

        nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w3); /* Za*Za */
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w2,mr_mip->w4); /* Zb*Zb */

        nres_modmult_cuda(_MIPP_ a->X,mr_mip->w4,mr_mip->w5); /* Xa*Zb*Zb */
        nres_modmult_cuda(_MIPP_ b->X,mr_mip->w3,mr_mip->w6); /* Xb*Za*Za */

        if (mr_compare_cuda(mr_mip->w5,mr_mip->w6)!=0) result=FALSE;
        else
        {
            nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w3,mr_mip->w3);
            nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w4,mr_mip->w4);

            nres_modmult_cuda(_MIPP_ a->Y,mr_mip->w4,mr_mip->w5);
            nres_modmult_cuda(_MIPP_ b->Y,mr_mip->w3,mr_mip->w6);

            if (mr_compare_cuda(mr_mip->w5,mr_mip->w6)!=0) result=FALSE;
            else result=TRUE;
        }
        MR_OUT
        return result;
    }
    return FALSE;
#endif
}

__device__ int ecurve_add_cuda(_MIPD_ epoint *p,epoint *pa)
{  /* pa=pa+p; */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return MR_OVER;

    MR_IN(94)

    if (p==pa) 
    {
        ecurve_double_cuda(_MIPP_ pa);
        MR_OUT
        if (pa->marker==MR_EPOINT_INFINITY) return MR_OVER;
        return MR_DOUBLE;
    }
    if (pa->marker==MR_EPOINT_INFINITY)
    {
        epoint_copy_cuda(p,pa);
        MR_OUT 
        return MR_ADD;
    }
    if (p->marker==MR_EPOINT_INFINITY) 
    {
        MR_OUT
        return MR_ADD;
    }

    if (!ecurve_padd_cuda(_MIPP_ p,pa))
    {    
        ecurve_double_cuda(_MIPP_ pa);
        MR_OUT
        return MR_DOUBLE;
    }
    MR_OUT
    if (pa->marker==MR_EPOINT_INFINITY) return MR_OVER;
    return MR_ADD;
}

__device__ void epoint_negate_cuda(_MIPD_ epoint *p)
{ /* negate a point */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    if (p->marker==MR_EPOINT_INFINITY) return;

    MR_IN(121)
    if (size_cuda(p->Y)!=0) mr_psub_cuda(_MIPP_ mr_mip->modulus,p->Y,p->Y);
    MR_OUT
}

__device__ int ecurve_sub_cuda(_MIPD_ epoint *p,epoint *pa)
{
    int r;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return MR_OVER;

    MR_IN(104)

    if (p==pa)
    {
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pa);
        MR_OUT
        return MR_OVER;
    } 
    if (p->marker==MR_EPOINT_INFINITY) 
    {
        MR_OUT
        return MR_ADD;
    }

    epoint_negate_cuda(_MIPP_ p);
    r=ecurve_add_cuda(_MIPP_ p,pa);
    epoint_negate_cuda(_MIPP_ p);

    MR_OUT
    return r;
}

__device__ int ecurve_mult_cuda(_MIPD_ big e,epoint *pa,epoint *pt)
{ /* pt=e*pa; */
    int i,j,n,nb,nbs,nzs,nadds;
    epoint *table[MR_ECC_STORE_N];
#ifndef MR_AFFINE_ONLY
    big work[MR_ECC_STORE_N];
#endif

#ifdef MR_STATIC
    char mem[MR_ECP_RESERVE(MR_ECC_STORE_N)];  
#ifndef MR_AFFINE_ONLY
    char mem1[MR_BIG_RESERVE(MR_ECC_STORE_N)];
#endif
#else
    char *mem;
#ifndef MR_AFFINE_ONLY
    char *mem1;
#endif
#endif

#ifndef MR_ALWAYS_BINARY
    epoint *p;
    int ce,ch;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return 0;

    MR_IN(95)
    if (size_cuda(e)==0) 
    { /* multiplied by 0 */
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);
        MR_OUT
        return 0;
    }
    copy_cuda(e,mr_mip->w9);
/*    epoint_norm_cuda(_MIPP_ pa); */
    epoint_copy_cuda(pa,pt);

    if (size_cuda(mr_mip->w9)<0)
    { /* pt = -pt */
        negify_cuda(mr_mip->w9,mr_mip->w9);
        epoint_negate_cuda(_MIPP_ pt);
    }

    if (size_cuda(mr_mip->w9)==1)
    { 
        MR_OUT
        return 0;
    }

    premult_cuda(_MIPP_ mr_mip->w9,3,mr_mip->w10);      /* h=3*e */

#ifndef MR_STATIC
#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
#endif

#ifdef  MR_STATIC
        memset(mem,0,MR_ECP_RESERVE(MR_ECC_STORE_N));
#ifndef MR_AFFINE_ONLY
        memset(mem1,0,MR_BIG_RESERVE(MR_ECC_STORE_N));
#endif
#else
        mem=(char *)ecp_memalloc_cuda(_MIPP_ MR_ECC_STORE_N);
#ifndef MR_AFFINE_ONLY
        mem1=(char *)memalloc_cuda(_MIPP_ MR_ECC_STORE_N);
#endif
#endif

        for (i=0;i<=MR_ECC_STORE_N-1;i++)
        {
            table[i]=epoint_init_mem_cuda(_MIPP_ mem,i);
#ifndef MR_AFFINE_ONLY
            work[i]=mirvar_mem_cuda(_MIPP_ mem1,i);
#endif
        }

        epoint_copy_cuda(pt,table[0]);
        epoint_copy_cuda(table[0],table[MR_ECC_STORE_N-1]);
        ecurve_double_cuda(_MIPP_ table[MR_ECC_STORE_N-1]);
     /*   epoint_norm_cuda(_MIPP_ table[MR_ECC_STORE_N-1]); */

        for (i=1;i<MR_ECC_STORE_N-1;i++)
        { /* precomputation */
            epoint_copy_cuda(table[i-1],table[i]);
            ecurve_add_cuda(_MIPP_ table[MR_ECC_STORE_N-1],table[i]);
        }
        ecurve_add_cuda(_MIPP_ table[MR_ECC_STORE_N-2],table[MR_ECC_STORE_N-1]);

#ifndef MR_AFFINE_ONLY
        epoint_multi_norm_cuda(_MIPP_ MR_ECC_STORE_N,work,table);
#endif

        nb=logb2_cuda(_MIPP_ mr_mip->w10);
        nadds=0;
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);
        for (i=nb-1;i>=1;)
        { /* add_cuda/subtract_cuda */
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            n=mr_naf_window_cuda(_MIPP_ mr_mip->w9,mr_mip->w10,i,&nbs,&nzs,MR_ECC_STORE_N);
            for (j=0;j<nbs;j++)
                ecurve_double_cuda(_MIPP_ pt);
            if (n>0) {ecurve_add_cuda(_MIPP_ table[n/2],pt); nadds++;}
            if (n<0) {ecurve_sub_cuda(_MIPP_ table[(-n)/2],pt); nadds++;}
            i-=nbs;
            if (nzs)
            {
                for (j=0;j<nzs;j++) ecurve_double_cuda(_MIPP_ pt);
                i-=nzs;
            }
        }

        ecp_memkill_cuda(_MIPP_ mem,MR_ECC_STORE_N);
#ifndef MR_AFFINE_ONLY
        memkill_cuda(_MIPP_ mem1,MR_ECC_STORE_N);
#endif

#ifndef MR_STATIC
#ifndef MR_ALWAYS_BINARY
    }
    else
    { 
        mem=(char *)ecp_memalloc_cuda(_MIPP_ 1);
        p=epoint_init_mem_cuda(_MIPP_ mem,0);
        epoint_norm_cuda(_MIPP_ pt);
        epoint_copy_cuda(pt,p);

        nadds=0;
        expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w10)-1,mr_mip->w11);
        mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
        subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        while (size_cuda(mr_mip->w11) > 1)
        { /* add_cuda/subtract_cuda method */
            if (mr_mip->user!=NULL) (*mr_mip->user)();

            ecurve_double_cuda(_MIPP_ pt);
            ce=mr_compare_cuda(mr_mip->w9,mr_mip->w11); /* e(i)=1? */
            ch=mr_compare_cuda(mr_mip->w10,mr_mip->w11); /* h(i)=1? */
            if (ch>=0) 
            {  /* h(i)=1 */
                if (ce<0) {ecurve_add_cuda(_MIPP_ p,pt); nadds++;}
                mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
            }
            if (ce>=0) 
            {  /* e(i)=1 */
                if (ch<0) {ecurve_sub_cuda(_MIPP_ p,pt); nadds++;}
                mr_psub_cuda(_MIPP_ mr_mip->w9,mr_mip->w11,mr_mip->w9);  
            }
            subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        }
        ecp_memkill_cuda(_MIPP_ mem,1);
    }
#endif
#endif
    MR_OUT
    return nadds;
}

#ifndef MR_NO_ECC_MULTIADD
#ifndef MR_STATIC

__device__ void ecurve_multn_cuda(_MIPD_ int n,big *y,epoint **x,epoint *w)
{ /* pt=e[o]*p[0]+e[1]*p[1]+ .... e[n-1]*p[n-1]   */
    int i,j,k,m,nb,ea;
    epoint **G;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(114)

    m=1<<n;
    G=(epoint **)mr_alloc_cuda(_MIPP_ m,sizeof(epoint*));

    for (i=0,k=1;i<n;i++)
    {
        for (j=0; j < (1<<i) ;j++)
        {
            G[k]=epoint_init_cuda(_MIPPO_ );
            epoint_copy_cuda(x[i],G[k]);
            if (j!=0) ecurve_add_cuda(_MIPP_ G[j],G[k]);
            k++;
        }
    }

    nb=0;
    for (j=0;j<n;j++) if ((k=logb2_cuda(_MIPP_ y[j])) > nb) nb=k;

    epoint_set_cuda(_MIPP_ NULL,NULL,0,w);            /* w=0 */
    
#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        for (i=nb-1;i>=0;i--)
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ea=0;
            k=1;
            for (j=0;j<n;j++)
            {
                if (mr_testbit_cuda(_MIPP_ y[j],i)) ea+=k;
                k<<=1;
            }
            ecurve_double_cuda(_MIPP_ w);
            if (ea!=0) ecurve_add_cuda(_MIPP_ G[ea],w);
        }    
#ifndef MR_ALWAYS_BINARY
    }
    else mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
#endif

    for (i=1;i<m;i++) epoint_free_cuda(G[i]);
    mr_free_cuda(G);
    MR_OUT
}

#endif

/* PP=P+Q, PM=P-Q. Assumes P and Q are both normalized, and P!=Q */

__device__ static BOOL ecurve_add_sub_cuda(_MIPD_ epoint *P,epoint *Q,epoint *PP,epoint *PM)
{ 
 #ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    big t1,t2,lam;

    if (mr_mip->ERNUM) return FALSE;

    if (P->marker==MR_EPOINT_GENERAL || Q->marker==MR_EPOINT_GENERAL)
    { 
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return FALSE;
    }

    if (mr_compare_cuda(P->X,Q->X)==0)
    { /* P=Q or P=-Q - shouldn't happen */
        epoint_copy_cuda(P,PP);
        ecurve_add_cuda(_MIPP_ Q,PP);
        epoint_copy_cuda(P,PM);
        ecurve_sub_cuda(_MIPP_ Q,PM);

        MR_OUT
        return TRUE;
    }

    t1= mr_mip->w10;
    t2= mr_mip->w11; 
    lam = mr_mip->w13;   

    copy_cuda(P->X,t2);
    nres_modsub_cuda(_MIPP_ t2,Q->X,t2);

    redc_cuda(_MIPP_ t2,t2);
    invmodp_cuda(_MIPP_ t2,mr_mip->modulus,t2);
    nres_cuda(_MIPP_ t2,t2);
    
    nres_modadd_cuda(_MIPP_ P->X,Q->X,PP->X);
    copy_cuda(PP->X,PM->X);

    copy_cuda(P->Y,t1);
    nres_modsub_cuda(_MIPP_ t1,Q->Y,t1);
    copy_cuda(t1,lam);
    nres_modmult_cuda(_MIPP_ lam,t2,lam);
    copy_cuda(lam,t1);
    nres_modmult_cuda(_MIPP_ t1,t1,t1);
    nres_modsub_cuda(_MIPP_ t1,PP->X,PP->X);
    copy_cuda(Q->X,PP->Y);
    nres_modsub_cuda(_MIPP_ PP->Y,PP->X,PP->Y);
    nres_modmult_cuda(_MIPP_ PP->Y,lam,PP->Y);
    nres_modsub_cuda(_MIPP_ PP->Y,Q->Y,PP->Y);

    copy_cuda(P->Y,t1);
    nres_modadd_cuda(_MIPP_ t1,Q->Y,t1);
    copy_cuda(t1,lam);
    nres_modmult_cuda(_MIPP_ lam,t2,lam);
    copy_cuda(lam,t1);
    nres_modmult_cuda(_MIPP_ t1,t1,t1);
    nres_modsub_cuda(_MIPP_ t1,PM->X,PM->X);
    copy_cuda(Q->X,PM->Y);
    nres_modsub_cuda(_MIPP_ PM->Y,PM->X,PM->Y);
    nres_modmult_cuda(_MIPP_ PM->Y,lam,PM->Y);
    nres_modadd_cuda(_MIPP_ PM->Y,Q->Y,PM->Y);

    PP->marker=MR_EPOINT_NORMALIZED;
    PM->marker=MR_EPOINT_NORMALIZED;

    return TRUE;
}

__device__ void ecurve_mult2_cuda(_MIPD_ big e,epoint *p,big ea,epoint *pa,epoint *pt)
{ /* pt=e*p+ea*pa; */
    int e1,h1,e2,h2,bb;
    epoint *p1,*p2,*ps[2];
#ifdef MR_STATIC
    char mem[MR_ECP_RESERVE(4)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;

    MR_IN(103)

    if (size_cuda(e)==0) 
    {
        ecurve_mult_cuda(_MIPP_ ea,pa,pt);
        MR_OUT
        return;
    }
#ifdef MR_STATIC
    memset(mem,0,MR_ECP_RESERVE(4));
#else
    mem=(char *)ecp_memalloc_cuda(_MIPP_ 4);
#endif
    p2=epoint_init_mem_cuda(_MIPP_ mem,0);
    p1=epoint_init_mem_cuda(_MIPP_ mem,1);
    ps[0]=epoint_init_mem_cuda(_MIPP_ mem,2);
    ps[1]=epoint_init_mem_cuda(_MIPP_ mem,3);

    epoint_norm_cuda(_MIPP_ pa);
    epoint_copy_cuda(pa,p2);
    copy_cuda(ea,mr_mip->w9);
    if (size_cuda(mr_mip->w9)<0)
    { /* p2 = -p2 */
        negify_cuda(mr_mip->w9,mr_mip->w9);
        epoint_negate_cuda(_MIPP_ p2);
    }

    epoint_norm_cuda(_MIPP_ p);
    epoint_copy_cuda(p,p1);
    copy_cuda(e,mr_mip->w12);
    if (size_cuda(mr_mip->w12)<0)
    { /* p1= -p1 */
        negify_cuda(mr_mip->w12,mr_mip->w12);
        epoint_negate_cuda(_MIPP_ p1);
    }


    epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);            /* pt=0 */
    ecurve_add_sub_cuda(_MIPP_ p1,p2,ps[0],ps[1]);     /* only one inversion! ps[0]=p1+p2, ps[1]=p1-p2 */

    mr_jsf_cuda(_MIPP_ mr_mip->w9,mr_mip->w12,mr_mip->w10,mr_mip->w9,mr_mip->w13,mr_mip->w12);
  
/*    To use a simple NAF instead, substitute this for the JSF 
        premult_cuda(_MIPP_ mr_mip->w9,3,mr_mip->w10);      3*ea  
        premult_cuda(_MIPP_ mr_mip->w12,3,mr_mip->w13);     3*e  
*/ 

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        if (mr_compare_cuda(mr_mip->w10,mr_mip->w13)>=0) bb=logb2_cuda(_MIPP_ mr_mip->w10)-1;
        else                                        bb=logb2_cuda(_MIPP_ mr_mip->w13)-1;

        while (bb>=0) /* for the simple NAF, this should be 1 */
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ecurve_double_cuda(_MIPP_ pt);

            e1=h1=e2=h2=0;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w9,bb)) e2=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w10,bb)) h2=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w12,bb)) e1=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w13,bb)) h1=1;

            if (e1!=h1)
            {
                if (e2==h2)
                {
                    if (h1==1) ecurve_add_cuda(_MIPP_ p1,pt);
                    else       ecurve_sub_cuda(_MIPP_ p1,pt);
                }
                else
                {
                    if (h1==1)
                    {
                        if (h2==1) ecurve_add_cuda(_MIPP_ ps[0],pt);
                        else       ecurve_add_cuda(_MIPP_ ps[1],pt);
                    }
                    else
                    {
                        if (h2==1) ecurve_sub_cuda(_MIPP_ ps[1],pt);
                        else       ecurve_sub_cuda(_MIPP_ ps[0],pt);
                    }
                }
            }
            else if (e2!=h2)
            {
                if (h2==1) ecurve_add_cuda(_MIPP_ p2,pt);
                else       ecurve_sub_cuda(_MIPP_ p2,pt);
            }
            bb-=1;
        }
#ifndef MR_ALWAYS_BINARY
    }
    else
    {
         if (mr_compare_cuda(mr_mip->w10,mr_mip->w13)>=0)
              expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w10)-1,mr_mip->w11);
         else expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w13)-1,mr_mip->w11);

        while (size_cuda(mr_mip->w11) > 0)    /* for the NAF, this should be 1 */
        { /* add_cuda/subtract_cuda method */
            if (mr_mip->user!=NULL) (*mr_mip->user)();

            ecurve_double_cuda(_MIPP_ pt);

            e1=h1=e2=h2=0;
            if (mr_compare_cuda(mr_mip->w9,mr_mip->w11)>=0)
            { /* e1(i)=1? */
                e2=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w9,mr_mip->w11,mr_mip->w9);
            }
            if (mr_compare_cuda(mr_mip->w10,mr_mip->w11)>=0)
            { /* h1(i)=1? */
                h2=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
            } 
            if (mr_compare_cuda(mr_mip->w12,mr_mip->w11)>=0)
            { /* e2(i)=1? */
                e1=1;   
                mr_psub_cuda(_MIPP_ mr_mip->w12,mr_mip->w11,mr_mip->w12);
            }
            if (mr_compare_cuda(mr_mip->w13,mr_mip->w11)>=0) 
            { /* h2(i)=1? */
                h1=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w13,mr_mip->w11,mr_mip->w13);
            }

            if (e1!=h1)
            {
                if (e2==h2)
                {
                    if (h1==1) ecurve_add_cuda(_MIPP_ p1,pt);
                    else       ecurve_sub_cuda(_MIPP_ p1,pt);
                }
                else
                {
                    if (h1==1)
                    {
                        if (h2==1) ecurve_add_cuda(_MIPP_ ps[0],pt);
                        else       ecurve_add_cuda(_MIPP_ ps[1],pt);
                    }
                    else
                    {
                        if (h2==1) ecurve_sub_cuda(_MIPP_ ps[1],pt);
                        else       ecurve_sub_cuda(_MIPP_ ps[0],pt);
                    }
                }
            }
            else if (e2!=h2)
            {
                if (h2==1) ecurve_add_cuda(_MIPP_ p2,pt);
                else       ecurve_sub_cuda(_MIPP_ p2,pt);
            }

            subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        }
    }
#endif
    ecp_memkill_cuda(_MIPP_ mem,4);
    MR_OUT
}

#endif

#else

/*   Twisted Inverted Edwards curves 

 *   Assumes Twisted Inverted Edward's equation x^2+Ay^2 = x^2.y^2 + B
 *   Assumes points are not of order 2 or 4
*/

__device__ static void epoint_getrhs_cuda(_MIPD_ big x,big y)
{ 
  /* find RHS=(x^2-B)/(x^2-A) */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
    nres_modmult_cuda(_MIPP_ x,x,mr_mip->w6);
    nres_modsub_cuda(_MIPP_ mr_mip->w6,mr_mip->B,y);  
    nres_modsub_cuda(_MIPP_ mr_mip->w6,mr_mip->A,mr_mip->w6);

    nres_moddiv_cuda(_MIPP_ y,mr_mip->w6,y);
}

#ifndef MR_NOSUPPORT_COMPRESSION

__device__ BOOL epoint_x_cuda(_MIPD_ big x)
{ /* test if x is associated with a point on the   *
   * currently active curve                        */
    int j;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(147)
    
    if (x==NULL) return FALSE;

    nres_cuda(_MIPP_ x,mr_mip->w2);
    epoint_getrhs_cuda(_MIPP_ mr_mip->w2,mr_mip->w7);

    if (size_cuda(mr_mip->w7)==0)
    {
        MR_OUT
        return TRUE;
    }

    redc_cuda(_MIPP_ mr_mip->w7,mr_mip->w4);
    j=jack_cuda(_MIPP_ mr_mip->w4,mr_mip->modulus);

    MR_OUT
    if (j==1) return TRUE;
    return FALSE;
}

#endif

__device__ BOOL epoint_set_cuda(_MIPD_ big x,big y,int cb,epoint *p)
{ /* initialise a point on active ecurve            *
   * if x or y == NULL, set to point at infinity    *
   * if x==y, a y co-ordinate is calculated - if    *
   * possible - and cb suggests LSB 0/1  of y       *
   * (which "decompresses" y). Otherwise, check     *
   * validity of given (x,y) point, ignoring cb.    *
   * Returns TRUE for valid point, otherwise FALSE. */
  
    BOOL valid;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(97)

    if (x==NULL || y==NULL)
    {
        copy_cuda(mr_mip->one,p->X);
        zero_cuda(p->Y); 
        p->marker=MR_EPOINT_INFINITY;
        MR_OUT
        return TRUE;
    }

    valid=FALSE;
	nres_cuda(_MIPP_ x,p->X);
	if (x!=y)
	{ /* Check directly that x^2+Ay^2 == x^2.y^2+B */
		nres_cuda(_MIPP_ y,p->Y);
		nres_modmult_cuda(_MIPP_ p->X,p->X,mr_mip->w1);
		nres_modmult_cuda(_MIPP_ p->Y,p->Y,mr_mip->w2);
		nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w3);
		nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->B,mr_mip->w3);


		if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
			nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->A,mr_mip->w2);
		else
			nres_premult_cuda(_MIPP_ mr_mip->w2,mr_mip->Asize,mr_mip->w2);   
		nres_modadd_cuda(_MIPP_ mr_mip->w2,mr_mip->w1,mr_mip->w2);
		if (mr_compare_cuda(mr_mip->w2,mr_mip->w3)==0) valid=TRUE;
	}
	else
	{ /* find RHS */
		epoint_getrhs_cuda(_MIPP_ p->X,mr_mip->w7);
     /* no y supplied - calculate one. Find square root */
#ifndef MR_NOSUPPORT_COMPRESSION
        valid=nres_sqroot_cuda(_MIPP_ mr_mip->w7,p->Y);
    /* check LSB - have we got the right root? */
        redc_cuda(_MIPP_ p->Y,mr_mip->w1);
        if (remain_cuda(_MIPP_ mr_mip->w1,2)!=cb) 
            mr_psub_cuda(_MIPP_ mr_mip->modulus,p->Y,p->Y);

#else
		mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
		MR_OUT
		return FALSE;
#endif
    } 
    if (valid)
    {
        p->marker=MR_EPOINT_NORMALIZED;
        MR_OUT
        return TRUE;
    }

    MR_OUT
    return FALSE;
}

#ifndef MR_STATIC

__device__ void epoint_getxyz_cuda(_MIPD_ epoint *p,big x,big y,big z)
{ /* get (x,y,z) coordinates */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(143)
    convert_cuda(_MIPP_ 1,mr_mip->w1);
    if (p->marker==MR_EPOINT_INFINITY)
    {
        if (x!=NULL) copy_cuda(mr_mip->w1,x);
        if (y!=NULL) zero_cuda(y);
        if (z!=NULL) zero_cuda(z);
        MR_OUT
        return;
    }
    if (x!=NULL) redc_cuda(_MIPP_ p->X,x);
    if (y!=NULL) redc_cuda(_MIPP_ p->Y,y);
    if (z!=NULL) redc_cuda(_MIPP_ p->Z,z);

    MR_OUT
    return;
}

#endif

__device__ int epoint_get_cuda(_MIPD_ epoint* p,big x,big y)
{ /* Get point co-ordinates in affine, normal form       *
   * (converted from projective, Montgomery form)        *
   * if x==y, supplies x only. Return value is Least     *
   * Significant Bit of y (useful for point compression) */

    int lsb;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (p->marker==MR_EPOINT_INFINITY)
    {
        zero_cuda(y);
        convert_cuda(_MIPP_ 1,x);
        return 0;
    }
    if (mr_mip->ERNUM) return 0;

    MR_IN(98)

    if (!epoint_norm_cuda(_MIPP_ p)) 
    { /* not possible ! */
        MR_OUT
        return (-1);
    }

    redc_cuda(_MIPP_ p->X,x);
    redc_cuda(_MIPP_ p->Y,mr_mip->w1);

    if (x!=y) copy_cuda(mr_mip->w1,y);
    lsb=remain_cuda(_MIPP_ mr_mip->w1,2); 
    MR_OUT
    return lsb;
}

__device__ BOOL epoint_norm_cuda(_MIPD_ epoint *p)
{ /* normalise_cuda a point */
    
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (p->marker!=MR_EPOINT_GENERAL) return TRUE;

    if (mr_mip->ERNUM) return FALSE;

    MR_IN(117)

    copy_cuda(mr_mip->one,mr_mip->w8);

    if (nres_moddiv_cuda(_MIPP_ mr_mip->w8,p->Z,mr_mip->w8)>1) /* 1/Z  */
    {
        epoint_set_cuda(_MIPP_ NULL,NULL,0,p);
        mr_berror_cuda(_MIPP_ MR_ERR_COMPOSITE_MODULUS); 
        MR_OUT
        return FALSE;
    }
    
    nres_modmult_cuda(_MIPP_ p->X,mr_mip->w8,p->X);            /* X/Z */
    nres_modmult_cuda(_MIPP_ p->Y,mr_mip->w8,p->Y);            /* Y/Z */

    copy_cuda(mr_mip->one,p->Z);
   
    p->marker=MR_EPOINT_NORMALIZED;
    MR_OUT

    return TRUE;
}

__device__ void ecurve_double_cuda(_MIPD_ epoint *p)
{ /* double epoint on active ecurve */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    if (p->marker==MR_EPOINT_INFINITY) 
    { /* 2 times infinity == infinity ! */
        return;
    }
    nres_modadd_cuda(_MIPP_ p->X,p->Y,mr_mip->w1);

    nres_modmult_cuda(_MIPP_ p->X,p->X,p->X);                 /* A=X1^2        */
    nres_modmult_cuda(_MIPP_ p->Y,p->Y,p->Y);                 /* B=Y1^2        */
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w1);          /* (X+Y)^2       */
    nres_modsub_cuda(_MIPP_ mr_mip->w1,p->X,mr_mip->w1);
    nres_modsub_cuda(_MIPP_ mr_mip->w1,p->Y,mr_mip->w1);           /* E=(X+Y)^2-A-B   */

    if (mr_abs(mr_mip->Asize)==MR_TOOBIG)                      /* U = aB        */
        nres_modmult_cuda(_MIPP_ p->Y,mr_mip->A,p->Y);
    else
        nres_premult_cuda(_MIPP_ p->Y,mr_mip->Asize,p->Y);   

    if (p->marker!=MR_EPOINT_NORMALIZED)
        nres_modmult_cuda(_MIPP_ p->Z,p->Z,p->Z);
    else
        copy_cuda(mr_mip->one,p->Z);

    nres_modadd_cuda(_MIPP_ p->Z,p->Z,p->Z);
    if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)                           /* 2dZ^2 */
        nres_modmult_cuda(_MIPP_ p->Z,mr_mip->B,p->Z);
    else
        nres_premult_cuda(_MIPP_ p->Z,mr_mip->Bsize,p->Z);  
  
    nres_modadd_cuda(_MIPP_ p->X,p->Y,mr_mip->w2);           /* C=A+U         */
    nres_modsub_cuda(_MIPP_ p->X,p->Y,mr_mip->w3);           /* D=A-U         */

    nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w3,p->X);                /* X=C.D */

    nres_modsub_cuda(_MIPP_ mr_mip->w2,p->Z,mr_mip->w2);           /* C-2dZ^2 */
    nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->w1,p->Y);                /* Y=E.(C-2dZ^2) */
    nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->w1,p->Z);                /* Z=D.E */

    p->marker=MR_EPOINT_GENERAL;
    return;
}
   
__device__ static BOOL ecurve_padd_cuda(_MIPD_ epoint *p,epoint *pa)
{ /* primitive add_cuda two epoints on the active ecurve - pa+=p;   *
   * note that if p is normalized, its Z coordinate isn't used */
 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (p->marker==MR_EPOINT_INFINITY) return TRUE;
    if (pa->marker==MR_EPOINT_INFINITY)
    {
        epoint_copy_cuda(p,pa);
        return TRUE;
    }

    nres_modadd_cuda(_MIPP_ p->X,p->Y,mr_mip->w1);
    nres_modadd_cuda(_MIPP_ pa->X,pa->Y,mr_mip->w2);
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w1);  /* I=(X1+Y1)(X2+Y2) */
    if (p->marker!=MR_EPOINT_NORMALIZED) 
    {
        if (pa->marker==MR_EPOINT_NORMALIZED)
            copy_cuda(p->Z,pa->Z);
        else nres_modmult_cuda(_MIPP_ p->Z,pa->Z,pa->Z);    /* z = A = Z1*Z2 */
    }
    else
    {
        if (pa->marker==MR_EPOINT_NORMALIZED) copy_cuda(mr_mip->one,pa->Z);
    }

    nres_modmult_cuda(_MIPP_ pa->Z,pa->Z,mr_mip->w2);       /* w2 = B = dA^2   */
    if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)                   
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->B,mr_mip->w2);
    else
        nres_premult_cuda(_MIPP_ mr_mip->w2,mr_mip->Bsize,mr_mip->w2);
    nres_modmult_cuda(_MIPP_ p->X,pa->X,pa->X);             /* x = C = X1*X2 */
    nres_modmult_cuda(_MIPP_ p->Y,pa->Y,pa->Y);             /* y = D = Y1*Y2 */
    nres_modmult_cuda(_MIPP_ pa->X,pa->Y,mr_mip->w3);       /* w3 = E = C*D     */

    nres_modsub_cuda(_MIPP_ mr_mip->w1,pa->X,mr_mip->w1);
    nres_modsub_cuda(_MIPP_ mr_mip->w1,pa->Y,mr_mip->w1);   /* I=(X1+Y1)(X2+Y2)-C-D =X1*Y2+Y1*X2 */

    if (mr_abs(mr_mip->Asize)==MR_TOOBIG)                   /*   */
        nres_modmult_cuda(_MIPP_ pa->Y,mr_mip->A,pa->Y);
    else
        nres_premult_cuda(_MIPP_ pa->Y,mr_mip->Asize,pa->Y);
    nres_modsub_cuda(_MIPP_ pa->X,pa->Y,pa->X);   /* X = H = C-aD */

    nres_modmult_cuda(_MIPP_ pa->Z,pa->X,pa->Z);
    nres_modmult_cuda(_MIPP_ pa->Z,mr_mip->w1,pa->Z);

    nres_modsub_cuda(_MIPP_ mr_mip->w3,mr_mip->w2,pa->Y);
    nres_modmult_cuda(_MIPP_ pa->Y,mr_mip->w1,pa->Y);

    nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w2,mr_mip->w3);
    nres_modmult_cuda(_MIPP_ pa->X,mr_mip->w3,pa->X);

    if (size_cuda(pa->Z)==0)
    {
        copy_cuda(mr_mip->one,pa->X);
        zero_cuda(pa->Y);
        pa->marker=MR_EPOINT_INFINITY;
    }
    else pa->marker=MR_EPOINT_GENERAL;

    return TRUE;      
}

__device__ void epoint_copy_cuda(epoint *a,epoint *b)
{   
    if (a==b || b==NULL) return;

    copy_cuda(a->X,b->X);
    copy_cuda(a->Y,b->Y);
    copy_cuda(a->Z,b->Z);

    b->marker=a->marker;
    return;
}

__device__ BOOL epoint_comp_cuda(_MIPD_ epoint *a,epoint *b)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;
    if (a==b) return TRUE;
    if (a->marker==MR_EPOINT_INFINITY)
    {
        if (b->marker==MR_EPOINT_INFINITY) return TRUE;
        else return FALSE;
    }
    if (b->marker==MR_EPOINT_INFINITY)
        return FALSE;
    
    MR_IN(105)
    copy_cuda(a->Z,mr_mip->w1);
    copy_cuda(b->Z,mr_mip->w2);

    nres_modmult_cuda(_MIPP_ a->X,b->Z,mr_mip->w1);
    nres_modmult_cuda(_MIPP_ b->X,a->Z,mr_mip->w2);

    if (mr_compare_cuda(mr_mip->w1,mr_mip->w2)!=0) 
    {
        MR_OUT
        return FALSE;
    }

    nres_modmult_cuda(_MIPP_ a->Y,b->Z,mr_mip->w1);
    nres_modmult_cuda(_MIPP_ b->Y,a->Z,mr_mip->w2);

    if (mr_compare_cuda(mr_mip->w1,mr_mip->w2)!=0) 
    {
        MR_OUT
        return FALSE;
    }
    MR_OUT
    return TRUE;
 
}

__device__ int ecurve_add_cuda(_MIPD_ epoint *p,epoint *pa)
{  /* pa=pa+p; */

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return MR_OVER;

    MR_IN(94)

    if (p==pa) 
    {
        ecurve_double_cuda(_MIPP_ pa);
        MR_OUT
        if (pa->marker==MR_EPOINT_INFINITY) return MR_OVER;
        return MR_DOUBLE;
    }
    if (pa->marker==MR_EPOINT_INFINITY)
    {
        epoint_copy_cuda(p,pa);
        MR_OUT 
        return MR_ADD;
    }
    if (p->marker==MR_EPOINT_INFINITY) 
    {
        MR_OUT
        return MR_ADD;
    }

    if (!ecurve_padd_cuda(_MIPP_ p,pa))
    {    
        ecurve_double_cuda(_MIPP_ pa);
        MR_OUT
        return MR_DOUBLE;
    }
    MR_OUT
    if (pa->marker==MR_EPOINT_INFINITY) return MR_OVER;
    return MR_ADD;
}

__device__ void epoint_negate_cuda(_MIPD_ epoint *p)
{ /* negate a point */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    if (p->marker==MR_EPOINT_INFINITY) return;

    MR_IN(121)
    if (size_cuda(p->X)!=0) mr_psub_cuda(_MIPP_ mr_mip->modulus,p->X,p->X);
    MR_OUT
}

__device__ int ecurve_sub_cuda(_MIPD_ epoint *p,epoint *pa)
{
    int r;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return MR_OVER;

    MR_IN(104)

    if (p==pa)
    {
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pa);
        MR_OUT
        return MR_OVER;
    } 
    if (p->marker==MR_EPOINT_INFINITY) 
    {
        MR_OUT
        return MR_ADD;
    }

    epoint_negate_cuda(_MIPP_ p);
    r=ecurve_add_cuda(_MIPP_ p,pa);
    epoint_negate_cuda(_MIPP_ p);

    MR_OUT
    return r;
}

__device__ int ecurve_mult_cuda(_MIPD_ big e,epoint *pa,epoint *pt)
{ /* pt=e*pa; */
    int i,j,n,nb,nbs,nzs,nadds;
    epoint *table[MR_ECC_STORE_N];

#ifdef MR_STATIC
    char mem[MR_ECP_RESERVE(MR_ECC_STORE_N)];  
#else
    char *mem;
#endif

#ifndef MR_ALWAYS_BINARY
    epoint *p;
    int ce,ch;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return 0;

    MR_IN(95)
    if (size_cuda(e)==0) 
    { /* multiplied by 0 */
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);
        MR_OUT
        return 0;
    }
    copy_cuda(e,mr_mip->w9);
    epoint_copy_cuda(pa,pt);

    if (size_cuda(mr_mip->w9)<0)
    { /* pt = -pt */
        negify_cuda(mr_mip->w9,mr_mip->w9);
        epoint_negate_cuda(_MIPP_ pt);
    }

    if (size_cuda(mr_mip->w9)==1)
    { 
        MR_OUT
        return 0;
    }

    premult_cuda(_MIPP_ mr_mip->w9,3,mr_mip->w10);      /* h=3*e */

#ifndef MR_STATIC
#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
#endif

#ifdef  MR_STATIC
        memset(mem,0,MR_ECP_RESERVE(MR_ECC_STORE_N));
#else
        mem=(char *)ecp_memalloc_cuda(_MIPP_ MR_ECC_STORE_N);
#endif

        for (i=0;i<=MR_ECC_STORE_N-1;i++)
            table[i]=epoint_init_mem_cuda(_MIPP_ mem,i);

        epoint_copy_cuda(pt,table[0]);
        epoint_copy_cuda(table[0],table[MR_ECC_STORE_N-1]);
        ecurve_double_cuda(_MIPP_ table[MR_ECC_STORE_N-1]);

        for (i=1;i<MR_ECC_STORE_N-1;i++)
        { /* precomputation */
            epoint_copy_cuda(table[i-1],table[i]);
            ecurve_add_cuda(_MIPP_ table[MR_ECC_STORE_N-1],table[i]);
        }
        ecurve_add_cuda(_MIPP_ table[MR_ECC_STORE_N-2],table[MR_ECC_STORE_N-1]);

        nb=logb2_cuda(_MIPP_ mr_mip->w10);
        nadds=0;
        epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);
        for (i=nb-1;i>=1;)
        { /* add_cuda/subtract_cuda */
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            n=mr_naf_window_cuda(_MIPP_ mr_mip->w9,mr_mip->w10,i,&nbs,&nzs,MR_ECC_STORE_N);
            for (j=0;j<nbs;j++)
                ecurve_double_cuda(_MIPP_ pt);
            if (n>0) {ecurve_add_cuda(_MIPP_ table[n/2],pt); nadds++;}
            if (n<0) {ecurve_sub_cuda(_MIPP_ table[(-n)/2],pt); nadds++;}
            i-=nbs;
            if (nzs)
            {
                for (j=0;j<nzs;j++) ecurve_double_cuda(_MIPP_ pt);
                i-=nzs;
            }
        }

        ecp_memkill_cuda(_MIPP_ mem,MR_ECC_STORE_N);


#ifndef MR_STATIC
#ifndef MR_ALWAYS_BINARY
    }
    else
    { 
        mem=(char *)ecp_memalloc_cuda(_MIPP_ 1);
        p=epoint_init_mem_cuda(_MIPP_ mem,0);
        epoint_copy_cuda(pt,p);

        nadds=0;
        expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w10)-1,mr_mip->w11);
        mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
        subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        while (size_cuda(mr_mip->w11) > 1)
        { /* add_cuda/subtract_cuda method */
            if (mr_mip->user!=NULL) (*mr_mip->user)();

            ecurve_double_cuda(_MIPP_ pt);
            ce=mr_compare_cuda(mr_mip->w9,mr_mip->w11); /* e(i)=1? */
            ch=mr_compare_cuda(mr_mip->w10,mr_mip->w11); /* h(i)=1? */
            if (ch>=0) 
            {  /* h(i)=1 */
                if (ce<0) {ecurve_add_cuda(_MIPP_ p,pt); nadds++;}
                mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
            }
            if (ce>=0) 
            {  /* e(i)=1 */
                if (ch<0) {ecurve_sub_cuda(_MIPP_ p,pt); nadds++;}
                mr_psub_cuda(_MIPP_ mr_mip->w9,mr_mip->w11,mr_mip->w9);  
            }
            subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        }
        ecp_memkill_cuda(_MIPP_ mem,1);
    }
#endif
#endif
    MR_OUT
    return nadds;
}

#ifndef MR_NO_ECC_MULTIADD
#ifndef MR_STATIC

__device__ void ecurve_multn_cuda(_MIPD_ int n,big *y,epoint **x,epoint *w)
{ /* pt=e[0]*p[0]+e[1]*p[1]+ .... e[n-1]*p[n-1]   */
    int i,j,k,m,nb,ea;
    epoint **G;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(114)

    m=1<<n;
    G=(epoint **)mr_alloc_cuda(_MIPP_ m,sizeof(epoint*));

    for (i=0,k=1;i<n;i++)
    {
        for (j=0; j < (1<<i) ;j++)
        {
            G[k]=epoint_init_cuda(_MIPPO_ );
            epoint_copy_cuda(x[i],G[k]);
            if (j!=0) ecurve_add_cuda(_MIPP_ G[j],G[k]);
            k++;
        }
    }

    nb=0;
    for (j=0;j<n;j++) if ((k=logb2_cuda(_MIPP_ y[j])) > nb) nb=k;

    epoint_set_cuda(_MIPP_ NULL,NULL,0,w);            /* w=0 */
    
#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        for (i=nb-1;i>=0;i--)
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ea=0;
            k=1;
            for (j=0;j<n;j++)
            {
                if (mr_testbit_cuda(_MIPP_ y[j],i)) ea+=k;
                k<<=1;
            }
            ecurve_double_cuda(_MIPP_ w);
            if (ea!=0) ecurve_add_cuda(_MIPP_ G[ea],w);
        }    
#ifndef MR_ALWAYS_BINARY
    }
    else mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
#endif

    for (i=1;i<m;i++) epoint_free_cuda(G[i]);
    mr_free_cuda(G);
    MR_OUT
}

#endif

/* PP=P+Q, PM=P-Q. */

__device__ static BOOL ecurve_add_sub_cuda(_MIPD_ epoint *P,epoint *Q,epoint *PP,epoint *PM)
{ 
 #ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
   
    if (P->marker==MR_EPOINT_NORMALIZED)
    {
        if (Q->marker==MR_EPOINT_NORMALIZED)
            copy_cuda(mr_mip->one,mr_mip->w1);
        else copy_cuda(Q->Z,mr_mip->w1);
    }
    else 
    {
        if (Q->marker==MR_EPOINT_NORMALIZED)
            copy_cuda(P->Z,mr_mip->w1);
        else nres_modmult_cuda(_MIPP_ P->Z,Q->Z,mr_mip->w1);         /* w1 = A = Z1*Z2 */
    }
    nres_modmult_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w2);      /* w2 = B = dA^2   */
    if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)                   
        nres_modmult_cuda(_MIPP_ mr_mip->w2,mr_mip->B,mr_mip->w2);
    else
        nres_premult_cuda(_MIPP_ mr_mip->w2,mr_mip->Bsize,mr_mip->w2);
    nres_modmult_cuda(_MIPP_ P->X,Q->X,mr_mip->w3);              /* w3 = C = X1*X2 */
    nres_modmult_cuda(_MIPP_ P->Y,Q->Y,mr_mip->w4);              /* w4 = D = Y1*Y2 */
    nres_modmult_cuda(_MIPP_ mr_mip->w3,mr_mip->w4,mr_mip->w5);  /* w5 = E = C*D     */
    nres_modmult_cuda(_MIPP_ P->X,Q->Y,mr_mip->w7);              /* w7 = F = X1.Y2 */
    nres_modmult_cuda(_MIPP_ Q->X,P->Y,mr_mip->w8);              /* w8 = G = X2.Y1 */

    if (mr_abs(mr_mip->Asize)==MR_TOOBIG)                   /* w4 = aD */
        nres_modmult_cuda(_MIPP_ mr_mip->w4,mr_mip->A,mr_mip->w4);
    else
        nres_premult_cuda(_MIPP_ mr_mip->w4,mr_mip->Asize,mr_mip->w4);

/* P+Q */

    nres_modsub_cuda(_MIPP_ mr_mip->w3,mr_mip->w4,mr_mip->w6);   /* w6 = H = C-aD */
    nres_modadd_cuda(_MIPP_ mr_mip->w7,mr_mip->w8,PP->Z);        /* X1*Y2+X2*Y1   */
    nres_modadd_cuda(_MIPP_ mr_mip->w5,mr_mip->w2,PP->X);
    nres_modmult_cuda(_MIPP_ PP->X,mr_mip->w6,PP->X);
    nres_modsub_cuda(_MIPP_ mr_mip->w5,mr_mip->w2,PP->Y);
    nres_modmult_cuda(_MIPP_ PP->Y,PP->Z,PP->Y);
    nres_modmult_cuda(_MIPP_ PP->Z,mr_mip->w6,PP->Z);
    nres_modmult_cuda(_MIPP_ PP->Z,mr_mip->w1,PP->Z);

    if (size_cuda(PP->Z)==0)
    {
        copy_cuda(mr_mip->one,PP->X);
        zero_cuda(PP->Y);
        PP->marker=MR_EPOINT_INFINITY;
    }
    else PP->marker=MR_EPOINT_GENERAL;

/* P-Q */

    nres_modadd_cuda(_MIPP_ mr_mip->w3,mr_mip->w4,mr_mip->w6);  /* w6 = C+aD */
    nres_modsub_cuda(_MIPP_ mr_mip->w8,mr_mip->w7,PM->Z);       /* X2*Y1-X1*Y2 */
    nres_modsub_cuda(_MIPP_ mr_mip->w5,mr_mip->w2,PM->X);
    nres_modmult_cuda(_MIPP_ PM->X,mr_mip->w6,PM->X);
    nres_modadd_cuda(_MIPP_ mr_mip->w5,mr_mip->w2,PM->Y);
    nres_modmult_cuda(_MIPP_ PM->Y,PM->Z,PM->Y);
    nres_modmult_cuda(_MIPP_ PM->Z,mr_mip->w6,PM->Z);
    nres_modmult_cuda(_MIPP_ PM->Z,mr_mip->w1,PM->Z);

    if (size_cuda(PM->Z)==0)
    {
        copy_cuda(mr_mip->one,PM->X);
        zero_cuda(PM->Y);
        PM->marker=MR_EPOINT_INFINITY;
    }
    else PM->marker=MR_EPOINT_GENERAL;

    return TRUE;
}

__device__ void ecurve_mult2_cuda(_MIPD_ big e,epoint *p,big ea,epoint *pa,epoint *pt)
{ /* pt=e*p+ea*pa; */
    int e1,h1,e2,h2,bb;
    epoint *p1,*p2,*ps[2];
#ifdef MR_STATIC
    char mem[MR_ECP_RESERVE(4)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;

    MR_IN(103)

    if (size_cuda(e)==0) 
    {
        ecurve_mult_cuda(_MIPP_ ea,pa,pt);
        MR_OUT
        return;
    }
#ifdef MR_STATIC
    memset(mem,0,MR_ECP_RESERVE(4));
#else
    mem=(char *)ecp_memalloc_cuda(_MIPP_ 4);
#endif
    p2=epoint_init_mem_cuda(_MIPP_ mem,0);
    p1=epoint_init_mem_cuda(_MIPP_ mem,1);
    ps[0]=epoint_init_mem_cuda(_MIPP_ mem,2);
    ps[1]=epoint_init_mem_cuda(_MIPP_ mem,3);

    epoint_copy_cuda(pa,p2);
    copy_cuda(ea,mr_mip->w9);
    if (size_cuda(mr_mip->w9)<0)
    { /* p2 = -p2 */
        negify_cuda(mr_mip->w9,mr_mip->w9);
        epoint_negate_cuda(_MIPP_ p2);
    }

    epoint_copy_cuda(p,p1);
    copy_cuda(e,mr_mip->w12);
    if (size_cuda(mr_mip->w12)<0)
    { /* p1= -p1 */
        negify_cuda(mr_mip->w12,mr_mip->w12);
        epoint_negate_cuda(_MIPP_ p1);
    }

    epoint_set_cuda(_MIPP_ NULL,NULL,0,pt);            /* pt=0 */ 
    ecurve_add_sub_cuda(_MIPP_ p1,p2,ps[0],ps[1]);     /* ps[0]=p1+p2, ps[1]=p1-p2 */

    mr_jsf_cuda(_MIPP_ mr_mip->w9,mr_mip->w12,mr_mip->w10,mr_mip->w9,mr_mip->w13,mr_mip->w12);
  
/*    To use a simple NAF instead, substitute this for the JSF 
        premult_cuda(_MIPP_ mr_mip->w9,3,mr_mip->w10);      3*ea  
        premult_cuda(_MIPP_ mr_mip->w12,3,mr_mip->w13);     3*e  
*/ 

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        if (mr_compare_cuda(mr_mip->w10,mr_mip->w13)>=0) bb=logb2_cuda(_MIPP_ mr_mip->w10)-1;
        else                                        bb=logb2_cuda(_MIPP_ mr_mip->w13)-1;

        while (bb>=0) /* for the simple NAF, this should be 1 */
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ecurve_double_cuda(_MIPP_ pt);

            e1=h1=e2=h2=0;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w9,bb)) e2=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w10,bb)) h2=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w12,bb)) e1=1;
            if (mr_testbit_cuda(_MIPP_ mr_mip->w13,bb)) h1=1;

            if (e1!=h1)
            {
                if (e2==h2)
                {
                    if (h1==1) ecurve_add_cuda(_MIPP_ p1,pt);
                    else       ecurve_sub_cuda(_MIPP_ p1,pt);
                }
                else
                {
                    if (h1==1)
                    {
                        if (h2==1) ecurve_add_cuda(_MIPP_ ps[0],pt);
                        else       ecurve_add_cuda(_MIPP_ ps[1],pt);
                    }
                    else
                    {
                        if (h2==1) ecurve_sub_cuda(_MIPP_ ps[1],pt);
                        else       ecurve_sub_cuda(_MIPP_ ps[0],pt);
                    }
                }
            }
            else if (e2!=h2)
            {
                if (h2==1) ecurve_add_cuda(_MIPP_ p2,pt);
                else       ecurve_sub_cuda(_MIPP_ p2,pt);
            }
            bb-=1;
        }
#ifndef MR_ALWAYS_BINARY
    }
    else
    {
         if (mr_compare_cuda(mr_mip->w10,mr_mip->w13)>=0)
              expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w10)-1,mr_mip->w11);
         else expb2_cuda(_MIPP_ logb2_cuda(_MIPP_ mr_mip->w13)-1,mr_mip->w11);

        while (size_cuda(mr_mip->w11) > 0)    /* for the NAF, this should be 1 */
        { /* add_cuda/subtract_cuda method */
            if (mr_mip->user!=NULL) (*mr_mip->user)();

            ecurve_double_cuda(_MIPP_ pt);

            e1=h1=e2=h2=0;
            if (mr_compare_cuda(mr_mip->w9,mr_mip->w11)>=0)
            { /* e1(i)=1? */
                e2=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w9,mr_mip->w11,mr_mip->w9);
            }
            if (mr_compare_cuda(mr_mip->w10,mr_mip->w11)>=0)
            { /* h1(i)=1? */
                h2=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w10,mr_mip->w11,mr_mip->w10);
            } 
            if (mr_compare_cuda(mr_mip->w12,mr_mip->w11)>=0)
            { /* e2(i)=1? */
                e1=1;   
                mr_psub_cuda(_MIPP_ mr_mip->w12,mr_mip->w11,mr_mip->w12);
            }
            if (mr_compare_cuda(mr_mip->w13,mr_mip->w11)>=0) 
            { /* h2(i)=1? */
                h1=1;  
                mr_psub_cuda(_MIPP_ mr_mip->w13,mr_mip->w11,mr_mip->w13);
            }

            if (e1!=h1)
            {
                if (e2==h2)
                {
                    if (h1==1) ecurve_add_cuda(_MIPP_ p1,pt);
                    else       ecurve_sub_cuda(_MIPP_ p1,pt);
                }
                else
                {
                    if (h1==1)
                    {
                        if (h2==1) ecurve_add_cuda(_MIPP_ ps[0],pt);
                        else       ecurve_add_cuda(_MIPP_ ps[1],pt);
                    }
                    else
                    {
                        if (h2==1) ecurve_sub_cuda(_MIPP_ ps[1],pt);
                        else       ecurve_sub_cuda(_MIPP_ ps[0],pt);
                    }
                }
            }
            else if (e2!=h2)
            {
                if (h2==1) ecurve_add_cuda(_MIPP_ p2,pt);
                else       ecurve_sub_cuda(_MIPP_ p2,pt);
            }

            subdiv_cuda(_MIPP_ mr_mip->w11,2,mr_mip->w11);
        }
    }
#endif
    ecp_memkill_cuda(_MIPP_ mem,4);
    MR_OUT
}

#endif

#endif


#endif

#ifndef mrzzn2_c
#define mrzzn2_c

#ifdef MR_COUNT_OPS
extern int fpmq,fpsq,fpaq;
#endif

__device__ void zzn2_mirvar_cuda(_MIPD_ zzn2 *w) {
    w->a = mirvar_cuda(_MIPP_ 0);
    w->b = mirvar_cuda(_MIPP_ 0);
}

__device__ void zzn2_kill_cuda(_MIPD_ zzn2 *w) {
    mirkill_cuda(w->a);
    mirkill_cuda(w->b);
}

__device__ BOOL zzn2_iszero_cuda(zzn2 *x) {
    if (size_cuda(x->a) == 0 && size_cuda(x->b) == 0) return TRUE;
    return FALSE;
}

__device__ BOOL zzn2_isunity_cuda(_MIPD_ zzn2 *x) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM || size_cuda(x->b) != 0) return FALSE;

    if (mr_compare_cuda(x->a, mr_mip->one) == 0) return TRUE;
    return FALSE;

}

__device__ BOOL zzn2_compare_cuda(zzn2 *x, zzn2 *y) {
    if (mr_compare_cuda(x->a, y->a) == 0 && mr_compare_cuda(x->b, y->b) == 0) return TRUE;
    return FALSE;
}

__device__ void zzn2_from_int_cuda(_MIPD_ int i, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(156)
    if (i == 1) {
        copy_cuda(mr_mip->one, w->a);
    } else {
        convert_cuda(_MIPP_ i, mr_mip->w1);
        nres_cuda(_MIPP_ mr_mip->w1, w->a);
    }
    zero_cuda(w->b);
    MR_OUT
}

__device__ void zzn2_from_ints_cuda(_MIPD_ int i, int j, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(168)
    convert_cuda(_MIPP_ i, mr_mip->w1);
    nres_cuda(_MIPP_ mr_mip->w1, w->a);
    convert_cuda(_MIPP_ j, mr_mip->w1);
    nres_cuda(_MIPP_ mr_mip->w1, w->b);

    MR_OUT
}

__device__ void zzn2_from_zzns_cuda(big x, big y, zzn2 *w) {
    copy_cuda(x, w->a);
    copy_cuda(y, w->b);
}

__device__ void zzn2_from_bigs_cuda(_MIPD_ big x, big y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(166)
    nres_cuda(_MIPP_ x, w->a);
    nres_cuda(_MIPP_ y, w->b);
    MR_OUT
}

__device__ void zzn2_from_zzn_cuda(big x, zzn2 *w) {
    copy_cuda(x, w->a);
    zero_cuda(w->b);
}

__device__ void zzn2_from_big_cuda(_MIPD_ big x, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(167)
    nres_cuda(_MIPP_ x, w->a);
    zero_cuda(w->b);
    MR_OUT
}

__device__ void zzn2_copy_cuda(zzn2 *x, zzn2 *w) {
    if (x == w) return;
    copy_cuda(x->a, w->a);
    copy_cuda(x->b, w->b);
}

__device__ void zzn2_zero_cuda(zzn2 *w) {
    zero_cuda(w->a);
    zero_cuda(w->b);
}

__device__ void zzn2_negate_cuda(_MIPD_ zzn2 *x, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(157)
    zzn2_copy_cuda(x, w);
    nres_negate_cuda(_MIPP_ w->a, w->a);
    nres_negate_cuda(_MIPP_ w->b, w->b);
    MR_OUT
}

__device__ void zzn2_conj_cuda(_MIPD_ zzn2 *x, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    MR_IN(158)
    if (mr_mip->ERNUM) return;
    zzn2_copy_cuda(x, w);
    nres_negate_cuda(_MIPP_ w->b, w->b);
    MR_OUT
}

__device__ void zzn2_add_cuda(_MIPD_ zzn2 *x, zzn2 *y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
#ifdef MR_COUNT_OPS
        fpaq++;
#endif
    MR_IN(159)
    nres_modadd_cuda(_MIPP_ x->a, y->a, w->a);
    nres_modadd_cuda(_MIPP_ x->b, y->b, w->b);
    MR_OUT
}

__device__ void zzn2_sadd_cuda(_MIPD_ zzn2 *x, big y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(169)
    nres_modadd_cuda(_MIPP_ x->a, y, w->a);
    MR_OUT
}

__device__ void zzn2_sub_cuda(_MIPD_ zzn2 *x, zzn2 *y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
#ifdef MR_COUNT_OPS
        fpaq++;
#endif
    MR_IN(160)
    nres_modsub_cuda(_MIPP_ x->a, y->a, w->a);
    nres_modsub_cuda(_MIPP_ x->b, y->b, w->b);
    MR_OUT
}

__device__ void zzn2_ssub_cuda(_MIPD_ zzn2 *x, big y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(170)
    nres_modsub_cuda(_MIPP_ x->a, y, w->a);
    MR_OUT
}

__device__ void zzn2_smul_cuda(_MIPD_ zzn2 *x, big y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(161)
    if (size_cuda(x->a) != 0) nres_modmult_cuda(_MIPP_ x->a, y, w->a);
    else zero_cuda(w->a);
    if (size_cuda(x->b) != 0) nres_modmult_cuda(_MIPP_ x->b, y, w->b);
    else zero_cuda(w->b);
    MR_OUT
}

__device__ void zzn2_imul_cuda(_MIPD_ zzn2 *x, int y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(152)
    if (size_cuda(x->a) != 0) nres_premult_cuda(_MIPP_ x->a, y, w->a);
    else zero_cuda(w->a);
    if (size_cuda(x->b) != 0) nres_premult_cuda(_MIPP_ x->b, y, w->b);
    else zero_cuda(w->b);
    MR_OUT
}

__device__ void zzn2_sqr_cuda(_MIPD_ zzn2 *x, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;
#ifdef MR_COUNT_OPS
        fpsq++;
#endif
    MR_IN(210)

    nres_complex_cuda(_MIPP_ x->a, x->b, w->a, w->b);

    MR_OUT
}

__device__ void zzn2_mul_cuda(_MIPD_ zzn2 *x, zzn2 *y, zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;
    if (x == y) {
        zzn2_sqr_cuda(_MIPP_ x, w);
        return;
    }
    MR_IN(162)
    /* Uses w1, w2, and w5 */

    if (zzn2_iszero_cuda(x) || zzn2_iszero_cuda(y)) zzn2_zero_cuda(w);
    else {
#ifdef MR_COUNT_OPS
        fpmq++;
#endif
#ifndef MR_NO_LAZY_REDUCTION
        if (x->a->len != 0 && x->b->len != 0 && y->a->len != 0 && y->b->len != 0)
            nres_lazy_cuda(_MIPP_ x->a, x->b, y->a, y->b, w->a, w->b);
        else {
#endif
            nres_modmult_cuda(_MIPP_ x->a, y->a, mr_mip->w1);
            nres_modmult_cuda(_MIPP_ x->b, y->b, mr_mip->w2);
            nres_modadd_cuda(_MIPP_ x->a, x->b, mr_mip->w5);
            nres_modadd_cuda(_MIPP_ y->a, y->b, w->b);
            nres_modmult_cuda(_MIPP_ w->b, mr_mip->w5, w->b);
            nres_modsub_cuda(_MIPP_ w->b, mr_mip->w1, w->b);
            nres_modsub_cuda(_MIPP_ w->b, mr_mip->w2, w->b);
            nres_modsub_cuda(_MIPP_ mr_mip->w1, mr_mip->w2, w->a);
            if (mr_mip->qnr == -2)
                nres_modsub_cuda(_MIPP_ w->a, mr_mip->w2, w->a);
#ifndef MR_NO_LAZY_REDUCTION
        }
#endif
    }
    MR_OUT
}


/*
void zzn2_print(_MIPD_ char *label, zzn2 *x)
{
    char s1[1024], s2[1024];
    big a, b;

#ifdef MR_STATIC
    char mem_big[MR_BIG_RESERVE(2)];   
 	memset(mem_big, 0, MR_BIG_RESERVE(2)); 
    a=mirvar_mem_cuda(_MIPP_ mem_big,0);
    b=mirvar_mem_cuda(_MIPP_ mem_big,1);
#else
    a = mirvar_cuda(_MIPP_  0); 
    b = mirvar_cuda(_MIPP_  0); 
#endif
    redc_cuda(_MIPP_ x->a, a); otstr(_MIPP_ a, s1);
    redc_cuda(_MIPP_ x->b, b); otstr(_MIPP_ b, s2);

    printf("%s: [%s,%s]\n", label, s1, s2);
#ifndef MR_STATIC
    mr_free_cuda(a); mr_free_cuda(b);
#endif
}

static void nres_print(_MIPD_ char *label, big x)
{
    char s[1024];
    big a;
#ifdef MR_STATIC
    char mem_big[MR_BIG_RESERVE(1)];     
 	memset(mem_big, 0, MR_BIG_RESERVE(1)); 
    a=mirvar_mem_cuda(_MIPP_ mem_big,0);
#else
    a = mirvar_cuda(_MIPP_  0); 
#endif

    redc_cuda(_MIPP_ x, a);
    otstr(_MIPP_ a, s);

    printf("%s: %s\n", label, s);
#ifndef MR_STATIC
    mr_free_cuda(a);
#endif
}

*/
__device__ void zzn2_inv_cuda(_MIPD_ zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(163)
    nres_modmult_cuda(_MIPP_ w->a, w->a, mr_mip->w1);
    nres_modmult_cuda(_MIPP_ w->b, w->b, mr_mip->w2);
    nres_modadd_cuda(_MIPP_ mr_mip->w1, mr_mip->w2, mr_mip->w1);

    if (mr_mip->qnr == -2)
        nres_modadd_cuda(_MIPP_ mr_mip->w1, mr_mip->w2, mr_mip->w1);
    redc_cuda(_MIPP_ mr_mip->w1, mr_mip->w6);

    invmodp_cuda(_MIPP_ mr_mip->w6, mr_mip->modulus, mr_mip->w6);

    nres_cuda(_MIPP_ mr_mip->w6, mr_mip->w6);

    nres_modmult_cuda(_MIPP_ w->a, mr_mip->w6, w->a);
    nres_negate_cuda(_MIPP_ mr_mip->w6, mr_mip->w6);
    nres_modmult_cuda(_MIPP_ w->b, mr_mip->w6, w->b);
    MR_OUT
}

/* divide_cuda zzn2 by 2 */

__device__ void zzn2_div2_cuda(_MIPD_ zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(173)

    nres_div2_cuda(_MIPP_ w->a, w->a);
    nres_div2_cuda(_MIPP_ w->b, w->b);

    MR_OUT
}

/* divide_cuda zzn2 by 3 */

__device__ void zzn2_div3_cuda(_MIPD_ zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(200)

    nres_div3_cuda(_MIPP_ w->a, w->a);
    nres_div3_cuda(_MIPP_ w->b, w->b);

    MR_OUT
}

/* divide_cuda zzn2 by 5 */

__device__ void zzn2_div5_cuda(_MIPD_ zzn2 *w) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(209)

    nres_div5_cuda(_MIPP_ w->a, w->a);
    nres_div5_cuda(_MIPP_ w->b, w->b);

    MR_OUT
}

/* multiply_cuda zzn2 by i */

__device__ void zzn2_timesi_cuda(_MIPD_ zzn2 *u) {
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(164)
    copy_cuda(u->a, mr_mip->w1);
    nres_negate_cuda(_MIPP_ u->b, u->a);
    if (mr_mip->qnr == -2)
        nres_modadd_cuda(_MIPP_ u->a, u->a, u->a);

    copy_cuda(mr_mip->w1, u->b);
    MR_OUT
}

__device__ void zzn2_txx_cuda(_MIPD_ zzn2 *u) {
    /* multiply_cuda w by t^2 where x^2-t is irreducible polynomial for ZZn4

     for p=5 mod 8 t=sqrt(sqrt(-2)), qnr=-2
     for p=3 mod 8 t=sqrt(1+sqrt(-1)), qnr=-1
     for p=7 mod 8 and p=2,3 mod 5 t=sqrt(2+sqrt(-1)), qnr=-1 */
    zzn2 t;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(196)

    switch (mr_mip->pmod8) {
        case 5:
            zzn2_timesi_cuda(_MIPP_ u);
            break;
        case 3:
            t.a = mr_mip->w3;
            t.b = mr_mip->w4;
            zzn2_copy_cuda(u, &t);
            zzn2_timesi_cuda(_MIPP_ u);
            zzn2_add_cuda(_MIPP_ u, &t, u);
            break;
        case 7:
            t.a = mr_mip->w3;
            t.b = mr_mip->w4;
            zzn2_copy_cuda(u, &t);
            zzn2_timesi_cuda(_MIPP_ u);
            zzn2_add_cuda(_MIPP_ u, &t, u);
            zzn2_add_cuda(_MIPP_ u, &t, u);
            break;
        default:
            break;
    }
    MR_OUT
}

__device__ void zzn2_txd_cuda(_MIPD_ zzn2 *u) { /* divide_cuda w by t^2 where x^2-t is irreducible polynomial for ZZn4
  
   for p=5 mod 8 t=sqrt(sqrt(-2)), qnr=-2
   for p=3 mod 8 t=sqrt(1+sqrt(-1)), qnr=-1
   for p=7 mod 8 and p=2,3 mod 5 t=sqrt(2+sqrt(-1)), qnr=-1 */
    zzn2 t;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(197)
    t.a = mr_mip->w3;
    t.b = mr_mip->w4;
    switch (mr_mip->pmod8) {
        case 5:
            copy_cuda(u->b, t.a);
            nres_div2_cuda(_MIPP_ u->a, t.b);
            nres_negate_cuda(_MIPP_ t.b, t.b);
            zzn2_copy_cuda(&t, u);
            break;
        case 3:
            nres_modadd_cuda(_MIPP_ u->a, u->b, t.a);
            nres_modsub_cuda(_MIPP_ u->b, u->a, t.b);
            zzn2_div2_cuda(_MIPP_ &t);
            zzn2_copy_cuda(&t, u);
            break;
        case 7:
            nres_modadd_cuda(_MIPP_ u->a, u->a, t.a);
            nres_modadd_cuda(_MIPP_ t.a, u->b, t.a);
            nres_modadd_cuda(_MIPP_ u->b, u->b, t.b);
            nres_modsub_cuda(_MIPP_ t.b, u->a, t.b);
            zzn2_div5_cuda(_MIPP_ &t);
            zzn2_copy_cuda(&t, u);
/*
        nres_modadd_cuda(_MIPP_ u->a,u->b,t.a);
        nres_modadd_cuda(_MIPP_ t.a,u->b,t.a);
        nres_modsub_cuda(_MIPP_ u->b,u->a,t.b);
        zzn2_div3(_MIPP_ &t);
        zzn2_copy_cuda(&t,u);
*/
            break;
        default:
            break;
    }

    MR_OUT
}

/* find w[i]=1/x[i] mod n, for i=0 to m-1 *
   * x and w MUST be distinct             */

__device__ BOOL zzn2_multi_inverse_cuda(_MIPD_ int m, zzn2 *x, zzn2 *w) {
    int i;
    zzn2 t1, t2;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (m == 0) return TRUE;
    if (m < 0) return FALSE;
    MR_IN(214)

    if (x == w) {
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return FALSE;
    }

    if (m == 1) {
        zzn2_copy_cuda(&x[0], &w[0]);
        zzn2_inv_cuda(_MIPP_ &w[0]);

        MR_OUT
        return TRUE;
    }

    zzn2_from_int_cuda(_MIPP_ 1, &w[0]);
    zzn2_copy_cuda(&x[0], &w[1]);

    for (i = 2; i < m; i++) {
        if (zzn2_isunity_cuda(_MIPP_ &x[i - 1]))
            zzn2_copy_cuda(&w[i - 1], &w[i]);
        else
            zzn2_mul_cuda(_MIPP_ &w[i - 1], &x[i - 1], &w[i]);
    }

    t1.a = mr_mip->w8;
    t1.b = mr_mip->w9;
    t2.a = mr_mip->w10;
    t2.b = mr_mip->w11;

    zzn2_mul_cuda(_MIPP_ &w[m - 1], &x[m - 1], &t1);
    if (zzn2_iszero_cuda(&t1)) {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }

    zzn2_inv_cuda(_MIPP_ &t1);

    zzn2_copy_cuda(&x[m - 1], &t2);
    zzn2_mul_cuda(_MIPP_ &w[m - 1], &t1, &w[m - 1]);

    for (i = m - 2;; i--) {
        if (i == 0) {
            zzn2_mul_cuda(_MIPP_ &t2, &t1, &w[0]);
            break;
        }
        zzn2_mul_cuda(_MIPP_ &w[i], &t2, &w[i]);
        zzn2_mul_cuda(_MIPP_ &w[i], &t1, &w[i]);
        if (!zzn2_isunity_cuda(_MIPP_ &x[i])) zzn2_mul_cuda(_MIPP_ &t2, &x[i], &t2);
    }

    MR_OUT
    return TRUE;
}


/*
static void zzn2_print(_MIPD_ char *label, zzn2 *x)
{
    char s1[1024], s2[1024];
    big a, b;


    a = mirvar_cuda(_MIPP_  0); 
    b = mirvar_cuda(_MIPP_  0); 

    redc_cuda(_MIPP_ x->a, a); otstr(_MIPP_ a, s1);
    redc_cuda(_MIPP_ x->b, b); otstr(_MIPP_ b, s2);

    printf("%s: [%s,%s]\n", label, s1, s2);

    mr_free_cuda(a); mr_free_cuda(b);

}

static void nres_print(_MIPD_ char *label, big x)
{
    char s[1024];
    big a;

    a = mirvar_cuda(_MIPP_  0); 

    redc_cuda(_MIPP_ x, a);
    otstr(_MIPP_ a, s);

    printf("%s: %s\n", label, s);

    mr_free_cuda(a);
}

*/

__device__ void zzn2_pow_cuda(_MIPD_ zzn2 *x, big e, zzn2 *w) {
    int i, j, nb, n, nbw, nzs;
    big ONE = mirvar_cuda(_MIPP_ 1);
    big ZERO = mirvar_cuda(_MIPP_ 0);

    zzn2 u, u2, t[16];
    zzn2_mirvar_cuda(_MIPP_ &u);
    zzn2_mirvar_cuda(_MIPP_ &u2);
    for (i = 0; i < 16; i++) zzn2_mirvar_cuda(_MIPP_ &(t[i]));

    if (zzn2_iszero_cuda(x)) {
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn2_kill_cuda(_MIPP_ &u);
        zzn2_kill_cuda(_MIPP_ &u2);
        for (i = 0; i < 16; i++) zzn2_kill_cuda(_MIPP_ &(t[i]));
        return;
    }
    if (mr_compare_cuda(e, ZERO) == 0) {
        zzn2_from_int_cuda(_MIPP_ 1, w);
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn2_kill_cuda(_MIPP_ &u);
        zzn2_kill_cuda(_MIPP_ &u2);
        for (i = 0; i < 16; i++) zzn2_kill_cuda(_MIPP_ &(t[i]));
        return;
    }
    if (mr_compare_cuda(e, ONE) == 0) {
        zzn2_copy_cuda(x, w);
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn2_kill_cuda(_MIPP_ &u);
        zzn2_kill_cuda(_MIPP_ &u2);
        for (i = 0; i < 16; i++) zzn2_kill_cuda(_MIPP_ &(t[i]));
        return;
    }

    zzn2_copy_cuda(x, &u);
    zzn2_mul_cuda(_MIPP_ &u, &u, &u2);
    zzn2_copy_cuda(&u, &(t[0]));

    for (i = 1; i < 16; i++) zzn2_mul_cuda(_MIPP_ &u2, &(t[i - 1]), &(t[i]));

    nb = logb2_cuda(_MIPP_ e);
    if (nb > 1)
        for (i = nb - 2; i >= 0;) {
            n = mr_window_cuda(_MIPP_ e, i, &nbw, &nzs, 5);
            for (j = 0; j < nbw; j++) zzn2_mul_cuda(_MIPP_ &u, &u, &u);
            if (n > 0) zzn2_mul_cuda(_MIPP_ &u, &(t[n / 2]), &u);
            i -= nbw;
            if (nzs) {
                for (j = 0; j < nzs; j++) zzn2_mul_cuda(_MIPP_ &u, &u, &u);
                i -= nzs;
            }
        }

    zzn2_copy_cuda(&u, w);

    mirkill_cuda(ONE);
    mirkill_cuda(ZERO);
    zzn2_kill_cuda(_MIPP_ &u);
    zzn2_kill_cuda(_MIPP_ &u2);
    for (i = 0; i < 16; i++) zzn2_kill_cuda(_MIPP_ &(t[i]));
}

/* Lucas-style ladder exponentiation - for ZZn4 exponentiation 

void zzn2_powl(_MIPD_ zzn2 *x,big e,zzn2 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    int i,s;
    zzn2 t1,t3,t4;
    if (mr_mip->ERNUM) return;
    MR_IN(165)
    t1.a=mr_mip->w3;
    t1.b=mr_mip->w4;
    t3.a=mr_mip->w8;
    t3.b=mr_mip->w9;
    t4.a=mr_mip->w10;
    t4.b=mr_mip->w11;

    zzn2_from_int_cuda(_MIPP_ 1,&t1);

    s=size_cuda(e);
    if (s==0)
    {
        zzn2_copy_cuda(&t1,w);
        return;
    }
    zzn2_copy_cuda(x,w);
    if (s==1 || s==(-1)) return;

    i=logb2_cuda(_MIPP_ e)-1;

    zzn2_copy_cuda(w,&t3);
    zzn2_sqr_cuda(_MIPP_ w,&t4);
    zzn2_add_cuda(_MIPP_ &t4,&t4,&t4);
    zzn2_sub_cuda(_MIPP_ &t4,&t1,&t4);

    while (i-- && !mr_mip->ERNUM)
    {
        if (mr_testbit_cuda(_MIPP_ e,i))
        {
            zzn2_mul_cuda(_MIPP_ &t3,&t4,&t3);
            zzn2_add_cuda(_MIPP_ &t3,&t3,&t3);
            zzn2_sub_cuda(_MIPP_ &t3,w,&t3);
            zzn2_sqr_cuda(_MIPP_ &t4,&t4);
            zzn2_add_cuda(_MIPP_ &t4,&t4,&t4);
            zzn2_sub_cuda(_MIPP_ &t4,&t1,&t4);
        }
        else
        {
            zzn2_mul_cuda(_MIPP_ &t4,&t3,&t4);
            zzn2_add_cuda(_MIPP_ &t4,&t4,&t4);
            zzn2_sub_cuda(_MIPP_ &t4,w,&t4);
            zzn2_sqr_cuda(_MIPP_ &t3,&t3);
            zzn2_add_cuda(_MIPP_ &t3,&t3,&t3);
            zzn2_sub_cuda(_MIPP_ &t3,&t1,&t3);
        }

    }
    zzn2_copy_cuda(&t4,w);
    MR_OUT
}
*/


#endif

#ifndef mrzzn2b_c
#define mrzzn2b_c

__device__ BOOL zzn2_qr_cuda(_MIPD_ zzn2 *u)
{
    int j;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return FALSE;
    if (zzn2_iszero_cuda(u)) return TRUE;
    if (size_cuda(u->b)==0) return TRUE;

    if (mr_mip->qnr==-1 && size_cuda(u->a)==0) return TRUE;
    

    MR_IN(203)  

    nres_modmult_cuda(_MIPP_ u->b,u->b,mr_mip->w1);
    if (mr_mip->qnr==-2) nres_modadd_cuda(_MIPP_ mr_mip->w1,mr_mip->w1,mr_mip->w1);
    nres_modmult_cuda(_MIPP_ u->a,u->a,mr_mip->w2);
    nres_modadd_cuda(_MIPP_ mr_mip->w1,mr_mip->w2,mr_mip->w1);
    redc_cuda(_MIPP_ mr_mip->w1,mr_mip->w1); 
    j=jack_cuda(_MIPP_ mr_mip->w1,mr_mip->modulus);

    MR_OUT
    if (j==1) return TRUE; 
    return FALSE; 
}

__device__ BOOL zzn2_sqrt_cuda(_MIPD_ zzn2 *u,zzn2 *w)
{ /* sqrt(a+ib) = sqrt(a+sqrt(a*a-n*b*b)/2)+ib/(2*sqrt(a+sqrt(a*a-n*b*b)/2))
     where i*i=n */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    zzn2_copy_cuda(u,w);
    if (zzn2_iszero_cuda(w)) return TRUE;

    MR_IN(204)  

    if (size_cuda(w->b)==0)
    {
        if (!nres_sqroot_cuda(_MIPP_ w->a,mr_mip->w15))
        {
            nres_negate_cuda(_MIPP_ w->a,w->b);
            zero_cuda(w->a);
            if (mr_mip->qnr==-2) nres_div2_cuda(_MIPP_ w->b,w->b); 
            nres_sqroot_cuda(_MIPP_ w->b,w->b);    
        }
        else
            copy_cuda(mr_mip->w15,w->a);

        MR_OUT
        return TRUE;
    }

    if (mr_mip->qnr==-1 && size_cuda(w->a)==0)
    {
        nres_div2_cuda(_MIPP_ w->b,w->b);
        if (nres_sqroot_cuda(_MIPP_ w->b,mr_mip->w15))
        {
            copy_cuda(mr_mip->w15,w->b);
            copy_cuda(w->b,w->a);
        }
        else
        {
            nres_negate_cuda(_MIPP_ w->b,w->b);
            nres_sqroot_cuda(_MIPP_ w->b,w->b);
            nres_negate_cuda(_MIPP_ w->b,w->a);
        }

        MR_OUT
        return TRUE;
    }

    nres_modmult_cuda(_MIPP_ w->b,w->b,mr_mip->w7);
    if (mr_mip->qnr==-2) nres_modadd_cuda(_MIPP_ mr_mip->w7,mr_mip->w7,mr_mip->w7);
    nres_modmult_cuda(_MIPP_ w->a,w->a,mr_mip->w1);
    nres_modadd_cuda(_MIPP_ mr_mip->w7,mr_mip->w1,mr_mip->w7);

    if (!nres_sqroot_cuda(_MIPP_ mr_mip->w7,mr_mip->w7)) /* s=w7 */
    {
        zzn2_zero_cuda(w);
        MR_OUT
        return FALSE;
    }

    nres_modadd_cuda(_MIPP_ w->a,mr_mip->w7,mr_mip->w15);
    nres_div2_cuda(_MIPP_ mr_mip->w15,mr_mip->w15);

    if (!nres_sqroot_cuda(_MIPP_ mr_mip->w15,mr_mip->w15))
    {

        nres_modsub_cuda(_MIPP_ w->a,mr_mip->w7,mr_mip->w15);
        nres_div2_cuda(_MIPP_ mr_mip->w15,mr_mip->w15);
        if (!nres_sqroot_cuda(_MIPP_ mr_mip->w15,mr_mip->w15))
        {
            zzn2_zero_cuda(w);
            MR_OUT
            return FALSE;
        }
    }

    copy_cuda(mr_mip->w15,w->a);
    nres_modadd_cuda(_MIPP_ mr_mip->w15,mr_mip->w15,mr_mip->w15);
    nres_moddiv_cuda(_MIPP_ w->b,mr_mip->w15,w->b);

    MR_OUT
    return TRUE;
}

/* y=1/x, z=1/w 

BOOL zzn2_double_inverse(_MIPD_ zzn2 *x,zzn2 *y,zzn2 *w,zzn2 *z)
{
    zzn2 t1,t2;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    MR_IN(214)

    t1.a=mr_mip->w8;
    t1.b=mr_mip->w9;  
    t2.a=mr_mip->w10;
    t2.b=mr_mip->w11;

    zzn2_mul_cuda(_MIPP_ x,w,&t1);
    if (zzn2_iszero_cuda(_MIPP_ &t1))
    {
        mr_berror_cuda(_MIPP_ MR_ERR_DIV_BY_ZERO);
        MR_OUT
        return FALSE;
    }
    zzn2_inv_cuda(_MIPP_ &t1);
    
    zzn2_mul_cuda(_MIPP_ &w,&t1,&t2);
    zzn2_mul_cuda(_MIPP_ &x,&t1,&z);
    zzn2_copy_cuda(&t2,&y);

    MR_OUT
    return TRUE;

}
*/



#endif

#ifndef mrecn2_c
#define mrecn2_c

#ifdef MR_STATIC
#include <string.h>
#endif

#ifndef MR_EDWARDS

__device__ void ecn2_mirvar_cuda(_MIPD_ ecn2* q){
    zzn2_mirvar_cuda(_MIPP_ &(q->x));
    zzn2_mirvar_cuda(_MIPP_ &(q->y));
    zzn2_mirvar_cuda(_MIPP_ &(q->z));
}
__device__ void ecn2_kill_cuda(_MIPD_ ecn2* q){
    zzn2_kill_cuda(_MIPP_ &(q->x));
    zzn2_kill_cuda(_MIPP_ &(q->y));
    zzn2_kill_cuda(_MIPP_ &(q->z));
}

__device__ BOOL ecn2_iszero_cuda(ecn2 *a)
{
    if (a->marker==MR_EPOINT_INFINITY) return TRUE;
    return FALSE;
}

__device__ void ecn2_copy_cuda(ecn2 *a,ecn2 *b)
{
    zzn2_copy_cuda(&(a->x),&(b->x));
    zzn2_copy_cuda(&(a->y),&(b->y));
#ifndef MR_AFFINE_ONLY
    if (a->marker==MR_EPOINT_GENERAL)  zzn2_copy_cuda(&(a->z),&(b->z));
#endif
    b->marker=a->marker;
}

__device__ void ecn2_zero_cuda(ecn2 *a)
{
    zzn2_zero_cuda(&(a->x)); zzn2_zero_cuda(&(a->y)); 
#ifndef MR_AFFINE_ONLY
    if (a->marker==MR_EPOINT_GENERAL) zzn2_zero_cuda(&(a->z));
#endif
    a->marker=MR_EPOINT_INFINITY;
}

__device__ BOOL ecn2_compare_cuda(_MIPD_ ecn2 *a,ecn2 *b)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(193)
    ecn2_norm_cuda(_MIPP_ a);
    ecn2_norm_cuda(_MIPP_ b);
    MR_OUT
    if (zzn2_compare_cuda(&(a->x),&(b->x)) && zzn2_compare_cuda(&(a->y),&(b->y)) && a->marker==b->marker) return TRUE;
    return FALSE;
}

__device__ void ecn2_norm_cuda(_MIPD_ ecn2 *a)
{
    zzn2 t;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifndef MR_AFFINE_ONLY
    if (mr_mip->ERNUM) return;
    if (a->marker!=MR_EPOINT_GENERAL) return;

    MR_IN(194)

    zzn2_inv_cuda(_MIPP_ &(a->z));

    t.a=mr_mip->w3;
    t.b=mr_mip->w4;
    zzn2_copy_cuda(&(a->z),&t);

    zzn2_sqr_cuda(_MIPP_ &(a->z),&(a->z));
    zzn2_mul_cuda(_MIPP_ &(a->x),&(a->z),&(a->x));
    zzn2_mul_cuda(_MIPP_ &(a->z),&t,&(a->z));
    zzn2_mul_cuda(_MIPP_ &(a->y),&(a->z),&(a->y));
    zzn2_from_zzn_cuda(mr_mip->one,&(a->z));
    a->marker=MR_EPOINT_NORMALIZED;

    MR_OUT
#endif
}

__device__ void ecn2_get_cuda(_MIPD_ ecn2 *e,zzn2 *x,zzn2 *y,zzn2 *z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    
    zzn2_copy_cuda(&(e->x),x);
    zzn2_copy_cuda(&(e->y),y);
#ifndef MR_AFFINE_ONLY
    if (e->marker==MR_EPOINT_GENERAL) zzn2_copy_cuda(&(e->z),z);
    else                              zzn2_from_zzn_cuda(mr_mip->one,z);
#endif
}

__device__ void ecn2_getxy_cuda(ecn2 *e,zzn2 *x,zzn2 *y)
{
    zzn2_copy_cuda(&(e->x),x);
    zzn2_copy_cuda(&(e->y),y);
}

__device__ void ecn2_getx_cuda(ecn2 *e,zzn2 *x)
{
    zzn2_copy_cuda(&(e->x),x);
}

__device__ void ecn2_psi_cuda(_MIPD_ zzn2 *psi,ecn2 *P)
{ /* apply GLS morphism to P */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(212)
    ecn2_norm_cuda(_MIPP_ P);
    zzn2_conj_cuda(_MIPP_ &(P->x),&(P->x));
    zzn2_conj_cuda(_MIPP_ &(P->y),&(P->y));
    zzn2_mul_cuda(_MIPP_ &(P->x),&psi[0],&(P->x));
    zzn2_mul_cuda(_MIPP_ &(P->y),&psi[1],&(P->y));

    MR_OUT
}

#ifndef MR_AFFINE_ONLY
__device__ void ecn2_getz_cuda(_MIPD_ ecn2 *e,zzn2 *z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (e->marker==MR_EPOINT_GENERAL) zzn2_copy_cuda(&(e->z),z);
    else                              zzn2_from_zzn_cuda(mr_mip->one,z);
}
#endif

__device__ void ecn2_rhs_cuda(_MIPD_ zzn2 *x,zzn2 *rhs)
{ /* calculate RHS of elliptic curve equation */
    int twist;
    zzn2 A,B;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    twist=mr_mip->TWIST;

    MR_IN(202)

    A.a=mr_mip->w10;
    A.b=mr_mip->w11;
    B.a=mr_mip->w12;
    B.b=mr_mip->w13;

    if (mr_abs(mr_mip->Asize)<MR_TOOBIG) zzn2_from_int_cuda(_MIPP_ mr_mip->Asize,&A);
    else zzn2_from_zzn_cuda(mr_mip->A,&A);

    if (mr_abs(mr_mip->Bsize)<MR_TOOBIG) zzn2_from_int_cuda(_MIPP_ mr_mip->Bsize,&B);
    else zzn2_from_zzn_cuda(mr_mip->B,&B);
  
    if (twist)
    { /* assume its the quartic or sextic twist, if such is possible */
		if (twist==MR_QUARTIC_M)
		{
			zzn2_mul_cuda(_MIPP_ &A,x,&B);
			zzn2_txx_cuda(_MIPP_ &B);
		}
		if (twist==MR_QUARTIC_D)
		{
			zzn2_mul_cuda(_MIPP_ &A,x,&B);
			zzn2_txd_cuda(_MIPP_ &B);
		}
		if (twist==MR_SEXTIC_M)
		{
			zzn2_txx_cuda(_MIPP_ &B);
		}
		if (twist==MR_SEXTIC_D)
		{
			zzn2_txd_cuda(_MIPP_ &B);
		}
		if (twist==MR_QUADRATIC)
		{
			zzn2_txx_cuda(_MIPP_ &B);
            zzn2_txx_cuda(_MIPP_ &B);
            zzn2_txx_cuda(_MIPP_ &B);

            zzn2_mul_cuda(_MIPP_ &A,x,&A);
            zzn2_txx_cuda(_MIPP_ &A);
            zzn2_txx_cuda(_MIPP_ &A);
            zzn2_add_cuda(_MIPP_ &B,&A,&B);

		}
/*
        if (mr_mip->Asize==0 || mr_mip->Bsize==0)
        {
            if (mr_mip->Asize==0)
            { // CM Discriminant D=3 - its the sextic twist (Hope I got the right one!). This works for BN curves 
                zzn2_txd_cuda(_MIPP_ &B);
            }
            if (mr_mip->Bsize==0)
            { // CM Discriminant D=1 - its the quartic twist. 
                zzn2_mul_cuda(_MIPP_ &A,x,&B);
				zzn2_txx_cuda(_MIPP_ &B);
            }
        }
        else
        { // its the quadratic twist 

            zzn2_txx_cuda(_MIPP_ &B);
            zzn2_txx_cuda(_MIPP_ &B);
            zzn2_txx_cuda(_MIPP_ &B);

            zzn2_mul_cuda(_MIPP_ &A,x,&A);
            zzn2_txx_cuda(_MIPP_ &A);
            zzn2_txx_cuda(_MIPP_ &A);
            zzn2_add_cuda(_MIPP_ &B,&A,&B);

        }
*/
    }
    else
    {
        zzn2_mul_cuda(_MIPP_ &A,x,&A);
        zzn2_add_cuda(_MIPP_ &B,&A,&B);
    }

    zzn2_sqr_cuda(_MIPP_ x,&A);
    zzn2_mul_cuda(_MIPP_ &A,x,&A);
    zzn2_add_cuda(_MIPP_ &B,&A,rhs);

    MR_OUT
}

__device__ BOOL ecn2_set_cuda(_MIPD_ zzn2 *x,zzn2 *y,ecn2 *e)
{
    zzn2 lhs,rhs;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(195)

    lhs.a=mr_mip->w10;
    lhs.b=mr_mip->w11;
    rhs.a=mr_mip->w12;
    rhs.b=mr_mip->w13;

    ecn2_rhs_cuda(_MIPP_ x,&rhs);

    zzn2_sqr_cuda(_MIPP_ y,&lhs);

    if (!zzn2_compare_cuda(&lhs,&rhs))
    {
        MR_OUT
        return FALSE;
    }

    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(y,&(e->y));

    e->marker=MR_EPOINT_NORMALIZED;

    MR_OUT
    return TRUE;
}

#ifndef MR_NOSUPPORT_COMPRESSION


__device__ BOOL ecn2_setx_cuda(_MIPD_ zzn2 *x,ecn2 *e)
{
    zzn2 rhs;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(201)

    rhs.a=mr_mip->w12;
    rhs.b=mr_mip->w13;

    ecn2_rhs_cuda(_MIPP_ x,&rhs);
    if (!zzn2_iszero_cuda(&rhs))
    {
		if (!zzn2_qr_cuda(_MIPP_ &rhs))
		{
            MR_OUT
            return FALSE;
		}
        zzn2_sqrt_cuda(_MIPP_ &rhs,&rhs); 
    }

    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(&rhs,&(e->y));

    e->marker=MR_EPOINT_NORMALIZED;

    MR_OUT
    return TRUE;
}

#endif

#ifndef MR_AFFINE_ONLY
__device__ void ecn2_setxyz_cuda(_MIPD_ zzn2 *x,zzn2 *y,zzn2 *z,ecn2 *e)
{
    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(y,&(e->y));
    zzn2_copy_cuda(z,&(e->z));


	if (zzn2_isunity_cuda(_MIPP_ z)) e->marker=MR_EPOINT_NORMALIZED;
    else e->marker=MR_EPOINT_GENERAL;
}
#endif

/* Normalise an array of points of length m<MR_MAX_M_T_S - requires a zzn2 workspace array of length m */

__device__ BOOL ecn2_multi_norm_cuda(_MIPD_ int m,zzn2 *work,ecn2 *p)
{ 

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
#ifndef MR_AFFINE_ONLY
    int i;
    zzn2 one,t;
    zzn2 w[MR_MAX_M_T_S];
    if (mr_mip->coord==MR_AFFINE) return TRUE;
    if (mr_mip->ERNUM) return FALSE;   
    if (m>MR_MAX_M_T_S) return FALSE;

    MR_IN(215)

    one.a=mr_mip->w12;
    one.b=mr_mip->w13;
    t.a=mr_mip->w14;
    t.b=mr_mip->w15;

    zzn2_from_int_cuda(_MIPP_ 1,&one);

    for (i=0;i<m;i++)
    {
        if (p[i].marker==MR_EPOINT_NORMALIZED) w[i]=one;
        else w[i]=p[i].z;
    }
  
    if (!zzn2_multi_inverse_cuda(_MIPP_ m,w,work)) 
    {
       MR_OUT
       return FALSE;
    }

    for (i=0;i<m;i++)
    {
        p[i].marker=MR_EPOINT_NORMALIZED;
        if (!zzn2_isunity_cuda(_MIPP_ &work[i]))
        {
            zzn2_sqr_cuda(_MIPP_ &work[i],&t);
            zzn2_mul_cuda(_MIPP_ &(p[i].x),&t,&(p[i].x));    
            zzn2_mul_cuda(_MIPP_ &t,&work[i],&t);
            zzn2_mul_cuda(_MIPP_ &(p[i].y),&t,&(p[i].y));  
        }
    }    
    MR_OUT
#endif
    return TRUE;   
}

__device__ void ecn2_negate_cuda(_MIPD_ ecn2 *u,ecn2 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    ecn2_copy_cuda(u,w);
    if (w->marker!=MR_EPOINT_INFINITY)
        zzn2_negate_cuda(_MIPP_ &(w->y),&(w->y));
}

__device__ BOOL ecn2_add2_cuda(_MIPD_ ecn2 *Q,ecn2 *P,zzn2 *lam,zzn2 *ex1)
{
    BOOL Doubling;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    Doubling=ecn2_add3_cuda(_MIPP_ Q,P,lam,ex1,NULL);

    return Doubling;
}

__device__ BOOL ecn2_add1_cuda(_MIPD_ ecn2 *Q,ecn2 *P,zzn2 *lam)
{
    BOOL Doubling;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    Doubling=ecn2_add3_cuda(_MIPP_ Q,P,lam,NULL,NULL);

    return Doubling;
}

__device__ BOOL ecn2_add_cuda(_MIPD_ ecn2 *Q,ecn2 *P)
{
    BOOL Doubling;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 lam;

    lam.a = mr_mip->w14;
    lam.b = mr_mip->w15;

    Doubling=ecn2_add3_cuda(_MIPP_ Q,P,&lam,NULL,NULL);

    return Doubling;
}

__device__ BOOL ecn2_sub_cuda(_MIPD_ ecn2 *Q,ecn2 *P)
{
    BOOL Doubling;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 lam;

    lam.a = mr_mip->w14;
    lam.b = mr_mip->w15;

    ecn2_negate_cuda(_MIPP_ Q,Q);

    Doubling=ecn2_add3_cuda(_MIPP_ Q,P,&lam,NULL,NULL);

    ecn2_negate_cuda(_MIPP_ Q,Q);

    return Doubling;
}

__device__ BOOL ecn2_add_sub_cuda(_MIPD_ ecn2 *P,ecn2 *Q,ecn2 *PP,ecn2 *PM)
{ /* PP=P+Q, PM=P-Q. Assumes P and Q are both normalized, and P!=Q */
 #ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 t1,t2,lam;

    if (mr_mip->ERNUM) return FALSE;

    if (P->marker==MR_EPOINT_GENERAL || Q->marker==MR_EPOINT_GENERAL)
    { /* Sorry, some restrictions.. */
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return FALSE;
    }

    if (zzn2_compare_cuda(&(P->x),&(Q->x)))
    { /* P=Q or P=-Q - shouldn't happen */
        ecn2_copy_cuda(P,PP);
        ecn2_add_cuda(_MIPP_ Q,PP);
        ecn2_copy_cuda(P,PM);
        ecn2_sub_cuda(_MIPP_ Q,PM);

        MR_OUT
        return TRUE;
    }

    t1.a = mr_mip->w8;
    t1.b = mr_mip->w9; 
    t2.a = mr_mip->w10; 
    t2.b = mr_mip->w11; 
    lam.a = mr_mip->w12; 
    lam.b = mr_mip->w13;    

    zzn2_copy_cuda(&(P->x),&t2);
    zzn2_sub_cuda(_MIPP_ &t2,&(Q->x),&t2);
    zzn2_inv_cuda(_MIPP_ &t2);   /* only one inverse required */
    zzn2_add_cuda(_MIPP_ &(P->x),&(Q->x),&(PP->x));
    zzn2_copy_cuda(&(PP->x),&(PM->x));

    zzn2_copy_cuda(&(P->y),&t1);
    zzn2_sub_cuda(_MIPP_ &t1,&(Q->y),&t1);
    zzn2_copy_cuda(&t1,&lam);
    zzn2_mul_cuda(_MIPP_ &lam,&t2,&lam);
    zzn2_copy_cuda(&lam,&t1);
    zzn2_sqr_cuda(_MIPP_ &t1,&t1);
    zzn2_sub_cuda(_MIPP_ &t1,&(PP->x),&(PP->x));
    zzn2_copy_cuda(&(Q->x),&(PP->y));
    zzn2_sub_cuda(_MIPP_ &(PP->y),&(PP->x),&(PP->y));
    zzn2_mul_cuda(_MIPP_ &(PP->y),&lam,&(PP->y));
    zzn2_sub_cuda(_MIPP_ &(PP->y),&(Q->y),&(PP->y));

    zzn2_copy_cuda(&(P->y),&t1);
    zzn2_add_cuda(_MIPP_ &t1,&(Q->y),&t1);
    zzn2_copy_cuda(&t1,&lam);
    zzn2_mul_cuda(_MIPP_ &lam,&t2,&lam);
    zzn2_copy_cuda(&lam,&t1);
    zzn2_sqr_cuda(_MIPP_ &t1,&t1);
    zzn2_sub_cuda(_MIPP_ &t1,&(PM->x),&(PM->x));
    zzn2_copy_cuda(&(Q->x),&(PM->y));
    zzn2_sub_cuda(_MIPP_ &(PM->y),&(PM->x),&(PM->y));
    zzn2_mul_cuda(_MIPP_ &(PM->y),&lam,&(PM->y));
    zzn2_add_cuda(_MIPP_ &(PM->y),&(Q->y),&(PM->y));

    PP->marker=MR_EPOINT_NORMALIZED;
    PM->marker=MR_EPOINT_NORMALIZED;

    return TRUE;
}

__device__ BOOL ecn2_add3_cuda(_MIPD_ ecn2 *Q,ecn2 *P,zzn2 *lam,zzn2 *ex1,zzn2 *ex2)
{ /* P+=Q */
    BOOL Doubling=FALSE;
    int twist;
    int iA;
    zzn2 t1,t2,t3;
    zzn2 Yzzz;
 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    t1.a = mr_mip->w8;
    t1.b = mr_mip->w9; 
    t2.a = mr_mip->w10; 
    t2.b = mr_mip->w11; 
    t3.a = mr_mip->w12; 
    t3.b = mr_mip->w13; 
    Yzzz.a = mr_mip->w3;
    Yzzz.b = mr_mip->w4;

    twist=mr_mip->TWIST;
    if (mr_mip->ERNUM) return FALSE;

    if (P->marker==MR_EPOINT_INFINITY)
    {
        ecn2_copy_cuda(Q,P);
        return Doubling;
    }
    if (Q->marker==MR_EPOINT_INFINITY) return Doubling;

    MR_IN(205)

    if (Q!=P && Q->marker==MR_EPOINT_GENERAL)
    { /* Sorry, this code is optimized for mixed addition only */
        mr_berror_cuda(_MIPP_ MR_ERR_BAD_PARAMETERS);
        MR_OUT
        return Doubling;
    }
#ifndef MR_AFFINE_ONLY
    if (mr_mip->coord==MR_AFFINE)
    {
#endif
        if (!zzn2_compare_cuda(&(P->x),&(Q->x)))
        {
            zzn2_copy_cuda(&(P->y),&t1);
            zzn2_sub_cuda(_MIPP_ &t1,&(Q->y),&t1);
            zzn2_copy_cuda(&(P->x),&t2);
            zzn2_sub_cuda(_MIPP_ &t2,&(Q->x),&t2);
            zzn2_copy_cuda(&t1,lam);
            zzn2_inv_cuda(_MIPP_ &t2);
            zzn2_mul_cuda(_MIPP_ lam,&t2,lam);

            zzn2_add_cuda(_MIPP_ &(P->x),&(Q->x),&(P->x));
            zzn2_copy_cuda(lam,&t1);
            zzn2_sqr_cuda(_MIPP_ &t1,&t1);
            zzn2_sub_cuda(_MIPP_ &t1,&(P->x),&(P->x));
           
            zzn2_copy_cuda(&(Q->x),&(P->y));
            zzn2_sub_cuda(_MIPP_ &(P->y),&(P->x),&(P->y));
            zzn2_mul_cuda(_MIPP_ &(P->y),lam,&(P->y));
            zzn2_sub_cuda(_MIPP_ &(P->y),&(Q->y),&(P->y));
        }
        else
        {   
            if (!zzn2_compare_cuda(&(P->y),&(Q->y)) || zzn2_iszero_cuda(&(P->y)))
            {
                ecn2_zero_cuda(P);
                zzn2_from_int_cuda(_MIPP_ 1,lam);
                MR_OUT
                return Doubling;
            }
            zzn2_copy_cuda(&(P->x),&t1);
            zzn2_copy_cuda(&(P->x),&t2);
            zzn2_copy_cuda(&(P->x),lam);
            zzn2_sqr_cuda(_MIPP_ lam,lam);
            zzn2_add_cuda(_MIPP_ lam,lam,&t3);
            zzn2_add_cuda(_MIPP_ lam,&t3,lam);

            if (mr_abs(mr_mip->Asize)<MR_TOOBIG) zzn2_from_int_cuda(_MIPP_ mr_mip->Asize,&t3);
            else zzn2_from_zzn_cuda(mr_mip->A,&t3);
        
            if (twist)
            {
				if (twist==MR_QUARTIC_M)
				{
					zzn2_txx_cuda(_MIPP_ &t3);
				}
				if (twist==MR_QUARTIC_D)
				{
					zzn2_txd_cuda(_MIPP_ &t3);
				}
				if (twist==MR_QUADRATIC)
				{
					zzn2_txx_cuda(_MIPP_ &t3);
					zzn2_txx_cuda(_MIPP_ &t3);
				}
/*
				if (mr_mip->Bsize==0)
				{ // assume its the quartic twist 
					zzn2_txx_cuda(_MIPP_ &t3);
				}
				else
				{
					zzn2_txx_cuda(_MIPP_ &t3);
					zzn2_txx_cuda(_MIPP_ &t3);
				}
*/
            }
            zzn2_add_cuda(_MIPP_ lam,&t3,lam);
            zzn2_add_cuda(_MIPP_ &(P->y),&(P->y),&t3);
            zzn2_inv_cuda(_MIPP_ &t3);
            zzn2_mul_cuda(_MIPP_ lam,&t3,lam);

            zzn2_add_cuda(_MIPP_ &t2,&(P->x),&t2);
            zzn2_copy_cuda(lam,&(P->x));
            zzn2_sqr_cuda(_MIPP_ &(P->x),&(P->x));
            zzn2_sub_cuda(_MIPP_ &(P->x),&t2,&(P->x));
            zzn2_sub_cuda(_MIPP_ &t1,&(P->x),&t1);
            zzn2_mul_cuda(_MIPP_ &t1,lam,&t1);
            zzn2_sub_cuda(_MIPP_ &t1,&(P->y),&(P->y));
        }

        P->marker=MR_EPOINT_NORMALIZED;
        MR_OUT
        return Doubling;
#ifndef MR_AFFINE_ONLY
    }

    if (Q==P) Doubling=TRUE;

    zzn2_copy_cuda(&(Q->x),&t3);
    zzn2_copy_cuda(&(Q->y),&Yzzz);

    if (!Doubling)
    {
        if (P->marker!=MR_EPOINT_NORMALIZED)
        {
            zzn2_sqr_cuda(_MIPP_ &(P->z),&t1); /* 1S */
            zzn2_mul_cuda(_MIPP_ &t3,&t1,&t3);         /* 1M */
            zzn2_mul_cuda(_MIPP_ &t1,&(P->z),&t1);     /* 1M */
            zzn2_mul_cuda(_MIPP_ &Yzzz,&t1,&Yzzz);     /* 1M */
        }
        if (zzn2_compare_cuda(&t3,&(P->x)))
        {
            if (!zzn2_compare_cuda(&Yzzz,&(P->y)) || zzn2_iszero_cuda(&(P->y)))
            {
                ecn2_zero_cuda(P);
                zzn2_from_int_cuda(_MIPP_ 1,lam);
                MR_OUT
                return Doubling;
            }
            else Doubling=TRUE;
        }
    }
    if (!Doubling)
    { /* Addition */
        zzn2_sub_cuda(_MIPP_ &t3,&(P->x),&t3);
        zzn2_sub_cuda(_MIPP_ &Yzzz,&(P->y),lam);
        if (P->marker==MR_EPOINT_NORMALIZED) zzn2_copy_cuda(&t3,&(P->z));
        else zzn2_mul_cuda(_MIPP_ &(P->z),&t3,&(P->z)); /* 1M */
        zzn2_sqr_cuda(_MIPP_ &t3,&t1);                  /* 1S */
        zzn2_mul_cuda(_MIPP_ &t1,&t3,&Yzzz);            /* 1M */
        zzn2_mul_cuda(_MIPP_ &t1,&(P->x),&t1);          /* 1M */
        zzn2_copy_cuda(&t1,&t3);
        zzn2_add_cuda(_MIPP_ &t3,&t3,&t3);
        zzn2_sqr_cuda(_MIPP_ lam,&(P->x));              /* 1S */
        zzn2_sub_cuda(_MIPP_ &(P->x),&t3,&(P->x));
        zzn2_sub_cuda(_MIPP_ &(P->x),&Yzzz,&(P->x));
        zzn2_sub_cuda(_MIPP_ &t1,&(P->x),&t1);
        zzn2_mul_cuda(_MIPP_ &t1,lam,&t1);              /* 1M */
        zzn2_mul_cuda(_MIPP_ &Yzzz,&(P->y),&Yzzz);      /* 1M */
        zzn2_sub_cuda(_MIPP_ &t1,&Yzzz,&(P->y));

/*
        zzn2_sub_cuda(_MIPP_ &(P->x),&t3,&t1);     
        zzn2_sub_cuda(_MIPP_ &(P->y),&Yzzz,lam); 
        if (P->marker==MR_EPOINT_NORMALIZED) zzn2_copy_cuda(&t1,&(P->z));
        else zzn2_mul_cuda(_MIPP_ &(P->z),&t1,&(P->z)); 
        zzn2_sqr_cuda(_MIPP_ &t1,&t2);             
        zzn2_add_cuda(_MIPP_ &(P->x),&t3,&t3);     
        zzn2_mul_cuda(_MIPP_ &t3,&t2,&t3);         
        zzn2_sqr_cuda(_MIPP_ lam,&(P->x));        
        zzn2_sub_cuda(_MIPP_ &(P->x),&t3,&(P->x));

        zzn2_mul_cuda(_MIPP_ &t2,&t1,&t2);         
        zzn2_add_cuda(_MIPP_ &(P->x),&(P->x),&t1);
        zzn2_sub_cuda(_MIPP_ &t3,&t1,&t3);
        zzn2_mul_cuda(_MIPP_ &t3,lam,&t3);         

        zzn2_add_cuda(_MIPP_ &(P->y),&Yzzz,&t1);

        zzn2_mul_cuda(_MIPP_ &t2,&t1,&t2);         
        zzn2_sub_cuda(_MIPP_ &t3,&t2,&(P->y));
        zzn2_div2_cuda(_MIPP_ &(P->y));
*/
    }
    else
    { /* doubling */
        zzn2_sqr_cuda(_MIPP_ &(P->y),&t3);  /* 1S */

        iA=mr_mip->Asize;
        if (iA!=0)
        {
            if (P->marker==MR_EPOINT_NORMALIZED) zzn2_from_int_cuda(_MIPP_ 1,&t1);
            else zzn2_sqr_cuda(_MIPP_ &(P->z),&t1);  /* 1S */
            if (ex2!=NULL) zzn2_copy_cuda(&t1,ex2);

            if (iA==-3 && twist<=MR_QUADRATIC)
            {
                if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &t1); /* quadratic twist */
                zzn2_sub_cuda(_MIPP_ &(P->x),&t1,lam);
                zzn2_add_cuda(_MIPP_ &t1,&(P->x),&t1);
                zzn2_mul_cuda(_MIPP_ lam,&t1,lam);        /* 1M */
                zzn2_add_cuda(_MIPP_ lam,lam,&t2);
                zzn2_add_cuda(_MIPP_ lam,&t2,lam);
            }
            else
            {
                zzn2_sqr_cuda(_MIPP_ &(P->x),lam);  /* 1S */
                zzn2_add_cuda(_MIPP_ lam,lam,&t2);         
                zzn2_add_cuda(_MIPP_ lam,&t2,lam);      
          
                if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &t1);    /* quadratic twist */
                zzn2_sqr_cuda(_MIPP_ &t1,&t1);          /* 1S */ 
				if (twist==MR_QUARTIC_M) zzn2_txx_cuda(_MIPP_ &t1);    /* quartic twist */ 
				if (twist==MR_QUARTIC_D) zzn2_txd_cuda(_MIPP_ &t1);    /* quartic twist */ 
                if (iA!=1)
                { /* optimized for iA=1 case */
                    if (iA<MR_TOOBIG) zzn2_imul_cuda(_MIPP_ &t1,iA,&t1);
                    else zzn2_smul_cuda(_MIPP_ &t1,mr_mip->A,&t1);
                }
                zzn2_add_cuda(_MIPP_ lam,&t1,lam);
            }
        }
        else
        {
            zzn2_sqr_cuda(_MIPP_ &(P->x),lam);  /* 1S */
            zzn2_add_cuda(_MIPP_ lam,lam,&t2);
            zzn2_add_cuda(_MIPP_ lam,&t2,lam);
        }
        zzn2_mul_cuda(_MIPP_ &(P->x),&t3,&t1);    /* 1M */
        zzn2_add_cuda(_MIPP_ &t1,&t1,&t1);
        zzn2_add_cuda(_MIPP_ &t1,&t1,&t1);
        zzn2_sqr_cuda(_MIPP_ lam,&(P->x));        /* 1S */
        zzn2_add_cuda(_MIPP_ &t1,&t1,&t2);
        zzn2_sub_cuda(_MIPP_ &(P->x),&t2,&(P->x));
        if (P->marker==MR_EPOINT_NORMALIZED) zzn2_copy_cuda(&(P->y),&(P->z));
        else zzn2_mul_cuda(_MIPP_ &(P->z),&(P->y),&(P->z));   /* 1M */
        zzn2_add_cuda(_MIPP_ &(P->z),&(P->z),&(P->z));
        zzn2_add_cuda(_MIPP_ &t3,&t3,&t3);
        if (ex1!=NULL) zzn2_copy_cuda(&t3,ex1);
        zzn2_sqr_cuda(_MIPP_ &t3,&t3);                  /* 1S */
        zzn2_add_cuda(_MIPP_ &t3,&t3,&t3);
        zzn2_sub_cuda(_MIPP_ &t1,&(P->x),&t1);
        zzn2_mul_cuda(_MIPP_ lam,&t1,&(P->y));          /* 1M */  
        zzn2_sub_cuda(_MIPP_ &(P->y),&t3,&(P->y));
    }

    P->marker=MR_EPOINT_GENERAL;
    MR_OUT
    return Doubling;
#endif
}

/* Dahmen, Okeya and Schepers "Affine Precomputation with Sole Inversion in Elliptic Curve Cryptography" */
/* Precomputes table into T. Assumes first P has been copied to P[0], then calculates 3P, 5P, 7P etc. into T */

#define MR_PRE_2 (14+4*MR_ECC_STORE_N2)

__device__ static void ecn2_pre_cuda(_MIPD_ int sz,BOOL norm,ecn2 *PT)
{
    int twist;
    int i,j;
    zzn2 A,B,C,D,E,T,W;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    zzn2 *d=(zzn2 *)mr_alloc_cuda(_MIPP_ sz,sizeof(zzn2));
    zzn2 *e=(zzn2 *)mr_alloc_cuda(_MIPP_ sz,sizeof(zzn2));
    char *mem = (char *)memalloc_cuda(_MIPP_ 14+4*sz);
#else
    zzn2 d[MR_ECC_STORE_N2],e[MR_ECC_STORE_N2];
    char mem[MR_BIG_RESERVE(MR_PRE_2)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_PRE_2));   
#endif

    twist=mr_mip->TWIST;
    j=0;

    A.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    A.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    B.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    B.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    C.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    C.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    D.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    D.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    E.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    E.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    T.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    T.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    W.a= mirvar_mem_cuda(_MIPP_ mem, j++);
    W.b= mirvar_mem_cuda(_MIPP_ mem, j++);

    for (i=0;i<sz;i++)
    {
        d[i].a= mirvar_mem_cuda(_MIPP_ mem, j++);
        d[i].b= mirvar_mem_cuda(_MIPP_ mem, j++);
        e[i].a= mirvar_mem_cuda(_MIPP_ mem, j++);
        e[i].b= mirvar_mem_cuda(_MIPP_ mem, j++);
    }

    zzn2_add_cuda(_MIPP_ &(PT[0].y),&(PT[0].y),&d[0]);   /* 1. d_0=2.y */
    zzn2_sqr_cuda(_MIPP_ &d[0],&C);                      /* 2. C=d_0^2 */

    zzn2_sqr_cuda(_MIPP_ &(PT[0].x),&T);
    zzn2_add_cuda(_MIPP_ &T,&T,&A);
    zzn2_add_cuda(_MIPP_ &T,&A,&T);
           
    if (mr_abs(mr_mip->Asize)<MR_TOOBIG) zzn2_from_int_cuda(_MIPP_ mr_mip->Asize,&A);
    else zzn2_from_zzn_cuda(mr_mip->A,&A);
        
    if (twist)
    {
		if (twist==MR_QUARTIC_M)
		{
			zzn2_txx_cuda(_MIPP_ &A);
		}
		if (twist==MR_QUARTIC_D)
		{
			zzn2_txd_cuda(_MIPP_ &A);
		}
		if (twist==MR_QUADRATIC)
		{
			zzn2_txx_cuda(_MIPP_ &A);
			zzn2_txx_cuda(_MIPP_ &A);
		}
/*
		if (mr_mip->Bsize==0)
		{ // assume its the quartic twist 
			zzn2_txx_cuda(_MIPP_ &A);
		}
		else
		{
			zzn2_txx_cuda(_MIPP_ &A);
			zzn2_txx_cuda(_MIPP_ &A);
		}
*/
    }
    zzn2_add_cuda(_MIPP_ &A,&T,&A);             /* 3. A=3x^2+a */
    zzn2_copy_cuda(&A,&W);

    zzn2_add_cuda(_MIPP_ &C,&C,&B);
    zzn2_add_cuda(_MIPP_ &B,&C,&B);
    zzn2_mul_cuda(_MIPP_ &B,&(PT[0].x),&B);     /* 4. B=3C.x */

    zzn2_sqr_cuda(_MIPP_ &A,&d[1]);
    zzn2_sub_cuda(_MIPP_ &d[1],&B,&d[1]);       /* 5. d_1=A^2-B */

    zzn2_sqr_cuda(_MIPP_ &d[1],&E);             /* 6. E=d_1^2 */
    
    zzn2_mul_cuda(_MIPP_ &B,&E,&B);             /* 7. B=E.B */

    zzn2_sqr_cuda(_MIPP_ &C,&C);                /* 8. C=C^2 */

    zzn2_mul_cuda(_MIPP_ &E,&d[1],&D);          /* 9. D=E.d_1 */

    zzn2_mul_cuda(_MIPP_ &A,&d[1],&A);
    zzn2_add_cuda(_MIPP_ &A,&C,&A);
    zzn2_negate_cuda(_MIPP_ &A,&A);             /* 10. A=-d_1*A-C */

    zzn2_add_cuda(_MIPP_ &D,&D,&T);
    zzn2_sqr_cuda(_MIPP_ &A,&d[2]);
    zzn2_sub_cuda(_MIPP_ &d[2],&T,&d[2]);
    zzn2_sub_cuda(_MIPP_ &d[2],&B,&d[2]);       /* 11. d_2=A^2-2D-B */

    if (sz>3)
    {
        zzn2_sqr_cuda(_MIPP_ &d[2],&E);             /* 12. E=d_2^2 */

        zzn2_add_cuda(_MIPP_ &T,&D,&T);
        zzn2_add_cuda(_MIPP_ &T,&B,&T);
        zzn2_mul_cuda(_MIPP_ &T,&E,&B);             /* 13. B=E(B+3D) */
        
        zzn2_add_cuda(_MIPP_ &A,&A,&T);
        zzn2_add_cuda(_MIPP_ &C,&T,&C);
        zzn2_mul_cuda(_MIPP_ &C,&D,&C);             /* 14. C=D(2A+C) */

        zzn2_mul_cuda(_MIPP_ &d[2],&E,&D);          /* 15. D=E.d_2 */

        zzn2_mul_cuda(_MIPP_ &A,&d[2],&A);
        zzn2_add_cuda(_MIPP_ &A,&C,&A);
        zzn2_negate_cuda(_MIPP_ &A,&A);             /* 16. A=-d_2*A-C */

 
        zzn2_sqr_cuda(_MIPP_ &A,&d[3]);
        zzn2_sub_cuda(_MIPP_ &d[3],&D,&d[3]);
        zzn2_sub_cuda(_MIPP_ &d[3],&B,&d[3]);       /* 17. d_3=A^2-D-B */

        for (i=4;i<sz;i++)
        {
            zzn2_sqr_cuda(_MIPP_ &d[i-1],&E);       /* 19. E=d(i-1)^2 */
            zzn2_mul_cuda(_MIPP_ &B,&E,&B);         /* 20. B=E.B */
            zzn2_mul_cuda(_MIPP_ &C,&D,&C);         /* 21. C=D.C */
            zzn2_mul_cuda(_MIPP_ &E,&d[i-1],&D);    /* 22. D=E.d(i-1) */

            zzn2_mul_cuda(_MIPP_ &A,&d[i-1],&A);
            zzn2_add_cuda(_MIPP_ &A,&C,&A);
            zzn2_negate_cuda(_MIPP_ &A,&A);         /* 23. A=-d(i-1)*A-C */

            zzn2_sqr_cuda(_MIPP_ &A,&d[i]);
            zzn2_sub_cuda(_MIPP_ &d[i],&D,&d[i]);
            zzn2_sub_cuda(_MIPP_ &d[i],&B,&d[i]);   /* 24. d(i)=A^2-D-B */
        }
    }

    zzn2_copy_cuda(&d[0],&e[0]);
    for (i=1;i<sz;i++)
        zzn2_mul_cuda(_MIPP_ &e[i-1],&d[i],&e[i]);
       
    zzn2_copy_cuda(&e[sz-1],&A);
    zzn2_inv_cuda(_MIPP_ &A);

    for (i=sz-1;i>0;i--)
    {
        zzn2_copy_cuda(&d[i],&B);
        zzn2_mul_cuda(_MIPP_ &e[i-1],&A,&d[i]);  
        zzn2_mul_cuda(_MIPP_ &A,&B,&A);
    }
    zzn2_copy_cuda(&A,&d[0]);

    for (i=1;i<sz;i++)
    {
        zzn2_sqr_cuda(_MIPP_ &e[i-1],&T);
        zzn2_mul_cuda(_MIPP_ &d[i],&T,&d[i]); /** */
    }

    zzn2_mul_cuda(_MIPP_ &W,&d[0],&W);
    zzn2_sqr_cuda(_MIPP_ &W,&A);
    zzn2_sub_cuda(_MIPP_ &A,&(PT[0].x),&A);
    zzn2_sub_cuda(_MIPP_ &A,&(PT[0].x),&A);
    zzn2_sub_cuda(_MIPP_ &(PT[0].x),&A,&B);
    zzn2_mul_cuda(_MIPP_ &B,&W,&B);
    zzn2_sub_cuda(_MIPP_ &B,&(PT[0].y),&B);

    zzn2_sub_cuda(_MIPP_ &B,&(PT[0].y),&T);
    zzn2_mul_cuda(_MIPP_ &T,&d[1],&T);

    zzn2_sqr_cuda(_MIPP_ &T,&(PT[1].x));
    zzn2_sub_cuda(_MIPP_ &(PT[1].x),&A,&(PT[1].x));
    zzn2_sub_cuda(_MIPP_ &(PT[1].x),&(PT[0].x),&(PT[1].x));

    zzn2_sub_cuda(_MIPP_ &A,&(PT[1].x),&(PT[1].y));
    zzn2_mul_cuda(_MIPP_ &(PT[1].y),&T,&(PT[1].y));
    zzn2_sub_cuda(_MIPP_ &(PT[1].y),&B,&(PT[1].y));

    for (i=2;i<sz;i++)
    {
        zzn2_sub_cuda(_MIPP_ &(PT[i-1].y),&B,&T);
        zzn2_mul_cuda(_MIPP_ &T,&d[i],&T);

        zzn2_sqr_cuda(_MIPP_ &T,&(PT[i].x));
        zzn2_sub_cuda(_MIPP_ &(PT[i].x),&A,&(PT[i].x));
        zzn2_sub_cuda(_MIPP_ &(PT[i].x),&(PT[i-1].x),&(PT[i].x));

        zzn2_sub_cuda(_MIPP_ &A,&(PT[i].x),&(PT[i].y));
        zzn2_mul_cuda(_MIPP_ &(PT[i].y),&T,&(PT[i].y));
        zzn2_sub_cuda(_MIPP_ &(PT[i].y),&B,&(PT[i].y));
    }
    for (i=0;i<sz;i++) PT[i].marker=MR_EPOINT_NORMALIZED;

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, 14+4*sz);
    mr_free_cuda(d); mr_free_cuda(e);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_PRE_2));
#endif
}

#ifndef MR_DOUBLE_BIG
#define MR_MUL_RESERVE (1+4*MR_ECC_STORE_N2)
#else
#define MR_MUL_RESERVE (2+4*MR_ECC_STORE_N2)
#endif

__device__ int ecn2_mul_cuda(_MIPD_ big k,ecn2 *P)
{
    int i,j,nb,n,nbs,nzs,nadds;
	BOOL neg;
    big h;
    ecn2 T[MR_ECC_STORE_N2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = (char *)memalloc_cuda(_MIPP_ MR_MUL_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL_RESERVE)];
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL_RESERVE));
#endif

    j=0;
#ifndef MR_DOUBLE_BIG
    h=mirvar_mem_cuda(_MIPP_ mem, j++);
#else
    h=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
#endif
    for (i=0;i<MR_ECC_STORE_N2;i++)
    {
        T[i].x.a= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].x.b= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].y.a= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].y.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    }

    MR_IN(207)

    ecn2_norm_cuda(_MIPP_ P);

	nadds=0;
  
	neg=FALSE;
	if (size_cuda(k)<0)
	{
		negify_cuda(k,k);
		ecn2_negate_cuda(_MIPP_ P,&T[0]);
		neg=TRUE;
	}
	else ecn2_copy_cuda(P,&T[0]);
		
	premult_cuda(_MIPP_ k,3,h);
    
	nb=logb2_cuda(_MIPP_ h);
    ecn2_pre_cuda(_MIPP_ MR_ECC_STORE_N2,TRUE,T);

    ecn2_zero_cuda(P);

    for (i=nb-1;i>=1;)
    {
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        n=mr_naf_window_cuda(_MIPP_ k,h,i,&nbs,&nzs,MR_ECC_STORE_N2);
 
        for (j=0;j<nbs;j++) ecn2_add_cuda(_MIPP_ P,P);
       
        if (n>0) {nadds++; ecn2_add_cuda(_MIPP_ &T[n/2],P);}
        if (n<0) {nadds++; ecn2_sub_cuda(_MIPP_ &T[(-n)/2],P);}
        i-=nbs;
        if (nzs)
        {
            for (j=0;j<nzs;j++) ecn2_add_cuda(_MIPP_ P,P);
            i-=nzs;
        }
    }
	if (neg) negify_cuda(k,k);

    ecn2_norm_cuda(_MIPP_ P);
    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL_RESERVE));
#endif
	return nadds;
}

/* Double addition, using Joint Sparse Form */
/* R=aP+bQ */

#ifndef MR_NO_ECC_MULTIADD

#define MR_MUL2_JSF_RESERVE 20

__device__ int ecn2_mul2_jsf_cuda(_MIPD_ big a,ecn2 *P,big b,ecn2 *Q,ecn2 *R)
{
    int e1,h1,e2,h2,bb,nadds;
    ecn2 P1,P2,PS,PD;
    big c,d,e,f;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = (char *)memalloc_cuda(_MIPP_ MR_MUL2_JSF_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE)];
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE));
#endif

    c = mirvar_mem_cuda(_MIPP_ mem, 0);
    d = mirvar_mem_cuda(_MIPP_ mem, 1);
    e = mirvar_mem_cuda(_MIPP_ mem, 2);
    f = mirvar_mem_cuda(_MIPP_ mem, 3);
    P1.x.a= mirvar_mem_cuda(_MIPP_ mem, 4);
    P1.x.b= mirvar_mem_cuda(_MIPP_ mem, 5);
    P1.y.a= mirvar_mem_cuda(_MIPP_ mem, 6);
    P1.y.b= mirvar_mem_cuda(_MIPP_ mem, 7);
    P2.x.a= mirvar_mem_cuda(_MIPP_ mem, 8);
    P2.x.b= mirvar_mem_cuda(_MIPP_ mem, 9);
    P2.y.a= mirvar_mem_cuda(_MIPP_ mem, 10);
    P2.y.b= mirvar_mem_cuda(_MIPP_ mem, 11);
    PS.x.a= mirvar_mem_cuda(_MIPP_ mem, 12);
    PS.x.b= mirvar_mem_cuda(_MIPP_ mem, 13);
    PS.y.a= mirvar_mem_cuda(_MIPP_ mem, 14);
    PS.y.b= mirvar_mem_cuda(_MIPP_ mem, 15);
    PD.x.a= mirvar_mem_cuda(_MIPP_ mem, 16);
    PD.x.b= mirvar_mem_cuda(_MIPP_ mem, 17);
    PD.y.a= mirvar_mem_cuda(_MIPP_ mem, 18);
    PD.y.b= mirvar_mem_cuda(_MIPP_ mem, 19);

    MR_IN(206)

    ecn2_norm_cuda(_MIPP_ Q); 
    ecn2_copy_cuda(Q,&P2); 

    copy_cuda(b,d);
    if (size_cuda(d)<0) 
    {
        negify_cuda(d,d);
        ecn2_negate_cuda(_MIPP_ &P2,&P2);
    }

    ecn2_norm_cuda(_MIPP_ P); 
    ecn2_copy_cuda(P,&P1); 

    copy_cuda(a,c);
    if (size_cuda(c)<0) 
    {
        negify_cuda(c,c);
        ecn2_negate_cuda(_MIPP_ &P1,&P1);
    }

    mr_jsf_cuda(_MIPP_ d,c,e,d,f,c);   /* calculate joint sparse form */
 
    if (mr_compare_cuda(e,f)>0) bb=logb2_cuda(_MIPP_ e)-1;
    else                   bb=logb2_cuda(_MIPP_ f)-1;

    ecn2_add_sub_cuda(_MIPP_ &P1,&P2,&PS,&PD);
    ecn2_zero_cuda(R);
	nadds=0;
   
    while (bb>=0) 
    { /* add_cuda/subtract_cuda method */
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        ecn2_add_cuda(_MIPP_ R,R);
        e1=h1=e2=h2=0;

        if (mr_testbit_cuda(_MIPP_ d,bb)) e2=1;
        if (mr_testbit_cuda(_MIPP_ e,bb)) h2=1;
        if (mr_testbit_cuda(_MIPP_ c,bb)) e1=1;
        if (mr_testbit_cuda(_MIPP_ f,bb)) h1=1;

        if (e1!=h1)
        {
            if (e2==h2)
            {
                if (h1==1) {ecn2_add_cuda(_MIPP_ &P1,R); nadds++;}
                else       {ecn2_sub_cuda(_MIPP_ &P1,R); nadds++;}
            }
            else
            {
                if (h1==1)
                {
                    if (h2==1) {ecn2_add_cuda(_MIPP_ &PS,R); nadds++;}
                    else       {ecn2_add_cuda(_MIPP_ &PD,R); nadds++;}
                }
                else
                {
                    if (h2==1) {ecn2_sub_cuda(_MIPP_ &PD,R); nadds++;}
                    else       {ecn2_sub_cuda(_MIPP_ &PS,R); nadds++;}
                }
            }
        }
        else if (e2!=h2)
        {
            if (h2==1) {ecn2_add_cuda(_MIPP_ &P2,R); nadds++;}
            else       {ecn2_sub_cuda(_MIPP_ &P2,R); nadds++;}
        }
        bb-=1;
    }
    ecn2_norm_cuda(_MIPP_ R); 

    MR_OUT
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_JSF_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE));
#endif
	return nadds;

}

/* General purpose multi-exponentiation engine, using inter-leaving algorithm. Calculate aP+bQ+cR+dS...
   Inputs are divided into two groups of sizes wa<4 and wb<4. For the first group if the points are fixed the 
   first precomputed Table Ta[] may be taken from ROM. For the second group if the points are variable Tb[j] will
   have to computed online. Each group has its own precomputed store size_cuda, sza (=8?) and szb (=20?) respectively. 
   The values a,b,c.. are provided in ma[] and mb[], and 3.a,3.b,3.c (as required by the NAF) are provided in 
   ma3[] and mb3[]. If only one group is required, set wb=0 and pass NULL pointers.
   */

__device__ int ecn2_muln_engine_cuda(_MIPD_ int wa,int sza,int wb,int szb,big *ma,big *ma3,big *mb,big *mb3,ecn2 *Ta,ecn2 *Tb,ecn2 *R)
{ /* general purpose interleaving algorithm engine for multi-exp */
    int i,j,tba[4],pba[4],na[4],sa[4],tbb[4],pbb[4],nb[4],sb[4],nbits,nbs,nzs;
    int nadds;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    ecn2_zero_cuda(R);

    nbits=0;
    for (i=0;i<wa;i++) {sa[i]=exsign_cuda(ma[i]); tba[i]=0; j=logb2_cuda(_MIPP_ ma3[i]); if (j>nbits) nbits=j; }
    for (i=0;i<wb;i++) {sb[i]=exsign_cuda(mb[i]); tbb[i]=0; j=logb2_cuda(_MIPP_ mb3[i]); if (j>nbits) nbits=j; }
    
    nadds=0;
    for (i=nbits-1;i>=1;i--)
    {
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        if (R->marker!=MR_EPOINT_INFINITY) ecn2_add_cuda(_MIPP_ R,R);
        for (j=0;j<wa;j++)
        { /* deal with the first group */
            if (tba[j]==0)
            {
                na[j]=mr_naf_window_cuda(_MIPP_ ma[j],ma3[j],i,&nbs,&nzs,sza);
                tba[j]=nbs+nzs;
                pba[j]=nbs;
            }
            tba[j]--;  pba[j]--; 
            if (pba[j]==0)
            {
                if (sa[j]==PLUS)
                {
                    if (na[j]>0) {ecn2_add_cuda(_MIPP_ &Ta[j*sza+na[j]/2],R); nadds++;}
                    if (na[j]<0) {ecn2_sub_cuda(_MIPP_ &Ta[j*sza+(-na[j])/2],R); nadds++;}
                }
                else
                {
                    if (na[j]>0) {ecn2_sub_cuda(_MIPP_ &Ta[j*sza+na[j]/2],R); nadds++;}
                    if (na[j]<0) {ecn2_add_cuda(_MIPP_ &Ta[j*sza+(-na[j])/2],R); nadds++;}
                }
            }         
        }
        for (j=0;j<wb;j++)
        { /* deal with the second group */
            if (tbb[j]==0)
            {
                nb[j]=mr_naf_window_cuda(_MIPP_ mb[j],mb3[j],i,&nbs,&nzs,szb);
                tbb[j]=nbs+nzs;
                pbb[j]=nbs;
            }
            tbb[j]--;  pbb[j]--; 
            if (pbb[j]==0)
            {
                if (sb[j]==PLUS)
                {
                    if (nb[j]>0) {ecn2_add_cuda(_MIPP_ &Tb[j*szb+nb[j]/2],R);  nadds++;}
                    if (nb[j]<0) {ecn2_sub_cuda(_MIPP_ &Tb[j*szb+(-nb[j])/2],R);  nadds++;}
                }
                else
                {
                    if (nb[j]>0) {ecn2_sub_cuda(_MIPP_ &Tb[j*szb+nb[j]/2],R);  nadds++;}
                    if (nb[j]<0) {ecn2_add_cuda(_MIPP_ &Tb[j*szb+(-nb[j])/2],R);  nadds++;}
                }
            }         
        }
    }
    ecn2_norm_cuda(_MIPP_ R);  
    return nadds;
}

/* Routines to support Galbraith, Lin, Scott (GLS) method for ECC */
/* requires an endomorphism psi */

/* *********************** */

/* Precompute T - first half from i.P, second half from i.psi(P) */ 

__device__ void ecn2_precomp_gls_cuda(_MIPD_ int sz,BOOL norm,ecn2 *P,zzn2 *psi,ecn2 *T)
{
    int i,j;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    j=0;

    MR_IN(219)

    ecn2_norm_cuda(_MIPP_ P);
    ecn2_copy_cuda(P,&T[0]);
    
    ecn2_pre_cuda(_MIPP_ sz,norm,T); /* precompute table */

    for (i=sz;i<sz+sz;i++)
    {
        ecn2_copy_cuda(&T[i-sz],&T[i]);
        ecn2_psi_cuda(_MIPP_ psi,&T[i]);
    }

    MR_OUT
}

#ifndef MR_NO_ECC_MULTIADD

/* Calculate a[0].P+a[1].psi(P) using interleaving method */

#define MR_MUL2_GLS_RESERVE (2+2*MR_ECC_STORE_N2*4)

__device__ int ecn2_mul2_gls_cuda(_MIPD_ big *a,ecn2 *P,zzn2 *psi,ecn2 *R)
{
    int i,j,nadds;
    ecn2 T[2*MR_ECC_STORE_N2];
    big a3[2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = (char *)memalloc_cuda(_MIPP_ MR_MUL2_GLS_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE));   
#endif

    for (j=i=0;i<2;i++)
        a3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);

    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        T[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++);       
        T[i].marker=MR_EPOINT_INFINITY;
    }
    MR_IN(220)

    ecn2_precomp_gls_cuda(_MIPP_ MR_ECC_STORE_N2,TRUE,P,psi,T);

    for (i=0;i<2;i++) premult_cuda(_MIPP_ a[i],3,a3[i]); /* calculate for NAF */

    nadds=ecn2_muln_engine_cuda(_MIPP_ 0,0,2,MR_ECC_STORE_N2,NULL,NULL,a,a3,NULL,T,R);

    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_GLS_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE));
#endif
    return nadds;
}

/* Calculates a[0]*P+a[1]*psi(P) + b[0]*Q+b[1]*psi(Q) 
   where P is fixed, and precomputations are already done off-line_cuda into FT
   using ecn2_precomp_gls_cuda. Useful for signature verification */

#define MR_MUL4_GLS_V_RESERVE (4+2*MR_ECC_STORE_N2*4)

__device__ int ecn2_mul4_gls_v_cuda(_MIPD_ big *a,int ns,ecn2 *FT,big *b,ecn2 *Q,zzn2 *psi,ecn2 *R)
{ 
    int i,j,nadds;
    ecn2 VT[2*MR_ECC_STORE_N2];
    big a3[2],b3[2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = (char *)memalloc_cuda(_MIPP_ MR_MUL4_GLS_V_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE));   
#endif
    j=0;
    for (i=0;i<2;i++)
    {
        a3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);
        b3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);
    }
    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        VT[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++);       
        VT[i].marker=MR_EPOINT_INFINITY;
    }

    MR_IN(217)

    ecn2_precomp_gls_cuda(_MIPP_ MR_ECC_STORE_N2,TRUE,Q,psi,VT); /* precompute for the variable points */
    for (i=0;i<2;i++)
    { /* needed for NAF */
        premult_cuda(_MIPP_ a[i],3,a3[i]);
        premult_cuda(_MIPP_ b[i],3,b3[i]);
    }
    nadds=ecn2_muln_engine_cuda(_MIPP_ 2,ns,2,MR_ECC_STORE_N2,a,a3,b,b3,FT,VT,R);
    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL4_GLS_V_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE));
#endif
    return nadds;
}

/* Calculate a.P+b.Q using interleaving method. P is fixed and FT is precomputed from it */

__device__ void ecn2_precomp_cuda(_MIPD_ int sz,BOOL norm,ecn2 *P,ecn2 *T)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(216)

    ecn2_norm_cuda(_MIPP_ P);
    ecn2_copy_cuda(P,&T[0]);
    ecn2_pre_cuda(_MIPP_ sz,norm,T); 

    MR_OUT
}

#ifndef MR_DOUBLE_BIG
#define MR_MUL2_RESERVE (2+2*MR_ECC_STORE_N2*4)
#else
#define MR_MUL2_RESERVE (4+2*MR_ECC_STORE_N2*4)
#endif

__device__ int ecn2_mul2_cuda(_MIPD_ big a,int ns,ecn2 *FT,big b,ecn2 *Q,ecn2 *R)
{
    int i,j,nadds;
    ecn2 T[2*MR_ECC_STORE_N2];
    big a3,b3;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = (char *)memalloc_cuda(_MIPP_ MR_MUL2_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_RESERVE));   
#endif

    j=0;
#ifndef MR_DOUBLE_BIG
    a3=mirvar_mem_cuda(_MIPP_ mem, j++);
	b3=mirvar_mem_cuda(_MIPP_ mem, j++);
#else
    a3=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
	b3=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
#endif    
    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        T[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++);       
        T[i].marker=MR_EPOINT_INFINITY;
    }

    MR_IN(218)

    ecn2_precomp_cuda(_MIPP_ MR_ECC_STORE_N2,TRUE,Q,T);

    premult_cuda(_MIPP_ a,3,a3); 
	premult_cuda(_MIPP_ b,3,b3); 

    nadds=ecn2_muln_engine_cuda(_MIPP_ 1,ns,1,MR_ECC_STORE_N2,&a,&a3,&b,&b3,FT,T,R);

    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_RESERVE));
#endif
    return nadds;
}
#endif
#endif

#ifndef MR_STATIC

__device__ BOOL ecn2_brick_init_cuda(_MIPD_ ebrick *B,zzn2 *x,zzn2 *y,big a,big b,big n,int window,int nb)
{ /* Uses Montgomery arithmetic internally              *
   * (x,y) is the fixed base                            *
   * a,b and n are parameters and modulus of the curve  *
   * window is the window size_cuda in bits and              *
   * nb is the maximum number of bits in the multiplier */
    int i,j,k,t,bp,len,bptr,is;
    ecn2 *table;
    ecn2 w;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (nb<2 || window<1 || window>nb || mr_mip->ERNUM) return FALSE;

    t=MR_ROUNDUP(nb,window);

    if (t<2) return FALSE;

    MR_IN(221)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return FALSE;
    }
#endif

    B->window=window;
    B->max=nb;
    table=(ecn2 *)mr_alloc_cuda(_MIPP_ (1<<window),sizeof(ecn2));
    if (table==NULL)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_OUT_OF_MEMORY);   
        MR_OUT
        return FALSE;
    }
    B->a=mirvar_cuda(_MIPP_ 0);
    B->b=mirvar_cuda(_MIPP_ 0);
    B->n=mirvar_cuda(_MIPP_ 0);
    copy_cuda(a,B->a);
    copy_cuda(b,B->b);
    copy_cuda(n,B->n);

    ecurve_init_cuda(_MIPP_ a,b,n,MR_AFFINE);
    mr_mip->TWIST=MR_QUADRATIC;

    w.x.a=mirvar_cuda(_MIPP_ 0);
    w.x.b=mirvar_cuda(_MIPP_ 0);
    w.y.a=mirvar_cuda(_MIPP_ 0);
    w.y.b=mirvar_cuda(_MIPP_ 0);
    w.marker=MR_EPOINT_INFINITY;
    ecn2_set_cuda(_MIPP_ x,y,&w);

    table[0].x.a=mirvar_cuda(_MIPP_ 0);
    table[0].x.b=mirvar_cuda(_MIPP_ 0);
    table[0].y.a=mirvar_cuda(_MIPP_ 0);
    table[0].y.b=mirvar_cuda(_MIPP_ 0);
    table[0].marker=MR_EPOINT_INFINITY;
    table[1].x.a=mirvar_cuda(_MIPP_ 0);
    table[1].x.b=mirvar_cuda(_MIPP_ 0);
    table[1].y.a=mirvar_cuda(_MIPP_ 0);
    table[1].y.b=mirvar_cuda(_MIPP_ 0);
    table[1].marker=MR_EPOINT_INFINITY;

    ecn2_copy_cuda(&w,&table[1]);
    for (j=0;j<t;j++)
        ecn2_add_cuda(_MIPP_ &w,&w);

    k=1;
    for (i=2;i<(1<<window);i++)
    {
        table[i].x.a=mirvar_cuda(_MIPP_ 0);
        table[i].x.b=mirvar_cuda(_MIPP_ 0);
        table[i].y.a=mirvar_cuda(_MIPP_ 0);
        table[i].y.b=mirvar_cuda(_MIPP_ 0);
        table[i].marker=MR_EPOINT_INFINITY;
        if (i==(1<<k))
        {
            k++;
            ecn2_copy_cuda(&w,&table[i]);
            
            for (j=0;j<t;j++)
                ecn2_add_cuda(_MIPP_ &w,&w);
            continue;
        }
        bp=1;
        for (j=0;j<k;j++)
        {
            if (i&bp)
			{
				is=1<<j;
                ecn2_add_cuda(_MIPP_ &table[is],&table[i]);
			}
            bp<<=1;
        }
    }
    mr_free_cuda(w.x.a);
    mr_free_cuda(w.x.b);
    mr_free_cuda(w.y.a);
    mr_free_cuda(w.y.b);

/* create the table */

    len=n->len;
    bptr=0;
    B->table=(mr_small *)mr_alloc_cuda(_MIPP_ 4*len*(1<<window),sizeof(mr_small));

    for (i=0;i<(1<<window);i++)
    {
        for (j=0;j<len;j++) B->table[bptr++]=table[i].x.a->w[j];
        for (j=0;j<len;j++) B->table[bptr++]=table[i].x.b->w[j];

        for (j=0;j<len;j++) B->table[bptr++]=table[i].y.a->w[j];
        for (j=0;j<len;j++) B->table[bptr++]=table[i].y.b->w[j];

        mr_free_cuda(table[i].x.a);
        mr_free_cuda(table[i].x.b);
        mr_free_cuda(table[i].y.a);
        mr_free_cuda(table[i].y.b);
    }
        
    mr_free_cuda(table);  

    MR_OUT
    return TRUE;
}

__device__ void ecn2_brick_end_cuda(ebrick *B)
{
    mirkill_cuda(B->n);
    mirkill_cuda(B->b);
    mirkill_cuda(B->a);
    mr_free_cuda(B->table);  
}

#else

/* use precomputated table in ROM */

__device__ void ecn2_brick_init_cuda(ebrick *B,const mr_small* rom,big a,big b,big n,int window,int nb)
{
    B->table=rom;
    B->a=a; /* just pass a pointer */
    B->b=b;
    B->n=n;
    B->window=window;  /* 2^4=16  stored values */
    B->max=nb;
}

#endif

/*
void ecn2_mul_brick(_MIPD_ ebrick *B,big e,zzn2 *x,zzn2 *y)
{
    int i,j,t,len,maxsize,promptr;
    ecn2 w,z;
 
#ifdef MR_STATIC
    char mem[MR_BIG_RESERVE(10)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (size_cuda(e)<0) mr_berror_cuda(_MIPP_ MR_ERR_NEG_POWER);
    t=MR_ROUNDUP(B->max,B->window);
    
    MR_IN(116)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return;
    }
#endif

    if (logb2_cuda(_MIPP_ e) > B->max)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_EXP_TOO_BIG);
        MR_OUT
        return;
    }

    ecurve_init_cuda(_MIPP_ B->a,B->b,B->n,MR_BEST);
    mr_mip->TWIST=MR_QUADRATIC;
  
#ifdef MR_STATIC
    memset(mem,0,MR_BIG_RESERVE(10));
#else
    mem=memalloc_cuda(_MIPP_ 10);
#endif

    w.x.a=mirvar_mem_cuda(_MIPP_  mem, 0);
    w.x.b=mirvar_mem_cuda(_MIPP_  mem, 1);
    w.y.a=mirvar_mem_cuda(_MIPP_  mem, 2);
    w.y.b=mirvar_mem_cuda(_MIPP_  mem, 3);  
    w.z.a=mirvar_mem_cuda(_MIPP_  mem, 4);
    w.z.b=mirvar_mem_cuda(_MIPP_  mem, 5);      
    w.marker=MR_EPOINT_INFINITY;
    z.x.a=mirvar_mem_cuda(_MIPP_  mem, 6);
    z.x.b=mirvar_mem_cuda(_MIPP_  mem, 7);
    z.y.a=mirvar_mem_cuda(_MIPP_  mem, 8);
    z.y.b=mirvar_mem_cuda(_MIPP_  mem, 9);       
    z.marker=MR_EPOINT_INFINITY;

    len=B->n->len;
    maxsize=4*(1<<B->window)*len;

    for (i=t-1;i>=0;i--)
    {
        j=recode_cuda(_MIPP_ e,t,B->window,i);
        ecn2_add_cuda(_MIPP_ &w,&w);
        if (j>0)
        {
            promptr=4*j*len;
            init_big_from_rom_cuda(z.x.a,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.x.b,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.y.a,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.y.b,len,B->table,maxsize,&promptr);
            z.marker=MR_EPOINT_NORMALIZED;
            ecn2_add_cuda(_MIPP_ &z,&w);
        }
    }
    ecn2_norm_cuda(_MIPP_ &w);
    ecn2_getxy_cuda(&w,x,y);
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem,10);
#else
    memset(mem,0,MR_BIG_RESERVE(10));
#endif
    MR_OUT
}
*/

__device__ void ecn2_mul_brick_gls_cuda(_MIPD_ ebrick *B,big *e,zzn2 *psi,zzn2 *x,zzn2 *y)
{
    int i,j,k,t,len,maxsize,promptr,se[2];
    ecn2 w,z;
 
#ifdef MR_STATIC
    char mem[MR_BIG_RESERVE(10)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    for (k=0;k<2;k++) se[k]=exsign_cuda(e[k]);

    t=MR_ROUNDUP(B->max,B->window);
    
    MR_IN(222)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return;
    }
#endif

    if (logb2_cuda(_MIPP_ e[0])>B->max || logb2_cuda(_MIPP_ e[1])>B->max)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_EXP_TOO_BIG);
        MR_OUT
        return;
    }

    ecurve_init_cuda(_MIPP_ B->a,B->b,B->n,MR_BEST);
    mr_mip->TWIST=MR_QUADRATIC;
  
#ifdef MR_STATIC
    memset(mem,0,MR_BIG_RESERVE(10));
#else
    mem=(char *)memalloc_cuda(_MIPP_ 10);
#endif

    z.x.a=mirvar_mem_cuda(_MIPP_  mem, 0);
    z.x.b=mirvar_mem_cuda(_MIPP_  mem, 1);
    z.y.a=mirvar_mem_cuda(_MIPP_  mem, 2);
    z.y.b=mirvar_mem_cuda(_MIPP_  mem, 3);       
    z.marker=MR_EPOINT_INFINITY;

    w.x.a=mirvar_mem_cuda(_MIPP_  mem, 4);
    w.x.b=mirvar_mem_cuda(_MIPP_  mem, 5);
    w.y.a=mirvar_mem_cuda(_MIPP_  mem, 6);
    w.y.b=mirvar_mem_cuda(_MIPP_  mem, 7);  
#ifndef MR_AFFINE_ONLY
    w.z.a=mirvar_mem_cuda(_MIPP_  mem, 8);
    w.z.b=mirvar_mem_cuda(_MIPP_  mem, 9); 
#endif    
    w.marker=MR_EPOINT_INFINITY;

    len=B->n->len;
    maxsize=4*(1<<B->window)*len;

    for (i=t-1;i>=0;i--)
    {
        ecn2_add_cuda(_MIPP_ &w,&w);
        for (k=0;k<2;k++)
        {
            j=recode_cuda(_MIPP_ e[k],t,B->window,i);
            if (j>0)
            {
                promptr=4*j*len;
                init_big_from_rom_cuda(z.x.a,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.x.b,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.y.a,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.y.b,len,B->table,maxsize,&promptr);
                z.marker=MR_EPOINT_NORMALIZED;
                if (k==1) ecn2_psi_cuda(_MIPP_ psi,&z);
                if (se[k]==PLUS) ecn2_add_cuda(_MIPP_ &z,&w);
                else             ecn2_sub_cuda(_MIPP_ &z,&w);
            }
        }      
    }
    ecn2_norm_cuda(_MIPP_ &w);
    ecn2_getxy_cuda(&w,x,y);
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem,10);
#else
    memset(mem,0,MR_BIG_RESERVE(10));
#endif
    MR_OUT
}

#else

/* Now for curves in Inverted Twisted Edwards Form */

__device__ BOOL ecn2_iszero_cuda(ecn2 *a)
{
    if (a->marker==MR_EPOINT_INFINITY) return TRUE;
    return FALSE;
}

__device__ void ecn2_copy_cuda(ecn2 *a,ecn2 *b)
{
    zzn2_copy_cuda(&(a->x),&(b->x));
    zzn2_copy_cuda(&(a->y),&(b->y));
    if (a->marker==MR_EPOINT_GENERAL)  zzn2_copy_cuda(&(a->z),&(b->z));
    b->marker=a->marker;
}

__device__ void ecn2_zero_cuda(ecn2 *a)
{
    zzn2_zero_cuda(&(a->x));
    zzn2_zero_cuda(&(a->y)); 
    if (a->marker==MR_EPOINT_GENERAL) zzn2_zero_cuda(&(a->z)); 
    a->marker=MR_EPOINT_INFINITY;
}

__device__ BOOL ecn2_compare_cuda(_MIPD_ ecn2 *a,ecn2 *b)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(193)
    ecn2_norm_cuda(_MIPP_ a);
    ecn2_norm_cuda(_MIPP_ b);
    MR_OUT
    if (zzn2_compare_cuda(&(a->x),&(b->x)) && zzn2_compare_cuda(&(a->y),&(b->y)) && a->marker==b->marker) return TRUE;
    return FALSE;
}

__device__ void ecn2_norm_cuda(_MIPD_ ecn2 *a)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (mr_mip->ERNUM) return;
    if (a->marker!=MR_EPOINT_GENERAL) return;

    MR_IN(194)
    
    zzn2_inv_cuda(_MIPP_ &(a->z));

    zzn2_mul_cuda(_MIPP_ &(a->x),&(a->z),&(a->x));
    zzn2_mul_cuda(_MIPP_ &(a->y),&(a->z),&(a->y));
    zzn2_from_zzn_cuda(mr_mip->one,&(a->z));
    a->marker=MR_EPOINT_NORMALIZED;

    MR_OUT

}

__device__ void ecn2_get_cuda(_MIPD_ ecn2 *e,zzn2 *x,zzn2 *y,zzn2 *z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    
    zzn2_copy_cuda(&(e->x),x);
    zzn2_copy_cuda(&(e->y),y);
    if (e->marker==MR_EPOINT_GENERAL) zzn2_copy_cuda(&(e->z),z);
    else                              zzn2_from_zzn_cuda(mr_mip->one,z);
}

__device__ void ecn2_getxy_cuda(ecn2 *e,zzn2 *x,zzn2 *y)
{
    zzn2_copy_cuda(&(e->x),x);
    zzn2_copy_cuda(&(e->y),y);
}

__device__ void ecn2_getx_cuda(ecn2 *e,zzn2 *x)
{
    zzn2_copy_cuda(&(e->x),x);
}

__device__ void ecn2_getz_cuda(_MIPD_ ecn2 *e,zzn2 *z)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (e->marker==MR_EPOINT_GENERAL) zzn2_copy_cuda(&(e->z),z);
    else                              zzn2_from_zzn_cuda(mr_mip->one,z);
}

__device__ void ecn2_psi_cuda(_MIPD_ zzn2 *psi,ecn2 *P)
{ /* apply GLS morphism to P */
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(212)
    zzn2_conj_cuda(_MIPP_ &(P->x),&(P->x));
    zzn2_conj_cuda(_MIPP_ &(P->y),&(P->y));
	if (P->marker==MR_EPOINT_GENERAL)
		zzn2_conj_cuda(_MIPP_ &(P->z),&(P->z));
    zzn2_mul_cuda(_MIPP_ &(P->x),&psi[0],&(P->x));

    MR_OUT
}
/*
static void out_zzn2(zzn2 *x)
{
	redc_cuda(x->a,x->a);
	redc_cuda(x->b,x->b);
	cotnum(x->a,stdout);
	cotnum(x->b,stdout);
	nres_cuda(x->a,x->a);
	nres_cuda(x->b,x->b);
}
*/

/* find RHS=(x^2-B)/(x^2-A) */

__device__ void ecn2_rhs_cuda(_MIPD_ zzn2 *x,zzn2 *rhs)
{ /* calculate RHS of elliptic curve equation */
    int twist;
    zzn2 A,B;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    twist=mr_mip->TWIST;

    MR_IN(202)

    A.a=mr_mip->w8;
    A.b=mr_mip->w9;
    B.a=mr_mip->w10;
    B.b=mr_mip->w11;

    zzn2_from_zzn_cuda(mr_mip->A,&A);
    zzn2_from_zzn_cuda(mr_mip->B,&B);
  
    if (twist==MR_QUADRATIC)
    { /* quadratic twist */
        zzn2_txx_cuda(_MIPP_ &A);
        zzn2_txx_cuda(_MIPP_ &B);
    }

    zzn2_sqr_cuda(_MIPP_ x,rhs);

    zzn2_sub_cuda(_MIPP_ rhs,&B,&B);

    zzn2_sub_cuda(_MIPP_ rhs,&A,&A);

    zzn2_inv_cuda(_MIPP_ &A);
    zzn2_mul_cuda(_MIPP_ &A,&B,rhs);

    MR_OUT
}

__device__ BOOL ecn2_set_cuda(_MIPD_ zzn2 *x,zzn2 *y,ecn2 *e)
{
    zzn2 lhs,rhs;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(195)

    lhs.a=mr_mip->w12;
    lhs.b=mr_mip->w13;
    rhs.a=mr_mip->w14;
    rhs.b=mr_mip->w15;

    ecn2_rhs_cuda(_MIPP_ x,&rhs);

    zzn2_sqr_cuda(_MIPP_ y,&lhs);

    if (!zzn2_compare_cuda(&lhs,&rhs))
    {
        MR_OUT
        return FALSE;
    }

    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(y,&(e->y));

    e->marker=MR_EPOINT_NORMALIZED;

    MR_OUT
    return TRUE;
}

#ifndef MR_NOSUPPORT_COMPRESSION

__device__ BOOL ecn2_setx_cuda(_MIPD_ zzn2 *x,ecn2 *e)
{
    zzn2 rhs;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return FALSE;

    MR_IN(201)

    rhs.a=mr_mip->w12;
    rhs.b=mr_mip->w13;

    ecn2_rhs_cuda(_MIPP_ x,&rhs);

    if (!zzn2_iszero_cuda(&rhs))
    {
		if (!zzn2_qr_cuda(_MIPP_ &rhs))
		{
            MR_OUT
            return FALSE;
		}
        zzn2_sqrt_cuda(_MIPP_ &rhs,&rhs); 
    }

    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(&rhs,&(e->y));

    e->marker=MR_EPOINT_NORMALIZED;

    MR_OUT
    return TRUE;
}

#endif

__device__ void ecn2_setxyz_cuda(zzn2 *x,zzn2 *y,zzn2 *z,ecn2 *e)
{
    zzn2_copy_cuda(x,&(e->x));
    zzn2_copy_cuda(y,&(e->y));
    zzn2_copy_cuda(z,&(e->z));
    e->marker=MR_EPOINT_GENERAL;
}

/* Normalise an array of points of length m<MR_MAX_M_T_S - requires a zzn2 workspace array of length m */

__device__ BOOL ecn2_multi_norm_cuda(_MIPD_ int m,zzn2 *work,ecn2 *p)
{ 

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
    int i;
	zzn2 one;
    zzn2 w[MR_MAX_M_T_S];
    if (mr_mip->ERNUM) return FALSE;   
    if (m>MR_MAX_M_T_S) return FALSE;

    MR_IN(215)
    
	one.a=mr_mip->w12;
    one.b=mr_mip->w13;

	zzn2_from_zzn_cuda(mr_mip->one,&one);

    for (i=0;i<m;i++)
	{
		if (p[i].marker==MR_EPOINT_NORMALIZED) w[i]=one;
        else w[i]=p[i].z;
	}

    if (!zzn2_multi_inverse_cuda(_MIPP_ m,w,work)) 
    {
       MR_OUT
       return FALSE;
    }

    for (i=0;i<m;i++)
    {
        p[i].marker=MR_EPOINT_NORMALIZED;
        zzn2_mul_cuda(_MIPP_ &(p[i].x),&work[i],&(p[i].x));    
        zzn2_mul_cuda(_MIPP_ &(p[i].y),&work[i],&(p[i].y));  
		zzn2_from_zzn_cuda(mr_mip->one,&(p[i].z));
    }    
    MR_OUT

    return TRUE;   
}

__device__ BOOL ecn2_add_cuda(_MIPD_ ecn2 *Q,ecn2 *P)
{ /* P+=Q */
    BOOL Doubling=FALSE;
    int twist;
    zzn2 t2,t3,t4;
 
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
 
    t2.a = mr_mip->w8; 
    t2.b = mr_mip->w9; 
    t3.a = mr_mip->w10; 
    t3.b = mr_mip->w11;
    t4.a = mr_mip->w12;
    t4.b = mr_mip->w13;

    twist=mr_mip->TWIST;
    if (mr_mip->ERNUM) return FALSE;

    if (P->marker==MR_EPOINT_INFINITY)
    {
        ecn2_copy_cuda(Q,P);
        return Doubling;
    }
    if (Q->marker==MR_EPOINT_INFINITY) return Doubling;

    if (Q==P)
    {
        Doubling=TRUE;
        if (P->marker==MR_EPOINT_INFINITY) 
        { /* 2 times infinity == infinity ! */
            return Doubling;
        }
    }

    MR_IN(205)

    if (!Doubling)
    { /* Addition */
        zzn2_add_cuda(_MIPP_ &(Q->x),&(Q->y),&t2);
        zzn2_add_cuda(_MIPP_ &(P->x),&(P->y),&t4);
        zzn2_mul_cuda(_MIPP_ &t4,&t2,&t4);          /* I = t4 = (x1+y1)(x2+y2) */
        if (Q->marker!=MR_EPOINT_NORMALIZED)
        {
            if (P->marker==MR_EPOINT_NORMALIZED)
                zzn2_copy_cuda(&(Q->z),&(P->z));
            else
                zzn2_mul_cuda(_MIPP_ &(Q->z),&(P->z),&(P->z));  /* Z = z1*z2 */
        }  
        else
        {
            if (P->marker==MR_EPOINT_NORMALIZED)
                zzn2_from_zzn_cuda(mr_mip->one,&(P->z));
        }
        zzn2_sqr_cuda(_MIPP_ &(P->z),&t2);    /* P->z = z1.z2 */
        if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)
            zzn2_smul_cuda(_MIPP_ &t2,mr_mip->B,&t2);
        else
            zzn2_imul_cuda(_MIPP_ &t2,mr_mip->Bsize,&t2);
        if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &t2);              /* B = t2 = d*A^2 */
        zzn2_mul_cuda(_MIPP_ &(P->x),&(Q->x),&(P->x));     /* X = x1*x2 */
        zzn2_mul_cuda(_MIPP_ &(P->y),&(Q->y),&(P->y));     /* Y = y1*y2 */
        zzn2_sub_cuda(_MIPP_ &t4,&(P->x),&t4);
        zzn2_sub_cuda(_MIPP_ &t4,&(P->y),&t4);             /* I = (x1+y1)(x2+y2)-X-Y */ 
        zzn2_mul_cuda(_MIPP_ &(P->x),&(P->y),&t3);         /* E = t3 = X*Y */
        if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
            zzn2_smul_cuda(_MIPP_ &(P->y),mr_mip->A,&(P->y));
        else
            zzn2_imul_cuda(_MIPP_ &(P->y),mr_mip->Asize,&(P->y));
        if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &(P->y));         /* Y=aY */
        zzn2_sub_cuda(_MIPP_ &(P->x),&(P->y),&(P->x));    /* X=X-aY */
        zzn2_mul_cuda(_MIPP_ &(P->z),&(P->x),&(P->z));
        zzn2_mul_cuda(_MIPP_ &(P->z),&t4,&(P->z));
        zzn2_sub_cuda(_MIPP_ &t3,&t2,&(P->y));
        zzn2_mul_cuda(_MIPP_ &(P->y),&t4,&(P->y));
        zzn2_add_cuda(_MIPP_ &t3,&t2,&t4);
        zzn2_mul_cuda(_MIPP_ &(P->x),&t4,&(P->x));
    }
    else
    { /* doubling */
        zzn2_add_cuda(_MIPP_ &(P->x),&(P->y),&t2);
        zzn2_sqr_cuda(_MIPP_ &t2,&t2);
        zzn2_sqr_cuda(_MIPP_ &(P->x),&(P->x));
        zzn2_sqr_cuda(_MIPP_ &(P->y),&(P->y));
        zzn2_sub_cuda(_MIPP_ &t2,&(P->x),&t2);
        zzn2_sub_cuda(_MIPP_ &t2,&(P->y),&t2);   /* E=(X+Y)^2-X^2-Y^2 */

        if (P->marker!=MR_EPOINT_NORMALIZED)
            zzn2_sqr_cuda(_MIPP_ &(P->z),&(P->z));
        else
            zzn2_from_zzn_cuda(mr_mip->one,&(P->z));

        zzn2_add_cuda(_MIPP_ &(P->z),&(P->z),&(P->z));
        if (mr_abs(mr_mip->Bsize)==MR_TOOBIG)
            zzn2_smul_cuda(_MIPP_ &(P->z),mr_mip->B,&(P->z));
        else
            zzn2_imul_cuda(_MIPP_ &(P->z),mr_mip->Bsize,&(P->z));
        if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &(P->z));
        if (mr_abs(mr_mip->Asize)==MR_TOOBIG)
            zzn2_smul_cuda(_MIPP_ &(P->y),mr_mip->A,&(P->y));
        else
            zzn2_imul_cuda(_MIPP_ &(P->y),mr_mip->Asize,&(P->y));
        if (twist==MR_QUADRATIC) zzn2_txx_cuda(_MIPP_ &(P->y));
        zzn2_add_cuda(_MIPP_ &(P->x),&(P->y),&t3);
        zzn2_sub_cuda(_MIPP_ &(P->x),&(P->y),&t4);
        zzn2_mul_cuda(_MIPP_ &t3,&t4,&(P->x));

        zzn2_sub_cuda(_MIPP_ &t3,&(P->z),&t3);
        zzn2_mul_cuda(_MIPP_ &t2,&t3,&(P->y));
        zzn2_mul_cuda(_MIPP_ &t2,&t4,&(P->z));
    }

    if (zzn2_iszero_cuda(&(P->z)))
    {
        zzn2_from_zzn_cuda(mr_mip->one,&(P->x));
        zzn2_zero_cuda(&(P->y));
        P->marker=MR_EPOINT_INFINITY;
    }
    else P->marker=MR_EPOINT_GENERAL;
   
    MR_OUT
    return Doubling;
}

__device__ void ecn2_negate_cuda(_MIPD_ ecn2 *u,ecn2 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    ecn2_copy_cuda(u,w);
    if (w->marker!=MR_EPOINT_INFINITY)
        zzn2_negate_cuda(_MIPP_ &(w->x),&(w->x));
}


__device__ BOOL ecn2_sub_cuda(_MIPD_ ecn2 *Q,ecn2 *P)
{
    BOOL Doubling;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 lam;

    lam.a = mr_mip->w14;
    lam.b = mr_mip->w15;

    ecn2_negate_cuda(_MIPP_ Q,Q);

    Doubling=ecn2_add_cuda(_MIPP_ Q,P);

    ecn2_negate_cuda(_MIPP_ Q,Q);

    return Doubling;
}

/*

BOOL ecn2_add_sub_cuda(_MIPD_ ecn2 *P,ecn2 *Q,ecn2 *PP,ecn2 *PM)
{  PP=P+Q, PM=P-Q. Assumes P and Q are both normalized, and P!=Q 
 #ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 t1,t2,lam;

    if (mr_mip->ERNUM) return FALSE;

    PP->marker=MR_EPOINT_NORMALIZED;
    PM->marker=MR_EPOINT_NORMALIZED;

    return TRUE;
}

*/

/* Precomputation of  3P, 5P, 7P etc. into PT. Assume PT[0] contains P */

#define MR_PRE_2 (6+2*MR_ECC_STORE_N2)

__device__ static void ecn2_pre_cuda(_MIPD_ int sz,BOOL norm,ecn2 *PT)
{
    int i,j;
    ecn2 P2;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
	zzn2 *work=(zzn2 *)mr_alloc_cuda(_MIPP_ sz,sizeof(zzn2));
    char *mem = memalloc_cuda(_MIPP_ 6+2*sz);
#else
	zzn2 work[MR_ECC_STORE_N2];
    char mem[MR_BIG_RESERVE(MR_PRE_2)];
    memset(mem, 0, MR_BIG_RESERVE(MR_PRE_2));
#endif
    j=0;
    P2.x.a=mirvar_mem_cuda(_MIPP_ mem, j++);
    P2.x.b=mirvar_mem_cuda(_MIPP_ mem, j++);
    P2.y.a=mirvar_mem_cuda(_MIPP_ mem, j++);
    P2.y.b=mirvar_mem_cuda(_MIPP_ mem, j++);
    P2.z.a=mirvar_mem_cuda(_MIPP_ mem, j++);
    P2.z.b=mirvar_mem_cuda(_MIPP_ mem, j++);

    for (i=0;i<sz;i++)
    {
        work[i].a= mirvar_mem_cuda(_MIPP_ mem, j++);
        work[i].b= mirvar_mem_cuda(_MIPP_ mem, j++);
    }

    ecn2_copy_cuda(&PT[0],&P2);
    ecn2_add_cuda(_MIPP_ &P2,&P2);
    for (i=1;i<sz;i++)
    {
        ecn2_copy_cuda(&PT[i-1],&PT[i]);
        ecn2_add_cuda(_MIPP_ &P2,&PT[i]);
		
    }
	if (norm) ecn2_multi_norm_cuda(_MIPP_ sz,work,PT);

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, 6+2*sz);
	mr_free_cuda(work);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_PRE_2));
#endif
}

#ifndef MR_DOUBLE_BIG
#define MR_MUL_RESERVE (1+6*MR_ECC_STORE_N2)
#else
#define MR_MUL_RESERVE (2+6*MR_ECC_STORE_N2)
#endif

__device__ int ecn2_mul_cuda(_MIPD_ big k,ecn2 *P)
{
    int i,j,nb,n,nbs,nzs,nadds;
    big h;
	BOOL neg;
    ecn2 T[MR_ECC_STORE_N2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = memalloc_cuda(_MIPP_ MR_MUL_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL_RESERVE)];
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL_RESERVE));
#endif

    j=0;
#ifndef MR_DOUBLE_BIG
    h=mirvar_mem_cuda(_MIPP_ mem, j++);
#else
    h=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
#endif
    for (i=0;i<MR_ECC_STORE_N2;i++)
    {
        T[i].x.a= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].x.b= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].y.a= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].y.b= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].z.a= mirvar_mem_cuda(_MIPP_ mem, j++);
        T[i].z.b= mirvar_mem_cuda(_MIPP_ mem, j++);
    }

    MR_IN(207)

    ecn2_norm_cuda(_MIPP_ P);

	nadds=0;

	neg=FALSE;
	if (size_cuda(k)<0)
	{
		negify_cuda(k,k);
		ecn2_negate_cuda(_MIPP_ P,&T[0]);
		neg=TRUE;
	}
	else ecn2_copy_cuda(P,&T[0]);

    premult_cuda(_MIPP_ k,3,h);

    ecn2_pre_cuda(_MIPP_ MR_ECC_STORE_N2,FALSE,T);
    nb=logb2_cuda(_MIPP_ h);

    ecn2_zero_cuda(P);

    for (i=nb-1;i>=1;)
    {
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        n=mr_naf_window_cuda(_MIPP_ k,h,i,&nbs,&nzs,MR_ECC_STORE_N2);
 
        for (j=0;j<nbs;j++) ecn2_add_cuda(_MIPP_ P,P);
       
        if (n>0) {nadds++; ecn2_add_cuda(_MIPP_ &T[n/2],P);}
        if (n<0) {nadds++; ecn2_sub_cuda(_MIPP_ &T[(-n)/2],P);}
        i-=nbs;
        if (nzs)
        {
            for (j=0;j<nzs;j++) ecn2_add_cuda(_MIPP_ P,P);
            i-=nzs;
        }
    }
	if (neg) negify_cuda(k,k);

    ecn2_norm_cuda(_MIPP_ P);
    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL_RESERVE));
#endif
	return nadds;
}

/* Double addition, using Joint Sparse Form */
/* R=aP+bQ */

#define MR_MUL2_JSF_RESERVE 24

__device__ int ecn2_mul2_jsf_cuda(_MIPD_ big a,ecn2 *P,big b,ecn2 *Q,ecn2 *R)
{
    int e1,h1,e2,h2,bb,nadds;
    ecn2 P1,P2,PS,PD;
    big c,d,e,f;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = memalloc_cuda(_MIPP_ MR_MUL2_JSF_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE)];
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE));
#endif

    c = mirvar_mem_cuda(_MIPP_ mem, 0);
    d = mirvar_mem_cuda(_MIPP_ mem, 1);
    e = mirvar_mem_cuda(_MIPP_ mem, 2);
    f = mirvar_mem_cuda(_MIPP_ mem, 3);
    P1.x.a= mirvar_mem_cuda(_MIPP_ mem, 4);
    P1.x.b= mirvar_mem_cuda(_MIPP_ mem, 5);
    P1.y.a= mirvar_mem_cuda(_MIPP_ mem, 6);
    P1.y.b= mirvar_mem_cuda(_MIPP_ mem, 7);
    P2.x.a= mirvar_mem_cuda(_MIPP_ mem, 8);
    P2.x.b= mirvar_mem_cuda(_MIPP_ mem, 9);
    P2.y.a= mirvar_mem_cuda(_MIPP_ mem, 10);
    P2.y.b= mirvar_mem_cuda(_MIPP_ mem, 11);
    PS.x.a= mirvar_mem_cuda(_MIPP_ mem, 12);
    PS.x.b= mirvar_mem_cuda(_MIPP_ mem, 13);
    PS.y.a= mirvar_mem_cuda(_MIPP_ mem, 14);
    PS.y.b= mirvar_mem_cuda(_MIPP_ mem, 15);
    PS.z.a= mirvar_mem_cuda(_MIPP_ mem, 16);
    PS.z.b= mirvar_mem_cuda(_MIPP_ mem, 17);
    PD.x.a= mirvar_mem_cuda(_MIPP_ mem, 18);
    PD.x.b= mirvar_mem_cuda(_MIPP_ mem, 19);
    PD.y.a= mirvar_mem_cuda(_MIPP_ mem, 20);
    PD.y.b= mirvar_mem_cuda(_MIPP_ mem, 21);
    PD.z.a= mirvar_mem_cuda(_MIPP_ mem, 22);
    PD.z.b= mirvar_mem_cuda(_MIPP_ mem, 23);

    MR_IN(206)

    ecn2_norm_cuda(_MIPP_ Q); 
    ecn2_copy_cuda(Q,&P2); 

    copy_cuda(b,d);
    if (size_cuda(d)<0) 
    {
        negify_cuda(d,d);
        ecn2_negate_cuda(_MIPP_ &P2,&P2);
    }

    ecn2_norm_cuda(_MIPP_ P); 
    ecn2_copy_cuda(P,&P1); 

    copy_cuda(a,c);
    if (size_cuda(c)<0) 
    {
        negify_cuda(c,c);
        ecn2_negate_cuda(_MIPP_ &P1,&P1);
    }

    mr_jsf_cuda(_MIPP_ d,c,e,d,f,c);   /* calculate joint sparse form */
 
    if (mr_compare_cuda(e,f)>0) bb=logb2_cuda(_MIPP_ e)-1;
    else                bb=logb2_cuda(_MIPP_ f)-1;

    /*ecn2_add_sub_cuda(_MIPP_ &P1,&P2,&PS,&PD);*/

    ecn2_copy_cuda(&P1,&PS);
    ecn2_copy_cuda(&P1,&PD);
    ecn2_add_cuda(_MIPP_ &P2,&PS);
    ecn2_sub_cuda(_MIPP_ &P2,&PD);

    ecn2_zero_cuda(R);
	nadds=0;
   
    while (bb>=0) 
    { /* add_cuda/subtract_cuda method */
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        ecn2_add_cuda(_MIPP_ R,R);
        e1=h1=e2=h2=0;

        if (mr_testbit_cuda(_MIPP_ d,bb)) e2=1;
        if (mr_testbit_cuda(_MIPP_ e,bb)) h2=1;
        if (mr_testbit_cuda(_MIPP_ c,bb)) e1=1;
        if (mr_testbit_cuda(_MIPP_ f,bb)) h1=1;

        if (e1!=h1)
        {
            if (e2==h2)
            {
                if (h1==1) {ecn2_add_cuda(_MIPP_ &P1,R); nadds++;}
                else       {ecn2_sub_cuda(_MIPP_ &P1,R); nadds++;}
            }
            else
            {
                if (h1==1)
                {
                    if (h2==1) {ecn2_add_cuda(_MIPP_ &PS,R); nadds++;}
                    else       {ecn2_add_cuda(_MIPP_ &PD,R); nadds++;}
                }
                else
                {
                    if (h2==1) {ecn2_sub_cuda(_MIPP_ &PD,R); nadds++;}
                    else       {ecn2_sub_cuda(_MIPP_ &PS,R); nadds++;}
                }
            }
        }
        else if (e2!=h2)
        {
            if (h2==1) {ecn2_add_cuda(_MIPP_ &P2,R); nadds++;}
            else       {ecn2_sub_cuda(_MIPP_ &P2,R); nadds++;}
        }
        bb-=1;
    }
    ecn2_norm_cuda(_MIPP_ R); 

    MR_OUT
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_JSF_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_JSF_RESERVE));
#endif
	return nadds;

}

/* General purpose multi-exponentiation engine, using inter-leaving algorithm. Calculate aP+bQ+cR+dS...
   Inputs are divided into two groups of sizes wa<4 and wb<4. For the first group if the points are fixed the 
   first precomputed Table Ta[] may be taken from ROM. For the second group if the points are variable Tb[j] will
   have to computed online. Each group has its own precomputed store size_cuda, sza (=8?) and szb (=20?) respectively. 
   The values a,b,c.. are provided in ma[] and mb[], and 3.a,3.b,3.c (as required by the NAF) are provided in 
   ma3[] and mb3[]. If only one group is required, set wb=0 and pass NULL pointers.
   */

__device__ int ecn2_muln_engine_cuda(_MIPD_ int wa,int sza,int wb,int szb,big *ma,big *ma3,big *mb,big *mb3,ecn2 *Ta,ecn2 *Tb,ecn2 *R)
{ /* general purpose interleaving algorithm engine for multi-exp */
    int i,j,tba[4],pba[4],na[4],sa[4],tbb[4],pbb[4],nb[4],sb[4],nbits,nbs,nzs;
    int nadds;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    ecn2_zero_cuda(R);

    nbits=0;
    for (i=0;i<wa;i++) {sa[i]=exsign_cuda(ma[i]); tba[i]=0; j=logb2_cuda(_MIPP_ ma3[i]); if (j>nbits) nbits=j; }
    for (i=0;i<wb;i++) {sb[i]=exsign_cuda(mb[i]); tbb[i]=0; j=logb2_cuda(_MIPP_ mb3[i]); if (j>nbits) nbits=j; }
    
    nadds=0;
    for (i=nbits-1;i>=1;i--)
    {
        if (mr_mip->user!=NULL) (*mr_mip->user)();
        if (R->marker!=MR_EPOINT_INFINITY) ecn2_add_cuda(_MIPP_ R,R);
        for (j=0;j<wa;j++)
        { /* deal with the first group */
            if (tba[j]==0)
            {
                na[j]=mr_naf_window_cuda(_MIPP_ ma[j],ma3[j],i,&nbs,&nzs,sza);
                tba[j]=nbs+nzs;
                pba[j]=nbs;
            }
            tba[j]--;  pba[j]--; 
            if (pba[j]==0)
            {
                if (sa[j]==PLUS)
                {
                    if (na[j]>0) {ecn2_add_cuda(_MIPP_ &Ta[j*sza+na[j]/2],R); nadds++;}
                    if (na[j]<0) {ecn2_sub_cuda(_MIPP_ &Ta[j*sza+(-na[j])/2],R); nadds++;}
                }
                else
                {
                    if (na[j]>0) {ecn2_sub_cuda(_MIPP_ &Ta[j*sza+na[j]/2],R); nadds++;}
                    if (na[j]<0) {ecn2_add_cuda(_MIPP_ &Ta[j*sza+(-na[j])/2],R); nadds++;}
                }
            }         
        }
        for (j=0;j<wb;j++)
        { /* deal with the second group */
            if (tbb[j]==0)
            {
                nb[j]=mr_naf_window_cuda(_MIPP_ mb[j],mb3[j],i,&nbs,&nzs,szb);
                tbb[j]=nbs+nzs;
                pbb[j]=nbs;
            }
            tbb[j]--;  pbb[j]--; 
            if (pbb[j]==0)
            {
                if (sb[j]==PLUS)
                {
                    if (nb[j]>0) {ecn2_add_cuda(_MIPP_ &Tb[j*szb+nb[j]/2],R);  nadds++;}
                    if (nb[j]<0) {ecn2_sub_cuda(_MIPP_ &Tb[j*szb+(-nb[j])/2],R);  nadds++;}
                }
                else
                {
                    if (nb[j]>0) {ecn2_sub_cuda(_MIPP_ &Tb[j*szb+nb[j]/2],R);  nadds++;}
                    if (nb[j]<0) {ecn2_add_cuda(_MIPP_ &Tb[j*szb+(-nb[j])/2],R);  nadds++;}
                }
            }         
        }
    }
    ecn2_norm_cuda(_MIPP_ R);  
    return nadds;
}

/* Routines to support Galbraith, Lin, Scott (GLS) method for ECC */
/* requires an endomorphism psi */

/* *********************** */

/* Precompute T - first half from i.P, second half from i.psi(P) */
/* norm=TRUE if the table is to be normalised - which it should be */
/* if it is to be calculated off-line_cuda */

__device__ void ecn2_precomp_gls_cuda(_MIPD_ int sz,BOOL norm,ecn2 *P,zzn2 *psi,ecn2 *T)
{
    int i,j;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    j=0;

    MR_IN(219)

    ecn2_norm_cuda(_MIPP_ P);
    ecn2_copy_cuda(P,&T[0]);

    ecn2_pre_cuda(_MIPP_ sz,norm,T); /* precompute table */
    for (i=sz;i<sz+sz;i++)
    {
        ecn2_copy_cuda(&T[i-sz],&T[i]);
        ecn2_psi_cuda(_MIPP_ psi,&T[i]);
    }

    MR_OUT
}

/* Calculate a[0].P+a[1].psi(P) using interleaving method */

#define MR_MUL2_GLS_RESERVE (2+2*MR_ECC_STORE_N2*6)

__device__ int ecn2_mul2_gls_cuda(_MIPD_ big *a,ecn2 *P,zzn2 *psi,ecn2 *R)
{
    int i,j,nadds;
    ecn2 T[2*MR_ECC_STORE_N2];
    big a3[2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = memalloc_cuda(_MIPP_ MR_MUL2_GLS_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE));   
#endif

    for (j=i=0;i<2;i++)
        a3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);

    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        T[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++);  
        T[i].z.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].z.b=mirvar_mem_cuda(_MIPP_  mem, j++);          
        T[i].marker=MR_EPOINT_INFINITY;
    }
    MR_IN(220)

    ecn2_precomp_gls_cuda(_MIPP_ MR_ECC_STORE_N2,FALSE,P,psi,T);

    for (i=0;i<2;i++) premult_cuda(_MIPP_ a[i],3,a3[i]); /* calculate for NAF */

    nadds=ecn2_muln_engine_cuda(_MIPP_ 0,0,2,MR_ECC_STORE_N2,NULL,NULL,a,a3,NULL,T,R);

    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_GLS_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_GLS_RESERVE));
#endif
    return nadds;
}

/* Calculates a[0]*P+a[1]*psi(P) + b[0]*Q+b[1]*psi(Q) 
   where P is fixed, and precomputations are already done off-line_cuda into FT
   using ecn2_precomp_gls_cuda. Useful for signature verification */

#define MR_MUL4_GLS_V_RESERVE (4+2*MR_ECC_STORE_N2*6)

__device__ int ecn2_mul4_gls_v_cuda(_MIPD_ big *a,int ns,ecn2 *FT,big *b,ecn2 *Q,zzn2 *psi,ecn2 *R)
{ 
    int i,j,nadds;
    ecn2 VT[2*MR_ECC_STORE_N2];
    big a3[2],b3[2];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = memalloc_cuda(_MIPP_ MR_MUL4_GLS_V_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE));   
#endif
    j=0;
    for (i=0;i<2;i++)
    {
        a3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);
        b3[i]=mirvar_mem_cuda(_MIPP_ mem, j++);
    }
    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        VT[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++);  
        VT[i].z.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        VT[i].z.b=mirvar_mem_cuda(_MIPP_  mem, j++);         
        VT[i].marker=MR_EPOINT_INFINITY;
    }

    MR_IN(217)

    ecn2_precomp_gls_cuda(_MIPP_ MR_ECC_STORE_N2,FALSE,Q,psi,VT); /* precompute for the variable points */
    for (i=0;i<2;i++)
    { /* needed for NAF */
        premult_cuda(_MIPP_ a[i],3,a3[i]);
        premult_cuda(_MIPP_ b[i],3,b3[i]);
    }
    nadds=ecn2_muln_engine_cuda(_MIPP_ 2,ns,2,MR_ECC_STORE_N2,a,a3,b,b3,FT,VT,R);
    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL4_GLS_V_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL4_GLS_V_RESERVE));
#endif
    return nadds;
}

/* Calculate a.P+b.Q using interleaving method. P is fixed and T is precomputed from it */

__device__ void ecn2_precomp_cuda(_MIPD_ int sz,BOOL norm,ecn2 *P,ecn2 *T)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    MR_IN(216)

    ecn2_norm_cuda(_MIPP_ P);
    ecn2_copy_cuda(P,&T[0]);
    ecn2_pre_cuda(_MIPP_ sz,norm,T); 

    MR_OUT
}

#ifndef MR_DOUBLE_BIG
#define MR_MUL2_RESERVE (2+2*MR_ECC_STORE_N2*6)
#else
#define MR_MUL2_RESERVE (4+2*MR_ECC_STORE_N2*6)
#endif

__device__ int ecn2_mul2_cuda(_MIPD_ big a,int ns,ecn2 *FT,big b,ecn2 *Q,ecn2 *R)
{
    int i,j,nadds;
    ecn2 T[2*MR_ECC_STORE_N2];
    big a3,b3;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

#ifndef MR_STATIC
    char *mem = memalloc_cuda(_MIPP_ MR_MUL2_RESERVE);
#else
    char mem[MR_BIG_RESERVE(MR_MUL2_RESERVE)];       
 	memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_RESERVE));   
#endif

    j=0;
#ifndef MR_DOUBLE_BIG
    a3=mirvar_mem_cuda(_MIPP_ mem, j++);
	b3=mirvar_mem_cuda(_MIPP_ mem, j++);
#else
    a3=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
	b3=mirvar_mem_cuda(_MIPP_ mem, j); j+=2;
#endif    
    for (i=0;i<2*MR_ECC_STORE_N2;i++)
    {
        T[i].x.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].x.b=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].y.b=mirvar_mem_cuda(_MIPP_  mem, j++); 
        T[i].z.a=mirvar_mem_cuda(_MIPP_  mem, j++);
        T[i].z.b=mirvar_mem_cuda(_MIPP_  mem, j++);        
        T[i].marker=MR_EPOINT_INFINITY;
    }

    MR_IN(218)

    ecn2_precomp_cuda(_MIPP_ MR_ECC_STORE_N2,FALSE,Q,T);

    premult_cuda(_MIPP_ a,3,a3); 
	premult_cuda(_MIPP_ b,3,b3); 

    nadds=ecn2_muln_engine_cuda(_MIPP_ 1,ns,1,MR_ECC_STORE_N2,&a,&a3,&b,&b3,FT,T,R);

    ecn2_norm_cuda(_MIPP_ R);

    MR_OUT

#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem, MR_MUL2_RESERVE);
#else
    memset(mem, 0, MR_BIG_RESERVE(MR_MUL2_RESERVE));
#endif
    return nadds;
}


#ifndef MR_STATIC

__device__ BOOL ecn2_brick_init_cuda(_MIPD_ ebrick *B,zzn2 *x,zzn2 *y,big a,big b,big n,int window,int nb)
{ /* Uses Montgomery arithmetic internally              *
   * (x,y) is the fixed base                            *
   * a,b and n are parameters and modulus of the curve  *
   * window is the window size_cuda in bits and              *
   * nb is the maximum number of bits in the multiplier */
    int i,j,k,t,bp,len,bptr;
    ecn2 *table;
    ecn2 w;

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (nb<2 || window<1 || window>nb || mr_mip->ERNUM) return FALSE;

    t=MR_ROUNDUP(nb,window);
    if (t<2) return FALSE;

    MR_IN(221)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return FALSE;
    }
#endif

    B->window=window;
    B->max=nb;
    table=mr_alloc_cuda(_MIPP_ (1<<window),sizeof(ecn2));
    if (table==NULL)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_OUT_OF_MEMORY);   
        MR_OUT
        return FALSE;
    }
    B->a=mirvar_cuda(_MIPP_ 0);
    B->b=mirvar_cuda(_MIPP_ 0);
    B->n=mirvar_cuda(_MIPP_ 0);
    copy_cuda(a,B->a);
    copy_cuda(b,B->b);
    copy_cuda(n,B->n);

    ecurve_init_cuda(_MIPP_ a,b,n,MR_BEST);
    mr_mip->TWIST=MR_QUADRATIC;

    w.x.a=mirvar_cuda(_MIPP_ 0);
    w.x.b=mirvar_cuda(_MIPP_ 0);
    w.y.a=mirvar_cuda(_MIPP_ 0);
    w.y.b=mirvar_cuda(_MIPP_ 0);
    w.z.a=mirvar_cuda(_MIPP_ 0);
    w.z.b=mirvar_cuda(_MIPP_ 0);

    w.marker=MR_EPOINT_INFINITY;
    ecn2_set_cuda(_MIPP_ x,y,&w);

    table[0].x.a=mirvar_cuda(_MIPP_ 0);
    table[0].x.b=mirvar_cuda(_MIPP_ 0);
    table[0].y.a=mirvar_cuda(_MIPP_ 0);
    table[0].y.b=mirvar_cuda(_MIPP_ 0);
    table[0].z.a=mirvar_cuda(_MIPP_ 0);
    table[0].z.b=mirvar_cuda(_MIPP_ 0);
    table[0].marker=MR_EPOINT_INFINITY;
    table[1].x.a=mirvar_cuda(_MIPP_ 0);
    table[1].x.b=mirvar_cuda(_MIPP_ 0);
    table[1].y.a=mirvar_cuda(_MIPP_ 0);
    table[1].y.b=mirvar_cuda(_MIPP_ 0);
    table[1].z.a=mirvar_cuda(_MIPP_ 0);
    table[1].z.b=mirvar_cuda(_MIPP_ 0);
    table[1].marker=MR_EPOINT_INFINITY;

    ecn2_copy_cuda(&w,&table[1]);
    for (j=0;j<t;j++)
        ecn2_add_cuda(_MIPP_ &w,&w);

    k=1;
    for (i=2;i<(1<<window);i++)
    {
        table[i].x.a=mirvar_cuda(_MIPP_ 0);
        table[i].x.b=mirvar_cuda(_MIPP_ 0);
        table[i].y.a=mirvar_cuda(_MIPP_ 0);
        table[i].y.b=mirvar_cuda(_MIPP_ 0);
        table[i].z.a=mirvar_cuda(_MIPP_ 0);
        table[i].z.b=mirvar_cuda(_MIPP_ 0);
        table[i].marker=MR_EPOINT_INFINITY;
        if (i==(1<<k))
        {
            k++;
			ecn2_norm_cuda(_MIPP_ &w);
            ecn2_copy_cuda(&w,&table[i]);
            
            for (j=0;j<t;j++)
                ecn2_add_cuda(_MIPP_ &w,&w);
            continue;
        }
        bp=1;
        for (j=0;j<k;j++)
        {
            if (i&bp)
                ecn2_add_cuda(_MIPP_ &table[1<<j],&table[i]);
            bp<<=1;
        }
        ecn2_norm_cuda(_MIPP_ &table[i]);
    }
    mr_free_cuda(w.x.a);
    mr_free_cuda(w.x.b);
    mr_free_cuda(w.y.a);
    mr_free_cuda(w.y.b);
    mr_free_cuda(w.z.a);
    mr_free_cuda(w.z.b);

/* create the table */

    len=n->len;
    bptr=0;
    B->table=mr_alloc_cuda(_MIPP_ 4*len*(1<<window),sizeof(mr_small));

    for (i=0;i<(1<<window);i++)
    {
        for (j=0;j<len;j++) B->table[bptr++]=table[i].x.a->w[j];
        for (j=0;j<len;j++) B->table[bptr++]=table[i].x.b->w[j];

        for (j=0;j<len;j++) B->table[bptr++]=table[i].y.a->w[j];
        for (j=0;j<len;j++) B->table[bptr++]=table[i].y.b->w[j];

        mr_free_cuda(table[i].x.a);
        mr_free_cuda(table[i].x.b);
        mr_free_cuda(table[i].y.a);
        mr_free_cuda(table[i].y.b);
        mr_free_cuda(table[i].z.a);
        mr_free_cuda(table[i].z.b);
    }
        
    mr_free_cuda(table);  

    MR_OUT
    return TRUE;
}

__device__ void ecn2_brick_end_cuda(ebrick *B)
{
    mirkill_cuda(B->n);
    mirkill_cuda(B->b);
    mirkill_cuda(B->a);
    mr_free_cuda(B->table);  
}

#else

/* use precomputated table in ROM */

__device__ void ecn2_brick_init_cuda(ebrick *B,const mr_small* rom,big a,big b,big n,int window,int nb)
{
    B->table=rom;
    B->a=a; /* just pass a pointer */
    B->b=b;
    B->n=n;
    B->window=window;  /* 2^4=16  stored values */
    B->max=nb;
}

#endif

/*
void ecn2_mul_brick(_MIPD_ ebrick *B,big e,zzn2 *x,zzn2 *y)
{
    int i,j,t,len,maxsize,promptr;
    ecn2 w,z;
 
#ifdef MR_STATIC
    char mem[MR_BIG_RESERVE(10)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    if (size_cuda(e)<0) mr_berror_cuda(_MIPP_ MR_ERR_NEG_POWER);
    t=MR_ROUNDUP(B->max,B->window);
    
    MR_IN(116)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return;
    }
#endif

    if (logb2_cuda(_MIPP_ e) > B->max)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_EXP_TOO_BIG);
        MR_OUT
        return;
    }

    ecurve_init_cuda(_MIPP_ B->a,B->b,B->n,MR_BEST);
    mr_mip->TWIST=MR_QUADRATIC;
  
#ifdef MR_STATIC
    memset(mem,0,MR_BIG_RESERVE(10));
#else
    mem=memalloc_cuda(_MIPP_ 10);
#endif

    w.x.a=mirvar_mem_cuda(_MIPP_  mem, 0);
    w.x.b=mirvar_mem_cuda(_MIPP_  mem, 1);
    w.y.a=mirvar_mem_cuda(_MIPP_  mem, 2);
    w.y.b=mirvar_mem_cuda(_MIPP_  mem, 3);  
    w.z.a=mirvar_mem_cuda(_MIPP_  mem, 4);
    w.z.b=mirvar_mem_cuda(_MIPP_  mem, 5);      
    w.marker=MR_EPOINT_INFINITY;
    z.x.a=mirvar_mem_cuda(_MIPP_  mem, 6);
    z.x.b=mirvar_mem_cuda(_MIPP_  mem, 7);
    z.y.a=mirvar_mem_cuda(_MIPP_  mem, 8);
    z.y.b=mirvar_mem_cuda(_MIPP_  mem, 9);       
    z.marker=MR_EPOINT_INFINITY;

    len=B->n->len;
    maxsize=4*(1<<B->window)*len;

    for (i=t-1;i>=0;i--)
    {
        j=recode_cuda(_MIPP_ e,t,B->window,i);
        ecn2_add_cuda(_MIPP_ &w,&w);
        if (j>0)
        {
            promptr=4*j*len;
            init_big_from_rom_cuda(z.x.a,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.x.b,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.y.a,len,B->table,maxsize,&promptr);
            init_big_from_rom_cuda(z.y.b,len,B->table,maxsize,&promptr);
            z.marker=MR_EPOINT_NORMALIZED;
            ecn2_add_cuda(_MIPP_ &z,&w);
        }
    }
    ecn2_norm_cuda(_MIPP_ &w);
    ecn2_getxy_cuda(&w,x,y);
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem,10);
#else
    memset(mem,0,MR_BIG_RESERVE(10));
#endif
    MR_OUT
}
*/

__device__ void ecn2_mul_brick_gls_cuda(_MIPD_ ebrick *B,big *e,zzn2 *psi,zzn2 *x,zzn2 *y)
{
    int i,j,k,t,len,maxsize,promptr,se[2];
    ecn2 w,z;
 
#ifdef MR_STATIC
    char mem[MR_BIG_RESERVE(10)];
#else
    char *mem;
#endif
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif

    for (k=0;k<2;k++) se[k]=exsign_cuda(e[k]);

    t=MR_ROUNDUP(B->max,B->window);
    
    MR_IN(222)

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base != mr_mip->base2)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
        MR_OUT
        return;
    }
#endif

    if (logb2_cuda(_MIPP_ e[0])>B->max || logb2_cuda(_MIPP_ e[1])>B->max)
    {
        mr_berror_cuda(_MIPP_ MR_ERR_EXP_TOO_BIG);
        MR_OUT
        return;
    }

    ecurve_init_cuda(_MIPP_ B->a,B->b,B->n,MR_BEST);
    mr_mip->TWIST=MR_QUADRATIC;
  
#ifdef MR_STATIC
    memset(mem,0,MR_BIG_RESERVE(10));
#else
    mem=memalloc_cuda(_MIPP_ 10);
#endif

    z.x.a=mirvar_mem_cuda(_MIPP_  mem, 0);
    z.x.b=mirvar_mem_cuda(_MIPP_  mem, 1);
    z.y.a=mirvar_mem_cuda(_MIPP_  mem, 2);
    z.y.b=mirvar_mem_cuda(_MIPP_  mem, 3);       
    z.marker=MR_EPOINT_INFINITY;

    w.x.a=mirvar_mem_cuda(_MIPP_  mem, 4);
    w.x.b=mirvar_mem_cuda(_MIPP_  mem, 5);
    w.y.a=mirvar_mem_cuda(_MIPP_  mem, 6);
    w.y.b=mirvar_mem_cuda(_MIPP_  mem, 7);  
    w.z.a=mirvar_mem_cuda(_MIPP_  mem, 8);
    w.z.b=mirvar_mem_cuda(_MIPP_  mem, 9); 
    w.marker=MR_EPOINT_INFINITY;

    len=B->n->len;
    maxsize=4*(1<<B->window)*len;

    for (i=t-1;i>=0;i--)
    {
        ecn2_add_cuda(_MIPP_ &w,&w);
        for (k=0;k<2;k++)
        {
            j=recode_cuda(_MIPP_ e[k],t,B->window,i);
            if (j>0)
            {
                promptr=4*j*len;
                init_big_from_rom_cuda(z.x.a,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.x.b,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.y.a,len,B->table,maxsize,&promptr);
                init_big_from_rom_cuda(z.y.b,len,B->table,maxsize,&promptr);
                z.marker=MR_EPOINT_NORMALIZED;
                if (k==1) ecn2_psi_cuda(_MIPP_ psi,&z);
                if (se[k]==PLUS) ecn2_add_cuda(_MIPP_ &z,&w);
                else             ecn2_sub_cuda(_MIPP_ &z,&w);
            }
        }      
    }
    ecn2_norm_cuda(_MIPP_ &w);
    ecn2_getxy_cuda(&w,x,y);
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem,10);
#else
    memset(mem,0,MR_BIG_RESERVE(10));
#endif
    MR_OUT
}

#endif

#ifndef MR_NO_ECC_MULTIADD

__device__ void ecn2_mult4_cuda(_MIPD_ big *e,ecn2 *P,ecn2 *R)
{ /* R=e[0]*P[0]+e[1]*P[1]+ .... e[n-1]*P[n-1]   */
    int i,j,k,l,nb,ea,c;
    ecn2 G[16];
	zzn2 work[16];
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
#ifndef MR_STATIC
	char *mem=(char *)memalloc_cuda(_MIPP_ 120);
#else
    char mem[MR_BIG_RESERVE(120)];       
 	memset(mem, 0, MR_BIG_RESERVE(120));   
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(243)

	l=0;
	for (k=1;k<16;k++)
	{
		G[k].x.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].x.b=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].y.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].y.b=mirvar_mem_cuda(_MIPP_  mem, l++); 
		G[k].z.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].z.b=mirvar_mem_cuda(_MIPP_  mem, l++);        
		G[k].marker=MR_EPOINT_INFINITY;	

		i=k; j=1; c=0; while (i>=(2*j)) {j*=2; c++;}
		if (i>j) ecn2_copy_cuda(&G[i-j],&G[k]);
		ecn2_add_cuda(_MIPP_ &P[c],&G[k]);
	}

	for (i=0;i<15;i++)
	{
		work[i].a=mirvar_mem_cuda(_MIPP_  mem, l++);  
		work[i].b=mirvar_mem_cuda(_MIPP_  mem, l++);  
	}

	ecn2_multi_norm_cuda(_MIPP_ 15,work,&G[1]);

    nb=0;
    for (j=0;j<4;j++) if ((k=logb2_cuda(_MIPP_ e[j])) > nb) nb=k;

	ecn2_zero_cuda(R);

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        for (i=nb-1;i>=0;i--)
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ea=0;
            k=1;
            for (j=0;j<4;j++)
            {
                if (mr_testbit_cuda(_MIPP_ e[j],i)) ea+=k;
                k<<=1;
            }
            ecn2_add_cuda(_MIPP_ R,R);
            if (ea!=0) ecn2_add_cuda(_MIPP_ &G[ea],R);
        }    
#ifndef MR_ALWAYS_BINARY
    }
    else mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
#endif
#ifndef MR_STATIC
    memkill_cuda(_MIPP_ mem,120);
#else
 	memset(mem, 0, MR_BIG_RESERVE(120));  
#endif

    MR_OUT
}

#ifndef MR_STATIC

__device__ void ecn2_multn_cuda(_MIPD_ int n,big *e,ecn2 *P,ecn2 *R)
{ /* R=e[0]*P[0]+e[1]*P[1]+ .... e[n-1]*P[n-1]   */
    int i,j,k,l,nb,ea,c;
	int m=1<<n;
    ecn2 *G;
	zzn2 *work;
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
	char *mem=(char *)memalloc_cuda(_MIPP_ 8*(m-1));
    if (mr_mip->ERNUM) return;

    MR_IN(223)

    G=   (ecn2 *)mr_alloc_cuda(_MIPP_ m,sizeof(ecn2));
	work=(zzn2 *)mr_alloc_cuda(_MIPP_ m,sizeof(zzn2));

	l=0;
	for (k=1;k<m;k++)
	{
		G[k].x.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].x.b=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].y.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].y.b=mirvar_mem_cuda(_MIPP_  mem, l++); 
		G[k].z.a=mirvar_mem_cuda(_MIPP_  mem, l++);
		G[k].z.b=mirvar_mem_cuda(_MIPP_  mem, l++);        
		G[k].marker=MR_EPOINT_INFINITY;	

		i=k; j=1; c=0; while (i>=(2*j)) {j*=2; c++;}
		if (i>j) ecn2_copy_cuda(&G[i-j],&G[k]);
		ecn2_add_cuda(_MIPP_ &P[c],&G[k]);
	}

	for (i=0;i<m-1;i++)
	{
		work[i].a=mirvar_mem_cuda(_MIPP_  mem, l++);  
		work[i].b=mirvar_mem_cuda(_MIPP_  mem, l++);  
	}

	ecn2_multi_norm_cuda(_MIPP_ m-1,work,&G[1]);

    nb=0;
    for (j=0;j<n;j++) if ((k=logb2_cuda(_MIPP_ e[j])) > nb) nb=k;

	ecn2_zero_cuda(R);

#ifndef MR_ALWAYS_BINARY
    if (mr_mip->base==mr_mip->base2)
    {
#endif
        for (i=nb-1;i>=0;i--)
        {
            if (mr_mip->user!=NULL) (*mr_mip->user)();
            ea=0;
            k=1;
            for (j=0;j<n;j++)
            {
                if (mr_testbit_cuda(_MIPP_ e[j],i)) ea+=k;
                k<<=1;
            }
            ecn2_add_cuda(_MIPP_ R,R);
            if (ea!=0) ecn2_add_cuda(_MIPP_ &G[ea],R);
        }    
#ifndef MR_ALWAYS_BINARY
    }
    else mr_berror_cuda(_MIPP_ MR_ERR_NOT_SUPPORTED);
#endif

    memkill_cuda(_MIPP_ mem,8*(m-1));
	mr_free_cuda(work);
    mr_free_cuda(G);
    MR_OUT
}

#endif
#endif


#endif

#ifndef mrzzn4_c
#define mrzzn4_c

#define FUNC_BASE 226

__device__ void zzn4_mirvar_cuda(_MIPD_ zzn4 *w){
    zzn2_mirvar_cuda(_MIPP_ &(w->a));
    zzn2_mirvar_cuda(_MIPP_ &(w->b));
}

__device__ void zzn4_kill_cuda(_MIPD_ zzn4 *w){
    zzn2_kill_cuda(_MIPP_ &(w->a));
    zzn2_kill_cuda(_MIPP_ &(w->b));
}

__device__ BOOL zzn4_iszero_cuda(zzn4 *x)
{
    if (zzn2_iszero_cuda(&(x->a)) && zzn2_iszero_cuda(&(x->b))) return TRUE;
    return FALSE;
}

__device__ BOOL zzn4_isunity_cuda(_MIPD_ zzn4 *x)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM || !zzn2_iszero_cuda(&(x->b))) return FALSE;

    if (zzn2_isunity_cuda(_MIPP_ &x->a)) return TRUE;
    return FALSE;
}

__device__ BOOL zzn4_compare_cuda(zzn4 *x,zzn4 *y)
{
    if (zzn2_compare_cuda(&(x->a),&(y->a)) && zzn2_compare_cuda(&(x->b),&(y->b))) return TRUE;
    return FALSE;
}

__device__ void zzn4_from_int_cuda(_MIPD_ int i,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+0)
    if (i==1) 
    {
        copy_cuda(mr_mip->one,w->a.a);
		w->unitary=TRUE;
    }
    else
    {
        convert_cuda(_MIPP_ i,mr_mip->w1);
        nres_cuda(_MIPP_ mr_mip->w1,(w->a).a);
		w->unitary=FALSE;
    }
    zero_cuda((w->a).b);
	zero_cuda((w->b).a);
	zero_cuda((w->b).b);

    MR_OUT
}

__device__ void zzn4_copy_cuda(zzn4 *x,zzn4 *w)
{
    if (x==w) return;
    zzn2_copy_cuda(&(x->a),&(w->a));
    zzn2_copy_cuda(&(x->b),&(w->b));
	w->unitary=x->unitary;
}

__device__ void zzn4_zero_cuda(zzn4 *w)
{
    zzn2_zero_cuda(&(w->a));
    zzn2_zero_cuda(&(w->b));
	w->unitary=FALSE;
}

__device__ void zzn4_from_zzn2s_cuda(zzn2 *x,zzn2 *y,zzn4 *w)
{
    zzn2_copy_cuda(x,&(w->a));
    zzn2_copy_cuda(y,&(w->b));
	w->unitary=FALSE;
}

__device__ void zzn4_from_zzn2_cuda(zzn2 *x,zzn4 *w)
{
    zzn2_copy_cuda(x,&(w->a));
    zzn2_zero_cuda(&(w->b));
	w->unitary=FALSE;
}

__device__ void zzn4_from_zzn2h_cuda(zzn2 *x,zzn4 *w)
{
    zzn2_copy_cuda(x,&(w->b));
    zzn2_zero_cuda(&(w->a));
	w->unitary=FALSE;
}

__device__ void zzn4_from_zzn_cuda(big x,zzn4 *w)
{
	zzn2_from_zzn_cuda(x,&(w->a));
	zzn2_zero_cuda(&(w->b));
	w->unitary=FALSE;
}

__device__ void zzn4_from_big_cuda(_MIPD_ big x, zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+16)
	zzn2_from_big_cuda(_MIPP_ x,&(w->a));
    zzn2_zero_cuda(&(w->b));
    MR_OUT
}

__device__ void zzn4_negate_cuda(_MIPD_ zzn4 *x,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+1)
    zzn4_copy_cuda(x,w);
    zzn2_negate_cuda(_MIPP_ &(w->a),&(w->a));
    zzn2_negate_cuda(_MIPP_ &(w->b),&(w->b));
    MR_OUT
}

__device__ void zzn4_conj_cuda(_MIPD_ zzn4 *x,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    MR_IN(FUNC_BASE+2)
    if (mr_mip->ERNUM) return;
    zzn4_copy_cuda(x,w);
    zzn2_negate_cuda(_MIPP_ &(w->b),&(w->b));
    MR_OUT
}

__device__ void zzn4_add_cuda(_MIPD_ zzn4 *x,zzn4 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+3)
    zzn2_add_cuda(_MIPP_ &(x->a),&(y->a),&(w->a));
    zzn2_add_cuda(_MIPP_ &(x->b),&(y->b),&(w->b));
	w->unitary=FALSE;
    MR_OUT
}
  
__device__ void zzn4_sadd_cuda(_MIPD_ zzn4 *x,zzn2 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+4)
    zzn2_add_cuda(_MIPP_ &(x->a),y,&(w->a));
	w->unitary=FALSE;
    MR_OUT
} 
	
__device__ void zzn4_sub_cuda(_MIPD_ zzn4 *x,zzn4 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+5)
    zzn2_sub_cuda(_MIPP_ &(x->a),&(y->a),&(w->a));
    zzn2_sub_cuda(_MIPP_ &(x->b),&(y->b),&(w->b));
	w->unitary=FALSE;

    MR_OUT
}

__device__ void zzn4_ssub_cuda(_MIPD_ zzn4 *x,zzn2 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+6)
    zzn2_sub_cuda(_MIPP_ &(x->a),y,&(w->a));
	w->unitary=FALSE;

    MR_OUT
}

__device__ void zzn4_smul_cuda(_MIPD_ zzn4 *x,zzn2 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+7)
	if (!zzn2_iszero_cuda(&(x->a))) zzn2_mul_cuda(_MIPP_ &(x->a),y,&(w->a));
	else zzn2_zero_cuda(&(w->a));
	if (!zzn2_iszero_cuda(&(x->b))) zzn2_mul_cuda(_MIPP_ &(x->b),y,&(w->b));
	else zzn2_zero_cuda(&(w->b));
	w->unitary=FALSE;

    MR_OUT
}

__device__ void zzn4_lmul_cuda(_MIPD_ zzn4 *x,big y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+15)

	if (!zzn2_iszero_cuda(&(x->a))) zzn2_smul_cuda(_MIPP_ &(x->a),y,&(w->a));
	else zzn2_zero_cuda(&(w->a));
	if (!zzn2_iszero_cuda(&(x->b))) zzn2_smul_cuda(_MIPP_ &(x->b),y,&(w->b));
	else zzn2_zero_cuda(&(w->b));
	w->unitary=FALSE;

    MR_OUT
}


__device__ void zzn4_imul_cuda(_MIPD_ zzn4 *x,int y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+14)
	zzn2_imul_cuda(_MIPP_ &(x->a),y,&(w->a));
	zzn2_imul_cuda(_MIPP_ &(x->b),y,&(w->b));

    MR_OUT
}

__device__ void zzn4_sqr_cuda(_MIPD_ zzn4 *x,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 t1,t2;
    if (mr_mip->ERNUM) return;

    MR_IN(FUNC_BASE+8)

    t1.a=mr_mip->w10;
    t1.b=mr_mip->w11;
    t2.a=mr_mip->w8;
    t2.b=mr_mip->w9;

    zzn4_copy_cuda(x,w);
    if (x->unitary)
    { /* this is a lot faster.. - see Lenstra & Stam */
        zzn2_mul_cuda(_MIPP_ &(w->b),&(w->b),&t1);
        zzn2_add_cuda(_MIPP_ &(w->b),&(w->a),&(w->b));
        zzn2_mul_cuda(_MIPP_ &(w->b),&(w->b),&(w->b));
        zzn2_sub_cuda(_MIPP_ &(w->b),&t1,&(w->b));
        zzn2_txx_cuda(_MIPP_ &t1);
        zzn2_copy_cuda(&t1,&(w->a));
        zzn2_sub_cuda(_MIPP_ &(w->b),&(w->a),&(w->b));
        zzn2_add_cuda(_MIPP_ &(w->a),&(w->a),&(w->a));
        zzn2_sadd_cuda(_MIPP_ &(w->a),mr_mip->one,&(w->a));
        zzn2_ssub_cuda(_MIPP_ &(w->b),mr_mip->one,&(w->b));
    }
    else
    {
        zzn2_copy_cuda(&(w->b),&t2); // t2=b;
        zzn2_add_cuda(_MIPP_ &(w->a),&t2,&t1); // t1=a+b

        zzn2_txx_cuda(_MIPP_ &t2);      
        zzn2_add_cuda(_MIPP_ &t2,&(w->a),&t2); // t2=a+txx(b)

        zzn2_mul_cuda(_MIPP_ &(w->b),&(w->a),&(w->b)); // b*=a
        zzn2_mul_cuda(_MIPP_ &t1,&t2,&(w->a)); // a=t1*t2

        zzn2_copy_cuda(&(w->b),&t2); //t2=b
        zzn2_sub_cuda(_MIPP_ &(w->a),&t2,&(w->a)); //a-=b      
        zzn2_txx_cuda(_MIPP_ &t2); // t2=txx(b)
        zzn2_sub_cuda(_MIPP_ &(w->a),&t2,&(w->a)); // a-=txx(b);
        zzn2_add_cuda(_MIPP_ &(w->b),&(w->b),&(w->b)); // b+=b;

    }

    MR_OUT
}    

__device__ void zzn4_mul_cuda(_MIPD_ zzn4 *x,zzn4 *y,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    zzn2 t1,t2,t3;
    if (mr_mip->ERNUM) return;
	if (x==y) {zzn4_sqr_cuda(_MIPP_ x,w); return; }
    MR_IN(FUNC_BASE+9)

    t1.a=mr_mip->w12;
    t1.b=mr_mip->w13;
    t2.a=mr_mip->w8;
    t2.b=mr_mip->w9;
    t3.a=mr_mip->w10;
    t3.b=mr_mip->w11;
    zzn2_copy_cuda(&(x->a),&t1);
    zzn2_copy_cuda(&(x->b),&t2);
    zzn2_mul_cuda(_MIPP_ &t1,&(y->a),&t1);   /* t1= x->a * y->a */
    zzn2_mul_cuda(_MIPP_ &t2,&(y->b),&t2);   /* t2 = x->b * y->b */
    zzn2_copy_cuda(&(y->a),&t3);
    zzn2_add_cuda(_MIPP_ &t3,&(y->b),&t3);   /* y->a + y->b */

    zzn2_add_cuda(_MIPP_ &(x->b),&(x->a),&(w->b)); /* x->a + x->b */
    zzn2_mul_cuda(_MIPP_ &(w->b),&t3,&(w->b));     /* t3= (x->a + x->b)*(y->a + y->b) */
    zzn2_sub_cuda(_MIPP_ &(w->b),&t1,&(w->b));
    zzn2_sub_cuda(_MIPP_ &(w->b),&t2,&(w->b));     /*  w->b = t3-(t1+t2) */
    zzn2_copy_cuda(&t1,&(w->a));
    zzn2_txx_cuda(_MIPP_ &t2);
    zzn2_add_cuda(_MIPP_ &(w->a),&t2,&(w->a));	/* w->a = t1+tx(t2) */
    if (x->unitary && y->unitary) w->unitary=TRUE;
    else w->unitary=FALSE;

    MR_OUT
}

__device__ void zzn4_inv_cuda(_MIPD_ zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
	zzn2 t1,t2;
    if (mr_mip->ERNUM) return;
    if (w->unitary)
    {
        zzn4_conj_cuda(_MIPP_ w,w);
        return;
    }
	MR_IN(FUNC_BASE+10)

    t1.a=mr_mip->w8;
    t1.b=mr_mip->w9;
    t2.a=mr_mip->w10;
    t2.b=mr_mip->w11; 
    zzn2_mul_cuda(_MIPP_ &(w->a),&(w->a),&t1);
    zzn2_mul_cuda(_MIPP_ &(w->b),&(w->b),&t2);
    zzn2_txx_cuda(_MIPP_ &t2);
    zzn2_sub_cuda(_MIPP_ &t1,&t2,&t1);
    zzn2_inv_cuda(_MIPP_ &t1);
    zzn2_mul_cuda(_MIPP_ &(w->a),&t1,&(w->a));
    zzn2_negate_cuda(_MIPP_ &t1,&t1);
    zzn2_mul_cuda(_MIPP_ &(w->b),&t1,&(w->b));

	MR_OUT
}

/* divide_cuda zzn4 by 2 */

__device__ void zzn4_div2_cuda(_MIPD_ zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    MR_IN(FUNC_BASE+11)

    zzn2_div2_cuda(_MIPP_ &(w->a));
    zzn2_div2_cuda(_MIPP_ &(w->b));
	w->unitary=FALSE;

    MR_OUT
}

__device__ void zzn4_powq_cuda(_MIPD_ zzn2 *fr,zzn4 *w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
	MR_IN(FUNC_BASE+12)
	zzn2_conj_cuda(_MIPP_ &(w->a),&(w->a));
    zzn2_conj_cuda(_MIPP_ &(w->b),&(w->b));
	zzn2_mul_cuda(_MIPP_ &(w->b),fr,&(w->b));
	MR_OUT
}

__device__ void zzn4_tx_cuda(_MIPD_ zzn4* w)
{
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
	zzn2 t;	
	MR_IN(FUNC_BASE+13)

    t.a=mr_mip->w8;
    t.b=mr_mip->w9;
	zzn2_copy_cuda(&(w->b),&t);
	zzn2_txx_cuda(_MIPP_ &t);
	zzn2_copy_cuda(&(w->a),&(w->b));
	zzn2_copy_cuda(&t,&(w->a));

	MR_OUT
}


#endif

#ifndef sm3_c
#define sm3_c

#include <string.h>
//11sm3.c


#define SM3_DIGEST_LENGTH  32
#define SM3_BLOCK_LENGTH   64
#define SM3_PAD_MAXLEN     56

typedef          long long int64;
typedef unsigned long long uint64;
typedef          int int32;
typedef unsigned int uint32;

#define rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

#define P0(X)  (X^rol(X,9)^rol(X,17))
#define P1(X)  (X^rol(X,15)^rol(X,23))

#define FF_00_15(X,Y,Z)  (X^Y^Z)
#define FF_16_63(X,Y,Z)  ((X&Y)|(Y&Z)|(Z&X))

#define GG_00_15(X,Y,Z)  (X^Y^Z)
#define GG_16_63(X,Y,Z)  ((X&Y)|(( ~X ) & Z))

static const uint32 T_00_15 = 0x79cc4519;
static const uint32 T_16_63 = 0x7a879d8a;

#ifdef VERBOSE  /* SAK */
__device__ void SM3PrintContext_cuda(SM3_CTX *context, char *msg){
    printf("%s (%d,%d) %x %x %x %x %x %x %x %x\n",
           msg,
           context->count[0], context->count[1],
           context->state[0],
           context->state[1],
           context->state[2],
           context->state[3],
           context->state[4],
           context->state[5],
           context->state[6],
           context->state[7]);
}
#endif

/* Hash a single 512-bit block. This is the core of the algorithm. */

__device__ void SM3Transform_cuda(SM3_CTX *context, unsigned char buffer[64])
{
    uint32 a, b, c, d, e, f, g_cuda, h;
    uint32 W[68];
    uint32 W1[64];
    uint32 SS1, SS2, TT1, TT2;
    register int j;
    unsigned char *datap;
    unsigned int *pState = &context->state[0];
    
    //	const uint32 *B=(const uint32 *)buffer;
    
    j = 0;
    datap = buffer;
    do {
        W[j] = (((uint32)(datap[0]))<<24) | (((uint32)(datap[1]))<<16) |
        (((uint32)(datap[2]))<<8 ) | ((uint32)(datap[3]));
        datap += 4;
    } while(++j < 16);
    
    for( j = 16;j<68;j++)
    {
        W[j] = P1( W[j-16]^W[j-9]^rol(W[j-3],15) ) ^ rol(W[j-13], 7) ^W[j-6];
    }
    for( j = 0;j<64;j++)
    {
        W1[j] = W[j]^W[j+4];
    }
    
    /* Copy pState[] to working vars */
    a = pState[0];
    b = pState[1];
    c = pState[2];
    d = pState[3];
    e = pState[4];
    f = pState[5];
    g_cuda = pState[6];
    h = pState[7];
    
    for(j=0;j<16;j++)
    {
        SS1 = rol(a, 12) + e +rol(T_00_15, j);
        SS1 = rol( SS1, 7);
        SS2 = SS1 ^ rol(a, 12);
        TT1 = FF_00_15(a,b,c) + d + SS2 + W1[j];
        TT2 = GG_00_15(e,f,g_cuda) + h + SS1 + W[j];
        d = c;
        c = rol(b,9);
        b = a;
        a = TT1;
        h = g_cuda;
        g_cuda = rol(f,19);
        f = e;
        e = P0(TT2);
    }
    for(j=16;j<64;j++)
    {
        SS1 = rol(a, 12) + e + rol(T_16_63, j);
        SS1 = rol( SS1, 7);
        SS2 = SS1 ^ rol(a, 12);
        TT1 = FF_16_63(a,b,c) + d + SS2 + W1[j];
        TT2 = GG_16_63(e,f,g_cuda) + h + SS1 + W[j];
        d = c;
        c = rol(b,9);
        b = a;
        a = TT1;
        h = g_cuda;
        g_cuda = rol(f,19);
        f = e;
        e = P0(TT2);
    }
    
    /* Add the working vars back into pState[] */
    pState[0] ^= a;
    pState[1] ^= b;
    pState[2] ^= c;
    pState[3] ^= d;
    pState[4] ^= e;
    pState[5] ^= f;
    pState[6] ^= g_cuda;
    pState[7] ^= h;
    /* Wipe variables */
    a = b = c = d = e = f = g_cuda = h = 0;
}


/* SM3Init_cuda - Initialize new context */

__device__ void SM3Init_cuda(SM3_CTX* context)
{
    /* SM3 initialization constants */
    context->state[0] = 0x7380166f;
    context->state[1] = 0x4914b2b9;
    context->state[2] = 0x172442d7;
    context->state[3] = 0xda8a0600;
    context->state[4] = 0xa96f30bc;
    context->state[5] = 0x163138aa;
    context->state[6] = 0xe38dee4d;
    context->state[7] = 0xb0fb0e4e;
    context->tc = 0;
    context->bc = 0;
}


/* Run your data through this. */

__device__ void SM3Update_cuda(SM3_CTX* ctx, unsigned char* datap, uint32 len)
{
#ifdef VERBOSE
    SM3PrintContext(ctx, "before");
#endif
    //if ( ctx->bc == (ctx->tc % 64) ){printf("before sm3 bc = tc mod 64\n");}else{printf("end error\n");}
    ctx->tc += len;
    
    
    
    while(len > 0) {
        if(!ctx->bc) {
            while(len >= SM3_BLOCK_LENGTH) {
                SM3Transform_cuda(ctx, datap);
                datap += SM3_BLOCK_LENGTH;
                len -= SM3_BLOCK_LENGTH;
            }
            if(!len) return;
        }
        ctx->buffer[ctx->bc] = *datap++;
        len--;
        if(++ctx->bc == SM3_BLOCK_LENGTH) {
            SM3Transform_cuda(ctx, &ctx->buffer[0]);
            ctx->bc = 0;
        }
    }
    //if ( ctx->bc == (ctx->tc % 64) ){printf("after sm3 bc = tc mod 64\n");}else{printf("end error\n");}
#ifdef VERBOSE
    SM3PrintContext(ctx, "after ");
#endif
}


/* Add padding and return the message digest. */

__device__ void SM3Final_cuda(unsigned char digest[32], SM3_CTX* ctx)
{
    uint32 i;
    uint64 bitLength;
    register int    j;
    unsigned char   *datap;
    
    ctx->buffer[ctx->bc++] = 0x80;
    if( ctx->bc > SM3_PAD_MAXLEN) {
        while(ctx->bc < SM3_BLOCK_LENGTH) {
            ctx->buffer[ctx->bc++] = 0x00;
        }
        SM3Transform_cuda(ctx, &ctx->buffer[0]);
        ctx->bc = 0;
    }
    
    while(ctx->bc < SM3_PAD_MAXLEN) {
        ctx->buffer[ctx->bc++] = 0x00;
    }
    if(ctx->bc == SM3_PAD_MAXLEN) {
        bitLength = ctx->tc*8;
        ctx->buffer[56] = (unsigned char)(bitLength >> 56);
        ctx->buffer[57] = (unsigned char)(bitLength >> 48);
        ctx->buffer[58] = (unsigned char)(bitLength >> 40);
        ctx->buffer[59] = (unsigned char)(bitLength >> 32);
        ctx->buffer[60] = (unsigned char)(bitLength >> 24);
        ctx->buffer[61] = (unsigned char)(bitLength >> 16);
        ctx->buffer[62] = (unsigned char)(bitLength >> 8);
        ctx->buffer[63] = (unsigned char)bitLength;
    }
    SM3Transform_cuda(ctx, &ctx->buffer[0]);
    ctx->bc = 0;
    
    datap = digest;
    j = 0;
    do {
        i = ctx->state[j];
        datap[0] = i >> 24;
        datap[1] = i >> 16;
        datap[2] = i >> 8;
        datap[3] = i;
        datap += 4;
    } while(++j < 8);
    //memset(ctx, 0, sizeof *ctx);		
}

__device__ void SM3_hash_buffer_cuda(unsigned char *ib, int ile, unsigned char *ob, int ole)
{
    SM3_CTX ctx;
    
    if(ole < 1) return;
    //if(ob == NULL) return;
    
    memset(ob, 0, ole);
    if(ole < 32) return;
    
    if(ole > 32) ole = 32;
    SM3Init_cuda(&ctx);
    SM3Update_cuda(&ctx, ib, ile);
    SM3Final_cuda(ob, &ctx);
    
    //memset(&ctx, 0, sizeof(ctx));
}

__device__ void SM3_cuda(unsigned char* data, unsigned int len, unsigned char digest[32]){
    SM3_CTX sm3, *psm3;
    psm3 = &sm3;
    SM3Init_cuda( psm3 );
    SM3Update_cuda( psm3, data, len);
    SM3Final_cuda(digest,psm3);
}


#endif

#ifndef sm_r_ate_c
#define sm_r_ate_c

//
// R-ate Pairing Code
//


__device__ void set_frobenius_constant_cuda(_MIPD_ zzn2 *x){
 //   miracl *mip=get_mip_cuda();
 //   copy_cuda(mip->modulus,m.fn);
    big ONE = mirvar_cuda(_MIPP_ 1);
    big ZERO = mirvar_cuda(_MIPP_ 0);
    big TWO = mirvar_cuda(_MIPP_ 2);
    big sx = mirvar_cuda(_MIPP_ 6);
    big sxx = mirvar_cuda(_MIPP_ 0);
    big p = mirvar_cuda(_MIPP_ 0);
    

    copy_cuda(mr_mip->modulus, p);

#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    switch (mr_mip->pmod8)
    {
        case 5:
            zzn2_from_bigs_cuda(_MIPP_ ZERO, ONE, x);
            break;
        case 3:
            zzn2_from_bigs_cuda(_MIPP_ ONE, ONE, x);
            break;
        case 7:
            zzn2_from_bigs_cuda(_MIPP_ TWO, ONE, x);
            break;
        default:
            break;
    }
    xgcd_cuda(_MIPP_ sx, p, sx, sx, sx);
    decr_cuda(_MIPP_ p, 1, p);
    multiply_cuda(_MIPP_ sx, p, sxx);
    
    copy_cuda(mr_mip->modulus, p);
    divide_cuda(_MIPP_ sxx,p,p);
    

	zzn2_pow_cuda(_MIPP_ x, sxx, x);
    
    

    mirkill_cuda(ONE);
    mirkill_cuda(ZERO);
    mirkill_cuda(TWO);
    mirkill_cuda(sx);
    mirkill_cuda(p);
}

//
// Line from A to destination C. Let A=(x,y)
// Line Y-slope.X-c=0, through A, so intercept c=y-slope.x
// Line Y-slope.X-y+slope.x = (Y-y)-slope.(X-x) = 0
// Now evaluate at Q -> return (Qy-y)-slope.(Qx-x)
//

__device__ void line_cuda(_MIPD_ ecn2 *A,ecn2 *C,ecn2 *B,zzn2 *slope,zzn2 *extra,BOOL Doubling,zzn Qx,zzn Qy,zzn12 *wt)
{
    zzn12 w;
    
    zzn4 nn,dd,cc;
    zzn2 X,Y;
    zzn2 Z3,txx;
    zzn2 w1,w2,w3;
    
    zzn12_mirvar_cuda(_MIPP_ &w);
    zzn4_mirvar_cuda(_MIPP_ &nn);
    zzn4_mirvar_cuda(_MIPP_ &dd);
    zzn4_mirvar_cuda(_MIPP_ &cc);
    zzn2_mirvar_cuda(_MIPP_ &X);
    zzn2_mirvar_cuda(_MIPP_ &Y);
    zzn2_mirvar_cuda(_MIPP_ &Z3);
    zzn2_mirvar_cuda(_MIPP_ &w1);
    zzn2_mirvar_cuda(_MIPP_ &w2);
    zzn2_mirvar_cuda(_MIPP_ &w3);
    zzn2_mirvar_cuda(_MIPP_ &txx);
    
    ecn2_getz_cuda(_MIPP_ C, &Z3);
    
    
    if (Doubling)
    {
        zzn2 Z,ZZ;
        zzn2_mirvar_cuda(_MIPP_ &Z);
        zzn2_mirvar_cuda(_MIPP_ &ZZ);
        ecn2_get_cuda(_MIPP_ A, &X, &Y, &Z);
        zzn2_copy_cuda(&Z, &ZZ);
        zzn2_mul_cuda(_MIPP_ &ZZ, &ZZ, &ZZ);
        if (mr_mip->TWIST==MR_SEXTIC_M)
        { // "multiplied across" by i to simplify
            
            zzn2_mul_cuda(_MIPP_ &ZZ, &Z3, &w1); //
            zzn2_from_zzn_cuda((big)Qy, &txx);
            zzn2_txx_cuda(_MIPP_ &txx);
            zzn2_mul_cuda(_MIPP_ &w1, &txx, &w1);
            
            zzn2_mul_cuda(_MIPP_ slope, &X, &w2);
            zzn2_sub_cuda(_MIPP_ &w2, extra, &w2);
            
            zzn4_from_zzn2s_cuda(&w1, &w2, &nn);
            //nn.set((Z3*ZZ)*txx((ZZn2)Qy),slope*X-extra);

            zzn2_mul_cuda(_MIPP_ &ZZ,slope, &w1);
            zzn2_from_zzn_cuda((big)Qx, &w2);
            zzn2_mul_cuda(_MIPP_ &w1,&w2, &w1);
            zzn2_negate_cuda(_MIPP_ &w1, &w1);
            
            zzn4_from_zzn2h_cuda(&w1, &cc);
                        //cc.seth(-(ZZ*slope)*Qx);
        }
        if (mr_mip->TWIST==MR_SEXTIC_D)
        {
            //todo
          //  printf("to do \n");
            
            //nn.set((Z3*ZZ)*Qy,slope*X-extra);
            //dd.set(-(ZZ*slope)*Qx);
        }
        zzn2_kill_cuda(_MIPP_ &Z);
        zzn2_kill_cuda(_MIPP_ &ZZ);
    }
    else
    {
        zzn2 X2,Y2;
        zzn2_mirvar_cuda(_MIPP_ &X2);
        zzn2_mirvar_cuda(_MIPP_ &Y2);
        
        ecn2_getxy_cuda(B, &X2, &Y2);
        
        if (mr_mip->TWIST==MR_SEXTIC_M)
        {
            zzn2_from_zzn_cuda((big)Qy, &txx);
            zzn2_txx_cuda(_MIPP_ &txx);
            zzn2_mul_cuda(_MIPP_ &Z3, &txx, &w1);
            zzn2_mul_cuda(_MIPP_ slope, &X2, &w2);
            zzn2_mul_cuda(_MIPP_ &Y2, &Z3, &w3);
            zzn2_sub_cuda(_MIPP_ &w2, &w3, &w2);
            zzn4_from_zzn2s_cuda(&w1, &w2, &nn);
            //nn.set(Z3*txx((ZZn2)Qy),slope*X2-Y2*Z3);
            
            zzn2_from_zzn_cuda((big)Qx, &w1);
            zzn2_mul_cuda(_MIPP_ slope, &w1, &w1);
            zzn2_negate_cuda(_MIPP_ &w1, &w1);
            zzn4_from_zzn2h_cuda(&w1, &cc);
            
            //cc.seth(-slope*Qx);
        }
        if (mr_mip->TWIST==MR_SEXTIC_D)
        {
         //   printf("to do \n");
            // todo
            // nn.set(Z3*Qy,slope*X2-Y2*Z3);
           // dd.set(-slope*Qx);
        }
        zzn2_kill_cuda(_MIPP_ &X2);
        zzn2_kill_cuda(_MIPP_ &Y2);
    }
    zzn12_from_zzn4s_cuda(&nn, &dd, &cc, &w);
    //w.set(nn,dd,cc);
    
    zzn12_copy_cuda(&w, wt);
    zzn12_kill_cuda(_MIPP_ &w);
    zzn4_kill_cuda(_MIPP_ &nn);
    zzn4_kill_cuda(_MIPP_ &dd);
    zzn4_kill_cuda(_MIPP_ &cc);
    zzn2_kill_cuda(_MIPP_ &X);
    zzn2_kill_cuda(_MIPP_ &Y);
    zzn2_kill_cuda(_MIPP_ &Z3);
    zzn2_kill_cuda(_MIPP_ &w1);
    zzn2_kill_cuda(_MIPP_ &w2);
    zzn2_kill_cuda(_MIPP_ &w3);
    zzn2_kill_cuda(_MIPP_ &txx);
    return ;
}

__device__ void endomorph_cuda(_MIPD_ ecn *A,zzn *Beta)
{ // apply endomorphism (x,y) = (Beta*x,y) where Beta is cube root of unity
    big x;
    x = mirvar_cuda(_MIPP_ 0);
    copy_cuda(A->X,x);
    multiply_cuda(_MIPP_ x, (big) Beta, x);
    divide_cuda(_MIPP_ x,mr_mip->modulus,mr_mip->modulus);
    copy_cuda(x, A->X);
    mirkill_cuda(x);
    
    
    //ZZn x;
    //x=(A.get_point())->X;
    //x*=Beta;
    //copy_cuda(getbig(x),(A.get_point())->X);
}

__device__ void q_power_frobenius_cuda(_MIPD_ ecn2 *A,zzn2 *F)
{// apply endomorphism (x,y) = (Beta*x,y) where Beta is cube root of unity
    zzn2 x,y,z,w,r,cj;
    zzn2_mirvar_cuda(_MIPP_ &x);
    zzn2_mirvar_cuda(_MIPP_ &y);
    zzn2_mirvar_cuda(_MIPP_ &z);
    zzn2_mirvar_cuda(_MIPP_ &w);
    zzn2_mirvar_cuda(_MIPP_ &r);
    zzn2_mirvar_cuda(_MIPP_ &cj);
    
    ecn2_get_cuda(_MIPP_ A, &x, &y, &z);
    
    
    
    zzn2_copy_cuda(F, &r);
    
    if (mr_mip->TWIST==MR_SEXTIC_M) zzn2_inv_cuda(_MIPP_ &r);  // could be precalculated
    
    zzn2_mul_cuda(_MIPP_ &r, &r, &w);
    
    zzn2_conj_cuda(_MIPP_ &x, &cj);
    zzn2_mul_cuda(_MIPP_ &cj, &w, &x);
    
    
    zzn2_conj_cuda(_MIPP_ &y, &cj);
    zzn2_mul_cuda(_MIPP_ &cj, &w, &y);
    zzn2_mul_cuda(_MIPP_ &y, &r, &y);
    zzn2_conj_cuda(_MIPP_ &z, &z);
    
    ecn2_setxyz_cuda(_MIPP_ &x, &y, &z, A);
    //ZZn2 x,y,z,w,r;
    
    //A.get(x,y,z);
    //w=F*F;
    //r=F;
    //x=w*conj(x);
    //y=r*w*conj(y);
    //z.conj();
    //A.set(x,y,z);


    zzn2_kill_cuda(_MIPP_ &x);
    zzn2_kill_cuda(_MIPP_ &y);
    zzn2_kill_cuda(_MIPP_ &z);
    zzn2_kill_cuda(_MIPP_ &w);
    zzn2_kill_cuda(_MIPP_ &r);
    zzn2_kill_cuda(_MIPP_ &cj);
}


//
// Add A=A+B  (or A=A+A)
// Return line_cuda function value
//
__device__ void g_cuda(_MIPD_ ecn2 *A,ecn2 *B,zzn Qx,zzn Qy, zzn12 *wr)
{
    zzn2 lam,extra;
    zzn12 r;
    ecn2 P;//=A;
    BOOL Doubling;
  
    zzn2_mirvar_cuda(_MIPP_ &lam);
    zzn2_mirvar_cuda(_MIPP_ &extra);
    zzn12_mirvar_cuda(_MIPP_ &r);
    ecn2_mirvar_cuda(_MIPP_ &P);
    
    ecn2_copy_cuda(A, &P);
    
// Evaluate line_cuda from A
    
//    Doubling=A.add_cuda(B,lam,extra);
//    if (A.iszero())   return (ZZn12)1;
//    r=line_cuda(P,A,B,lam,extra,Doubling,Qx,Qy);
    
    Doubling = ecn2_add2_cuda(_MIPP_ B, A, &lam, &extra);
    
    if (ecn2_iszero_cuda(A)) {
        zzn12_from_int_cuda(_MIPP_ 1, &r);
    }else{
        line_cuda(_MIPP_ &P,A,B,&lam, &extra, Doubling, Qx, Qy, &r);
    }
    
    zzn12_copy_cuda(&r, wr);
    
    zzn2_kill_cuda(_MIPP_ &lam);
    zzn2_kill_cuda(_MIPP_ &extra);
    ecn2_kill_cuda(_MIPP_ &P);
    zzn12_kill_cuda(_MIPP_ &r);
    return ;
}


//
// R-ate Pairing G2 x G1 -> GT
//
// P is a point of order q in G1. Q(x,y) is a point of order q in G2.
// Note that P is a point on the sextic twist of the curve over Fp^2, Q(x,y) is a point on the
// curve over the base field Fp
//
#include <time.h>

__device__ BOOL fast_pairing_cuda(_MIPD_ ecn2 *P,zzn Qx,zzn Qy,big x,zzn2 *X,zzn12 *res)
{
    static double total_time = 0;
    //clock_t start=clock(), finish;

    ecn2 A, KA;
    zzn2 AX, AY;
    int i, nb;
    big n, ZERO;
    zzn12 r;
    zzn12 t0, t1;
    zzn12 x0, x1, x2, x3, x4, x5;
    zzn12 w1, w2;

    ecn2_mirvar_cuda(_MIPP_ &A);
    ecn2_mirvar_cuda(_MIPP_ &KA);
    zzn2_mirvar_cuda(_MIPP_ &AX);
    zzn2_mirvar_cuda(_MIPP_ &AY);
    n = mirvar_cuda(_MIPP_ 0);
    ZERO = mirvar_cuda(_MIPP_ 0);
    zzn12_mirvar_cuda(_MIPP_ &r);
    zzn12_mirvar_cuda(_MIPP_ &t0);
    zzn12_mirvar_cuda(_MIPP_ &t1);
    zzn12_mirvar_cuda(_MIPP_ &x0);
    zzn12_mirvar_cuda(_MIPP_ &x1);
    zzn12_mirvar_cuda(_MIPP_ &x2);
    zzn12_mirvar_cuda(_MIPP_ &x3);
    zzn12_mirvar_cuda(_MIPP_ &x4);
    zzn12_mirvar_cuda(_MIPP_ &x5);
    zzn12_mirvar_cuda(_MIPP_ &w1);
    zzn12_mirvar_cuda(_MIPP_ &w2);

    /*    if (x<0) n=-(6*x+2);
    else n=6*x+2;
    A=P;
    nb=bits(n);
    r=1;
    // Short Miller loop
    r.mark_as_miller();
    */

    if ((mr_compare_cuda(x, ZERO)) == -1) {
        premult_cuda(_MIPP_ x, 6, n);
        incr_cuda(_MIPP_ n, 2, n);
        negify_cuda(n, n);
    } else {
        premult_cuda(_MIPP_ x, 6, n);
        incr_cuda(_MIPP_ n, 2, n);
    }

    ecn2_copy_cuda(P, &A);
    nb = logb2_cuda(_MIPP_ n);

    zzn12_from_int_cuda(_MIPP_ 1, &r);
    zzn12_mark_miller_cuda(_MIPP_ &r);
    /*
     for (i=nb-2;i>=0;i--)
     {
     r*=r;
     r*=g_cuda(A,A,Qx,Qy);
     if (bit(n,i))
     r*=g_cuda(A,P,Qx,Qy);
     }
     */
    for (i = nb - 2; i >= 0; i--) {
        zzn12_mul_cuda(_MIPP_ &r, &r, &r);
        g_cuda(_MIPP_ &A, &A, Qx, Qy, &t0);
        zzn12_mul_cuda(_MIPP_ &r, &t0, &r);
        if (mr_testbit_cuda(_MIPP_ n, i)) {
            g_cuda(_MIPP_ &A, P, Qx, Qy, &t0);
            zzn12_mul_cuda(_MIPP_ &r, &t0, &r);
        }
    }
    // Combining ideas due to Longa, Aranha et al. and Naehrig
    /*    KA=P;
    q_power_frobenius_cuda(KA,X);
    if (x<0) {A=-A; r.conj();}
    r*=g_cuda(A,KA,Qx,Qy);
    q_power_frobenius_cuda(KA,X); KA=-KA;
    r*=g_cuda(A,KA,Qx,Qy);
    */
    ecn2_copy_cuda(P, &KA);
    q_power_frobenius_cuda(_MIPP_ &KA, X);
    if ((mr_compare_cuda(x, ZERO)) < 0) {
        ecn2_negate_cuda(_MIPP_ &A, &A);
        zzn12_conj_cuda(_MIPP_ &r, &r);
    }
    g_cuda(_MIPP_ &A, &KA, Qx, Qy, &t0);
    zzn12_mul_cuda(_MIPP_ &r, &t0, &r);
    q_power_frobenius_cuda(_MIPP_ &KA, X);
    ecn2_negate_cuda(_MIPP_ &KA, &KA);
    g_cuda(_MIPP_ &A, &KA, Qx, Qy, &t0);
    zzn12_mul_cuda(_MIPP_ &r, &t0, &r);



//    if (r.iszero()) return FALSE;
    if (zzn12_iszero_cuda(_MIPP_ &r)) {
        ecn2_kill_cuda(_MIPP_ &A);
        ecn2_kill_cuda(_MIPP_ &KA);
        zzn2_kill_cuda(_MIPP_ &AX);
        zzn2_kill_cuda(_MIPP_ &AY);
        mirkill_cuda(n);
        mirkill_cuda(ZERO);
        zzn12_kill_cuda(_MIPP_ &r);
        zzn12_kill_cuda(_MIPP_ &t0);
        zzn12_kill_cuda(_MIPP_ &t1);
        zzn12_kill_cuda(_MIPP_ &x0);
        zzn12_kill_cuda(_MIPP_ &x1);
        zzn12_kill_cuda(_MIPP_ &x2);
        zzn12_kill_cuda(_MIPP_ &x3);
        zzn12_kill_cuda(_MIPP_ &x4);
        zzn12_kill_cuda(_MIPP_ &x5);
        zzn12_kill_cuda(_MIPP_ &w1);
        zzn12_kill_cuda(_MIPP_ &w2);
        return FALSE;
    }

    // The final exponentiation

    /*    t0=r;

    r.conj();

    r/=t0;    // r^(p^6-1)
    r.mark_as_regular();  // no longer "miller"

    t0=r;
    r.powq(X); r.powq(X);
    r*=t0;    // r^[(p^6-1)*(p^2+1)]

    r.mark_as_unitary();  // from now on all inverses are just conjugates !! (and squarings are faster)

    res=r;*/

    zzn12_copy_cuda(&r, &t0);
    zzn12_conj_cuda(_MIPP_ &r, &r);
    zzn12_inv_cuda(_MIPP_ &t0);
    zzn12_mul_cuda(_MIPP_ &r, &t0, &r);
    zzn12_mark_regular_cuda(_MIPP_ &r);

    zzn12_copy_cuda(&r, &t0);
    zzn12_powq_cuda(_MIPP_ &r, X);
    zzn12_powq_cuda(_MIPP_ &r, X);
    zzn12_mul_cuda(_MIPP_ &r, &t0, &r);
    zzn12_mark_unitary_cuda(_MIPP_ &r);

    zzn12_copy_cuda(&r, res);

    // Newer new idea...
    // See "On the final exponentiation for calculating pairings on ordinary elliptic curves"
    // Michael Scott and Naomi Benger and Manuel Charlemagne and Luis J. Dominguez Perez and Ezekiel J. Kachisa
    /*   t0=res;    t0.powq(X);
       x0=t0;   x0.powq(X);

       x0*=(res*t0);
       x0.powq(X);

       x1=inverse(res);  // just a conjugation!

       x4=pow(res,-x);  // x is sparse..
       x3=x4; x3.powq(X);
   */
    zzn12_copy_cuda(res, &t0);
    zzn12_powq_cuda(_MIPP_ &t0, X);
    zzn12_copy_cuda(&t0, &x0);
    zzn12_powq_cuda(_MIPP_ &x0, X);

    zzn12_mul_cuda(_MIPP_ res, &t0, &w1);
    zzn12_mul_cuda(_MIPP_ &x0, &w1, &x0);
    zzn12_powq_cuda(_MIPP_ &x0, X);

    zzn12_copy_cuda(res, &x1);
    zzn12_inv_cuda(_MIPP_ &x1);

    zzn12_copy_cuda(res, &x4);
    zzn12_pow_cuda(_MIPP_ &x4, x);
    zzn12_inv_cuda(_MIPP_ &x4);

    zzn12_copy_cuda(&x4, &x3);
    zzn12_powq_cuda(_MIPP_ &x3, X);


    /*
    x2=pow(x4,-x);
    x5=inverse(x2);
    t0=pow(x2,-x);

    x2.powq(X);
    x4/=x2;

    x2.powq(X);

    res=t0; res.powq(X); t0*=res;
    */
    zzn12_copy_cuda(&x4, &x2);
    zzn12_pow_cuda(_MIPP_ &x2, x);
    zzn12_copy_cuda(&x2, &x5);
    zzn12_copy_cuda(&x2, &t0);
    zzn12_inv_cuda(_MIPP_ &x2);
    zzn12_pow_cuda(_MIPP_ &t0, x);

    zzn12_powq_cuda(_MIPP_ &x2, X);

    zzn12_copy_cuda(&x2, &w1);
    zzn12_inv_cuda(_MIPP_ &w1);
    zzn12_mul_cuda(_MIPP_ &x4, &w1, &x4);

    zzn12_powq_cuda(_MIPP_ &x2, X);

    zzn12_copy_cuda(&t0, res);
    zzn12_powq_cuda(_MIPP_ res, X);

    zzn12_mul_cuda(_MIPP_ &t0, res, &t0);

    /*
    t0*=t0;
    t0*=x4;
    t0*=x5;

    res=x3*x5;
    res*=t0;
    t0*=x2;

    res*=res;
    res*=t0;
    res*=res;

    t0=res*x1;
    res*=x0;
    t0*=t0;
    t0*=res;
    */
    zzn12_mul_cuda(_MIPP_ &t0, &t0, &t0);
    zzn12_mul_cuda(_MIPP_ &t0, &x4, &t0);
    zzn12_mul_cuda(_MIPP_ &t0, &x5, &t0);

    zzn12_mul_cuda(_MIPP_ &x3, &x5, res);
    zzn12_mul_cuda(_MIPP_ res, &t0, res);
    zzn12_mul_cuda(_MIPP_ &t0, &x2, &t0);

    zzn12_mul_cuda(_MIPP_ res, res, res);
    zzn12_mul_cuda(_MIPP_ res, &t0, res);
    zzn12_mul_cuda(_MIPP_ res, res, res);

    zzn12_mul_cuda(_MIPP_ res, &x1, &t0);
    zzn12_mul_cuda(_MIPP_ res, &x0, res);
    zzn12_mul_cuda(_MIPP_ &t0, &t0, &t0);
    zzn12_mul_cuda(_MIPP_ &t0, res, &t0);

    zzn12_copy_cuda(&t0, res);

    ecn2_kill_cuda(_MIPP_ &A);
    ecn2_kill_cuda(_MIPP_ &KA);
    zzn2_kill_cuda(_MIPP_ &AX);
    zzn2_kill_cuda(_MIPP_ &AY);
    mirkill_cuda(n);
    mirkill_cuda(ZERO);
    zzn12_kill_cuda(_MIPP_ &r);
    zzn12_kill_cuda(_MIPP_ &t0);
    zzn12_kill_cuda(_MIPP_ &t1);
    zzn12_kill_cuda(_MIPP_ &x0);
    zzn12_kill_cuda(_MIPP_ &x1);
    zzn12_kill_cuda(_MIPP_ &x2);
    zzn12_kill_cuda(_MIPP_ &x3);
    zzn12_kill_cuda(_MIPP_ &x4);
    zzn12_kill_cuda(_MIPP_ &x5);
    zzn12_kill_cuda(_MIPP_ &w1);
    zzn12_kill_cuda(_MIPP_ &w2);

    //finish = clock();
    //double time = (double)(finish - start) / (double)CLOCKS_PER_SEC;
    //total_time += time*1000;
    //printf("%lf ms\n", total_time);
    return TRUE;
}

//
// ecap_cuda(.) function
//

__device__ BOOL ecap_cuda(_MIPD_ ecn2 *P,ecn *Q,big x,zzn2 *X,zzn12 *r)
{
    /*BOOL Ok;
     Big xx,yy;
     ZZn Qx,Qy;
     
     P.norm();
     Q.get(xx,yy); Qx=xx; Qy=yy;
     
     Ok=fast_pairing_cuda(P,Qx,Qy,x,X,r);
     
     if (Ok) return TRUE;
     return FALSE;*/
    BOOL Ok;
    big xx,yy;
    zzn Qx,Qy;
    
    xx = mirvar_cuda(_MIPP_  0);
    yy = mirvar_cuda(_MIPP_  0);
    Qx = mirvar_cuda(_MIPP_  0);
    Qy = mirvar_cuda(_MIPP_  0);
    
    ecn2_norm_cuda(_MIPP_ P);
    epoint_get_cuda(_MIPP_ Q, xx, yy);
    
    nres_cuda(_MIPP_ xx, Qx);
    nres_cuda(_MIPP_ yy, Qy);
    


    Ok = fast_pairing_cuda(_MIPP_ P, Qx, Qy, x, X, r);
    
    
    mirkill_cuda(xx);
    mirkill_cuda(yy);
    mirkill_cuda(Qx);
    mirkill_cuda(Qy);
    
    if (Ok) return TRUE;
    return FALSE;
}

#endif

#ifndef smzzn12_c
#define smzzn12_c

#define FUNC_BASE 226

__device__ void zzn12_mirvar_cuda(_MIPD_ zzn12*w){
    zzn4_mirvar_cuda(_MIPP_ &(w->a));
    zzn4_mirvar_cuda(_MIPP_ &(w->b));
    zzn4_mirvar_cuda(_MIPP_ &(w->c));
    w->miller = FALSE;
    w->unitary = FALSE;
}

__device__ void zzn12_kill_cuda(_MIPD_ zzn12*w){
    zzn4_kill_cuda(_MIPP_ &(w->a));
    zzn4_kill_cuda(_MIPP_ &(w->b));
    zzn4_kill_cuda(_MIPP_ &(w->c));
}

__device__ void zzn12_mark_miller_cuda(_MIPD_ zzn12* w){
    w->miller = TRUE;
}

__device__ void zzn12_mark_regular_cuda(_MIPD_ zzn12* w){
    w->miller = FALSE;
    w->unitary = FALSE;
}

__device__ void zzn12_mark_unitary_cuda(_MIPD_ zzn12* w){
    w->miller = FALSE;
    w->unitary = TRUE;
}

__device__ void zzn12_from_int_cuda(_MIPD_ int i,zzn12 *x){
#ifdef MR_OS_THREADS
    miracl *mr_mip=get_mip_cuda();
#endif
    if (mr_mip->ERNUM) return;
    
    MR_IN(FUNC_BASE+0)
    zzn4_from_int_cuda(_MIPP_ i, &(x->a));
    zzn4_zero_cuda(&(x->b));
    zzn4_zero_cuda(&(x->c));
    x->miller = FALSE;
    if (i == 1) {
        x->unitary = TRUE;
    }else{
        x->unitary = FALSE;
    }
    MR_OUT
}

__device__ BOOL zzn12_isunity_cuda(_MIPD_ zzn12 *x){
    return x->unitary;
}

__device__ BOOL zzn12_iszero_cuda(_MIPD_ zzn12 *w){
    if(zzn4_iszero_cuda(&(w->a)) && zzn4_iszero_cuda(&(w->b)) && zzn4_iszero_cuda(&(w->c))){
        return TRUE;
    }
    return FALSE;
}

__device__ void zzn12_copy_cuda(zzn12 *w,zzn12 *x){
    zzn4_copy_cuda(&(w->a), &(x->a));
    zzn4_copy_cuda(&(w->b), &(x->b));
    zzn4_copy_cuda(&(w->c), &(x->c));
    x->miller = w->miller;
    x->unitary = w->unitary;
}
__device__ void zzn12_zero_cuda(zzn12 *w){
    zzn4_zero_cuda(&(w->a));
    zzn4_zero_cuda(&(w->b));
    zzn4_zero_cuda(&(w->c));
    w->unitary = FALSE;
    w->miller = FALSE;
}
__device__ void zzn12_powq_cuda(_MIPD_ zzn12 *w,zzn2 *X){
    //ZZn2 XX=X*X;
    //ZZn2 XXX=XX*X;
    //BOOL ku=unitary;
    //BOOL km=miller;
    //a.powq(XXX); b.powq(XXX); c.powq(XXX);
    //b*=X;
    //c*=XX;
    //unitary=ku;
    //miller=km;
    
    zzn12 y;
	zzn2 XX,XXX;
    BOOL ku,km;
	
	zzn12_mirvar_cuda(_MIPP_ &y);
    
    zzn12_copy_cuda(w, &y);
    
  
    zzn2_mirvar_cuda(_MIPP_ &XX);
    zzn2_mul_cuda(_MIPP_ X, X, &XX);
    
  
    zzn2_mirvar_cuda(_MIPP_ &XXX);
    zzn2_mul_cuda(_MIPP_ &XX, X, &XXX);
    
    ku = y.unitary;
    km = y.miller;
    
    zzn4_powq_cuda(_MIPP_ &XXX, &(y.a));
    zzn4_powq_cuda(_MIPP_ &XXX, &(y.b));
    zzn4_powq_cuda(_MIPP_ &XXX, &(y.c));
    
    zzn4_smul_cuda(_MIPP_ &(y.b), X, &(y.b));
    zzn4_smul_cuda(_MIPP_ &(y.c), &XX, &(y.c));
    
    y.unitary = ku;
    y.miller = km;
    
    zzn12_copy_cuda(&y, w);
    
    zzn2_kill_cuda(_MIPP_ &XX);
    zzn2_kill_cuda(_MIPP_ &XXX);
    zzn12_kill_cuda(_MIPP_ &y);
    
}
__device__ void zzn12_pow_cuda(_MIPD_ zzn12 *x,big e){
    int i,j,nb,n,nbw,nzs;
    zzn12 u,u2,t[32];
	big ONE = mirvar_cuda(_MIPP_ 1);
    big ZERO = mirvar_cuda(_MIPP_ 0);
    
    
    zzn12_mirvar_cuda(_MIPP_ &u);
    zzn12_mirvar_cuda(_MIPP_ &u2);
    for (i=0;i<32;i++) zzn12_mirvar_cuda(_MIPP_ &(t[i]));
    
    if (zzn12_iszero_cuda(_MIPP_ x)) {
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn12_kill_cuda(_MIPP_ &u);
        zzn12_kill_cuda(_MIPP_ &u2);
        for (i=0;i<32;i++) zzn12_kill_cuda(_MIPP_ &(t[i]));
        return;
    }
    if (mr_compare_cuda(e, ZERO)==0){
        zzn12_from_int_cuda(_MIPP_ 1,x);
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn12_kill_cuda(_MIPP_ &u);
        zzn12_kill_cuda(_MIPP_ &u2);
        for (i=0;i<32;i++) zzn12_kill_cuda(_MIPP_ &(t[i]));
        return ;
    }
    if (mr_compare_cuda(e, ONE)==0){
        mirkill_cuda(ONE);
        mirkill_cuda(ZERO);
        zzn12_kill_cuda(_MIPP_ &u);
        zzn12_kill_cuda(_MIPP_ &u2);
        for (i=0;i<32;i++) zzn12_kill_cuda(_MIPP_ &(t[i]));
        return;
    }
    
 //   zzn12_copy_cuda(x, &u);
 //	for (i=1;i<1000;i++) zzn12_mul_cuda(_MIPP_ x, x, &u);

	zzn12_copy_cuda(x, &u);
    zzn12_mul_cuda(_MIPP_ &u, &u, &u2);
    zzn12_copy_cuda(&u, &(t[0]));
    
    for (i=1;i<32;i++) zzn12_mul_cuda(_MIPP_ &u2, &(t[i-1]), &(t[i]));
    
    nb = logb2_cuda(_MIPP_ e);
    if (nb>1) for (i=nb-2;i>=0;)
    {
        n = mr_window_cuda(_MIPP_ e, i, &nbw, &nzs, 6);
        for (j=0;j<nbw;j++) zzn12_mul_cuda(_MIPP_ &u, &u, &u);
        if (n>0)  zzn12_mul_cuda(_MIPP_ &u, &(t[n/2]), &u);
        i-=nbw;
        if (nzs)
        {
            for (j=0;j<nzs;j++) zzn12_mul_cuda(_MIPP_ &u, &u, &u);
            i-=nzs;
        }
    }
    
    zzn12_copy_cuda(&u, x);
    
    mirkill_cuda(ONE);
    mirkill_cuda(ZERO);
    zzn12_kill_cuda(_MIPP_ &u);
    zzn12_kill_cuda(_MIPP_ &u2);
    for (i=0;i<32;i++) zzn12_kill_cuda(_MIPP_ &(t[i]));
}


__device__ void zzn12_add_cuda(_MIPD_ zzn12 *w,zzn12 *x,zzn12 *y){
    zzn4_add_cuda(_MIPP_ &(w->a), &(x->a), &(y->a));
    zzn4_add_cuda(_MIPP_ &(w->b), &(x->b), &(y->b));
    zzn4_add_cuda(_MIPP_ &(w->c), &(x->c), &(y->c));
    y->miller = w->miller;
    y->unitary = FALSE;
}
__device__ void zzn12_sub_cuda(_MIPD_ zzn12 *w,zzn12 *x,zzn12 *y){
    zzn4_sub_cuda(_MIPP_ &(w->a), &(x->a), &(y->a));
    zzn4_sub_cuda(_MIPP_ &(w->b), &(x->b), &(y->b));
    zzn4_sub_cuda(_MIPP_ &(w->c), &(x->c), &(y->c));
    y->miller = w->miller;
    y->unitary = FALSE;
}
__device__ void zzn12_smul_cuda(_MIPD_ zzn12 *w,zzn4 *x,zzn12 *y){
    zzn4_mul_cuda(_MIPP_ &(w->a), x, &(y->a));
    zzn4_mul_cuda(_MIPP_ &(w->b), x, &(y->b));
    zzn4_mul_cuda(_MIPP_ &(w->c), x, &(y->c));
    y->miller = w->miller;
    y->unitary = FALSE;
}

__device__ void zzn12_mul_cuda(_MIPD_ zzn12 *w,zzn12 *x,zzn12 *y){
    
    
    zzn4 tx;
	zzn12_copy_cuda(w, y);
    zzn4_mirvar_cuda(_MIPP_ &tx);
    if (zzn12_compare_cuda(w, x)){
        zzn4 A,B,C,D;
        
        zzn4_mirvar_cuda(_MIPP_ &A);
        zzn4_mirvar_cuda(_MIPP_ &B);
        zzn4_mirvar_cuda(_MIPP_ &C);
        zzn4_mirvar_cuda(_MIPP_ &D);
        
        if (w->unitary){
            // Granger & Scott PKC 2010 - only 3 squarings!
            //A=a; a*=a; D=a; a+=a; a+=D; A.conj(); A+=A; a-=A;
            zzn4_copy_cuda(&(y->a), &A);
            zzn4_mul_cuda(_MIPP_ &(y->a), &(y->a), &(y->a));
            zzn4_copy_cuda(&(y->a), &D);
            zzn4_add_cuda(_MIPP_ &(y->a), &(y->a), &(y->a));
            zzn4_add_cuda(_MIPP_ &(y->a), &D, &(y->a));
            zzn4_conj_cuda(_MIPP_ &A, &A);
            zzn4_add_cuda(_MIPP_ &A, &A, &A);
            zzn4_sub_cuda(_MIPP_ &(y->a), &A, &(y->a));
            
            //B=c; B*=B; B=tx(B); D=B; B+=B; B+=D;
            zzn4_copy_cuda(&(y->c), &B);
            zzn4_mul_cuda(_MIPP_ &B, &B, &B);
            zzn4_tx_cuda(_MIPP_ &B);
            zzn4_copy_cuda(&B, &D);
            zzn4_add_cuda(_MIPP_ &B, &B, &B);
            zzn4_add_cuda(_MIPP_ &B, &D, &B);
            
            //C=b; C*=C;          D=C; C+=C; C+=D;
            zzn4_copy_cuda(&(y->b), &C);
            zzn4_mul_cuda(_MIPP_ &C, &C, &C);
            zzn4_copy_cuda(&C, &D);
            zzn4_add_cuda(_MIPP_ &C, &C, &C);
            zzn4_add_cuda(_MIPP_ &C, &D, &C);
            
            //b.conj(); b+=b; c.conj(); c+=c; c=-c;
            zzn4_conj_cuda(_MIPP_ &(y->b), &(y->b));
            zzn4_add_cuda(_MIPP_ &(y->b), &(y->b), &(y->b));
            zzn4_conj_cuda(_MIPP_ &(y->c), &(y->c));
            zzn4_add_cuda(_MIPP_ &(y->c), &(y->c), &(y->c));
            zzn4_negate_cuda(_MIPP_ &(y->c), &(y->c));
            //b+=B; c+=C;
            zzn4_add_cuda(_MIPP_ &(y->b), &B, &(y->b));
            zzn4_add_cuda(_MIPP_ &(y->c), &C, &(y->c));
        }else{
            if (!(w->miller)){
                // Chung-Hasan SQR2
                //A=a; A*=A;
                zzn4_copy_cuda(&(y->a), &A);
                zzn4_mul_cuda(_MIPP_ &A, &A, &A);
                //B=b*c; B+=B;
                zzn4_mul_cuda(_MIPP_ &(y->b), &(y->c), &B);
                zzn4_add_cuda(_MIPP_ &B, &B, &B);
                //C=c; C*=C;
                zzn4_copy_cuda(&(y->c), &C);
                zzn4_mul_cuda(_MIPP_ &C, &C, &C);
                //D=a*b; D+=D;
                zzn4_mul_cuda(_MIPP_ &(y->a), &(y->b), &D);
                zzn4_add_cuda(_MIPP_ &D, &D, &D);
                //c+=(a+b); c*=c;
                zzn4_add_cuda(_MIPP_ &(y->a), &(y->b), &(y->c));
                zzn4_add_cuda(_MIPP_ &(y->c), &(y->c), &(y->c));
                zzn4_mul_cuda(_MIPP_ &(y->c), &(y->c), &(y->c));
                //a=A+tx(B);
                zzn4_copy_cuda(&B, &tx);
                zzn4_tx_cuda(_MIPP_ &tx);
                zzn4_add_cuda(_MIPP_ &A, &tx, &(y->a));
                //b=D+tx(C);
                zzn4_copy_cuda(&C, &tx);
                zzn4_tx_cuda(_MIPP_ &tx);
                zzn4_add_cuda(_MIPP_ &D, &tx, &(y->b));
                //c-=(A+B+C+D);
                zzn4_sub_cuda(_MIPP_ &(y->c), &A, &(y->c));
                zzn4_sub_cuda(_MIPP_ &(y->c), &B, &(y->c));
                zzn4_sub_cuda(_MIPP_ &(y->c), &C, &(y->c));
                zzn4_sub_cuda(_MIPP_ &(y->c), &D, &(y->c));
            }else{
                // Chung-Hasan SQR3 - actually calculate 2x^2 !
                // Slightly dangerous - but works as will be raised to p^{k/2}-1
                // which wipes out the 2.
                
     //           A=a; A*=A;       // a0^2    = S0
                zzn4_copy_cuda(&(y->a), &A);
                zzn4_mul_cuda(_MIPP_ &A, &A, &A);
     //           C=c; C*=b; C+=C; // 2a1.a2  = S3
                zzn4_copy_cuda(&(y->c), &C);
                zzn4_mul_cuda(_MIPP_ &C, &(y->b), &C);
                zzn4_add_cuda(_MIPP_ &C, &C, &C);
     //           D=c; D*=D;       // a2^2    = S4
                zzn4_copy_cuda(&(y->c), &D);
                zzn4_mul_cuda(_MIPP_ &D, &D, &D);
     //          c+=a;            // a0+a2
                zzn4_add_cuda(_MIPP_ &(y->c), &(y->a), &(y->c));
     //           B=b; B+=c; B*=B; // (a0+a1+a2)^2  =S1
                zzn4_copy_cuda(&(y->b), &B);
                zzn4_add_cuda(_MIPP_ &B, &(y->c), &B);
                zzn4_mul_cuda(_MIPP_ &B, &B, &B);
     //           c-=b; c*=c;      // (a0-a1+a2)^2  =S2
                zzn4_sub_cuda(_MIPP_ &(y->c), &(y->b), &(y->c));
                zzn4_mul_cuda(_MIPP_ &(y->c), &(y->c), &(y->c));
     //           C+=C; A+=A; D+=D;
                zzn4_add_cuda(_MIPP_ &C, &C, &C);
                zzn4_add_cuda(_MIPP_ &A, &A, &A);
                zzn4_add_cuda(_MIPP_ &D, &D, &D);
     //           a=A+tx(C);
                zzn4_copy_cuda(&C, &tx);
                zzn4_tx_cuda(_MIPP_ &tx);
                zzn4_add_cuda(_MIPP_ &A, &tx, &(y->a));
                
     //           b=B-c-C+tx(D);
                zzn4_copy_cuda(&D, &tx);
                zzn4_tx_cuda(_MIPP_ &tx);
                zzn4_sub_cuda(_MIPP_ &B, &(y->c), &(y->b));
                zzn4_sub_cuda(_MIPP_ &(y->b), &C, &(y->b));
                zzn4_add_cuda(_MIPP_ &(y->b), &tx, &(y->b));
     //           c+=B-A-D;  // is this code telling me something...?
                zzn4_add_cuda(_MIPP_ &(y->c), &B, &(y->c));
                zzn4_sub_cuda(_MIPP_ &(y->c), &A, &(y->c));
                zzn4_sub_cuda(_MIPP_ &(y->c), &D, &(y->c));
            }
        
        }
        zzn4_kill_cuda(_MIPP_ &A);
        zzn4_kill_cuda(_MIPP_ &B);
        zzn4_kill_cuda(_MIPP_ &C);
        zzn4_kill_cuda(_MIPP_ &D);
    }else{
        // Karatsuba
        zzn4 Z0,Z1,Z2,Z3,T0,T1;
        //BOOL zero_c,zero_b;
        BOOL zero_c,zero_b;
        
        zzn4_mirvar_cuda(_MIPP_ &Z0);
        zzn4_mirvar_cuda(_MIPP_ &Z1);
        zzn4_mirvar_cuda(_MIPP_ &Z2);
        zzn4_mirvar_cuda(_MIPP_ &Z3);
        zzn4_mirvar_cuda(_MIPP_ &T0);
        zzn4_mirvar_cuda(_MIPP_ &T1);
        //zero_c=(x.c).iszero();
        //zero_b=(x.b).iszero();
        zero_c = zzn4_iszero_cuda(&(x->c));
        zero_b = zzn4_iszero_cuda(&(x->b));
        //Z0=a*x.a;  //9
        zzn4_mul_cuda(_MIPP_ &(y->a), &(x->a), &Z0);
        //if (!zero_b) Z2=b*x.b;  //+6
        if (!zero_b) {
            zzn4_mul_cuda(_MIPP_ &(y->b), &(x->b), &Z2);
        }
        
        
        //T0=a+b;
        //T1=x.a+x.b;
        //Z1=T0*T1;  //+9
        //Z1-=Z0;
        //if (!zero_b) Z1-=Z2;
        //T0=b+c;
        //T1=x.b+x.c;
        //Z3=T0*T1;  //+6
        
        zzn4_add_cuda(_MIPP_ &(y->a), &(y->b), &T0);
        zzn4_add_cuda(_MIPP_ &(x->a), &(x->b), &T1);
        zzn4_mul_cuda(_MIPP_ &T0, &T1, &Z1);
        zzn4_sub_cuda(_MIPP_ &Z1, &Z0, &Z1);
        if (!zero_b) {
            zzn4_sub_cuda(_MIPP_ &Z1, &Z2, &Z1);
        }
        zzn4_add_cuda(_MIPP_ &(y->b), &(y->c), &T0);
        zzn4_add_cuda(_MIPP_ &(x->b), &(x->c), &T1);
        zzn4_mul_cuda(_MIPP_ &T0, &T1, &Z3);
        
        //if (!zero_b) Z3-=Z2;
        if (!zero_b){
            zzn4_sub_cuda(_MIPP_ &Z3, &Z2, &Z3);
        }
        
        //T0=a+c;
        zzn4_add_cuda(_MIPP_ &(y->a), &(y->c), &T0);
        //T1=x.a+x.c;
        //T0*=T1;   //+9=39 for "special case"
        //if (!zero_b) Z2+=T0;
        //else         Z2=T0;
        //Z2-=Z0;
        zzn4_add_cuda(_MIPP_ &(x->a), &(x->c), &T1);
        zzn4_mul_cuda(_MIPP_ &T0, &T1, &T0);
        if (!zero_b){
            zzn4_add_cuda(_MIPP_ &Z2, &T0, &Z2);
        }else{
            zzn4_copy_cuda(&T0, &Z2);
        }
        zzn4_sub_cuda(_MIPP_ &Z2, &Z0, &Z2);
        
        
       // b=Z1;
       // if (!zero_c)
       // { // exploit special form of BN curve line_cuda function
       //     T0=c*x.c;
       //     Z2-=T0;
       //     Z3-=T0;
       //     b+=tx(T0);
       // }
        zzn4_copy_cuda(&Z1, &(y->b));
        if (!zero_c){
            zzn4_mul_cuda(_MIPP_ &(y->c), &(x->c), &T0);
            zzn4_sub_cuda(_MIPP_ &Z2, &T0, &Z2);
            zzn4_sub_cuda(_MIPP_ &Z3, &T0, &Z3);
            zzn4_tx_cuda(_MIPP_ &T0);
            zzn4_add_cuda(_MIPP_ &(y->b), &T0, &(y->b));
        }
        
        //a=Z0+tx(Z3);
        //c=Z2;
        zzn4_tx_cuda(_MIPP_ &Z3);
        zzn4_add_cuda(_MIPP_ &Z0, &Z3, &(y->a));
        zzn4_copy_cuda(&Z2, &(y->c));
        //if (!x.unitary) unitary=FALSE;
        if(!(x->unitary)) y->unitary = FALSE;
        
        zzn4_kill_cuda(_MIPP_ &Z0);
        zzn4_kill_cuda(_MIPP_ &Z1);
        zzn4_kill_cuda(_MIPP_ &Z2);
        zzn4_kill_cuda(_MIPP_ &Z3);
        zzn4_kill_cuda(_MIPP_ &T0);
        zzn4_kill_cuda(_MIPP_ &T1);
    }
    zzn4_kill_cuda(_MIPP_ &tx);
}





__device__ void zzn12_inv_cuda(_MIPD_ zzn12 *w){
    zzn12 y;
    zzn4 f0,f1;
	zzn12_mirvar_cuda(_MIPP_ &y);
    //if (w.unitary)
    //{
    //    y=conj(w);
    //    return y;
    //}

    if( w->unitary){
        zzn12_conj_cuda(_MIPP_ w, &y);
        zzn12_copy_cuda(&y, w);
        zzn12_kill_cuda(_MIPP_ &y);
        return;
    }
    
    
    zzn4_mirvar_cuda(_MIPP_ &f0);
    zzn4_mirvar_cuda(_MIPP_ &f1);
    //y.a=w.a*w.a-tx(w.b*w.c);
    zzn4_mul_cuda(_MIPP_ &(w->a), &(w->a), &(y.a));
    zzn4_mul_cuda(_MIPP_ &(w->b), &(w->c), &f0);
    zzn4_tx_cuda(_MIPP_ &f0);
    zzn4_sub_cuda(_MIPP_ &(y.a), &f0, &(y.a));
    //y.b=tx(w.c*w.c)-w.a*w.b;
    zzn4_mul_cuda(_MIPP_ &(w->c), &(w->c), &(y.b));
    zzn4_tx_cuda(_MIPP_ &(y.b));
    zzn4_mul_cuda(_MIPP_ &(w->a), &(w->b), &f0);
    zzn4_sub_cuda(_MIPP_ &(y.b), &f0, &(y.b));
    //y.c=w.b*w.b-w.a*w.c;
    zzn4_mul_cuda(_MIPP_ &(w->b), &(w->b), &(y.c));
    zzn4_mul_cuda(_MIPP_ &(w->a), &(w->c), &f0);
    zzn4_sub_cuda(_MIPP_ &(y.c), &f0, &(y.c));
    
    //f0=tx(w.b*y.c)+w.a*y.a+tx(w.c*y.b);
    zzn4_mul_cuda(_MIPP_ &(w->b), &(y.c), &f0);
    zzn4_tx_cuda(_MIPP_ &f0);
    zzn4_mul_cuda(_MIPP_ &(w->a), &(y.a), &f1);
    zzn4_add_cuda(_MIPP_ &f0, &f1, &f0);
    zzn4_mul_cuda(_MIPP_ &(w->c), &(y.b), &f1);
    zzn4_tx_cuda(_MIPP_ &f1);
    zzn4_add_cuda(_MIPP_ &f0, &f1, &f0);
    //f0=inverse(f0);
    zzn4_inv_cuda(_MIPP_ &f0);
    //y.c*=f0;
    //y.b*=f0;
    //y.a*=f0;
    zzn4_mul_cuda(_MIPP_ &(y.a), &f0, &(y.a));
    zzn4_mul_cuda(_MIPP_ &(y.b), &f0, &(y.b));
    zzn4_mul_cuda(_MIPP_ &(y.c), &f0, &(y.c));
    
    zzn12_copy_cuda(&y, w);
    zzn4_kill_cuda(_MIPP_ &f0);
    zzn4_kill_cuda(_MIPP_ &f1);
    zzn12_kill_cuda(_MIPP_ &y);
}

__device__ void zzn12_from_zzn4s_cuda(zzn4 * a,zzn4 *b,zzn4 *c,zzn12 *w){
    zzn4_copy_cuda(a, &(w->a));
    zzn4_copy_cuda(b, &(w->b));
    zzn4_copy_cuda(c, &(w->c));
    w->miller = FALSE;
    w->unitary = FALSE;
}

__device__ void zzn12_conj_cuda(_MIPD_ zzn12 *w,zzn12 *x){
    zzn4_conj_cuda(_MIPP_ &(w->a), &(x->a));
    zzn4_conj_cuda(_MIPP_ &(w->b), &(x->b));
    zzn4_negate_cuda(_MIPP_ &(x->b), &(x->b));
    zzn4_conj_cuda(_MIPP_ &(w->c), &(x->c));
    x->unitary = w->unitary;
    x->miller = w->miller;
}

__device__ BOOL zzn12_compare_cuda(zzn12 *w,zzn12 *x){
    if (zzn4_compare_cuda(&(w->a), &(x->a)) &&
        zzn4_compare_cuda(&(w->b), &(x->b)) &&
        zzn4_compare_cuda(&(w->c), &(x->c)) ){
        return TRUE;
    }
    return FALSE;
}

__device__ void zzn12_tochar_cuda(_MIPD_ zzn12* w,unsigned char *c,unsigned int bInt){
    int hlen = 0;
    big m = mirvar_cuda(_MIPP_ 0);
    redc_cuda(_MIPP_ w->c.b.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->c.b.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->c.a.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->c.a.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->b.b.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->b.b.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->b.a.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->b.a.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->a.b.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->a.b.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->a.a.b, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    
    redc_cuda(_MIPP_ w->a.a.a, m);
    big_to_bytes_cuda(_MIPP_ bInt, m, (char *)c+hlen, TRUE);
    hlen += bInt;
    mirkill_cuda(m);

}

__device__ void zzn12_fromchar_cuda(_MIPD_ zzn12* w,unsigned char *c,unsigned int bInt){
    int hlen = 0;
    big m = mirvar_cuda(_MIPP_ 0);
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->c.b.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->c.b.a);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->c.a.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->c.a.a);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->b.b.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->b.b.a);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->b.a.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->b.a.a);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->a.b.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->a.b.a);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->a.a.b);
    hlen += bInt;
    
    bytes_to_big_cuda(_MIPP_ bInt, (char *)c+hlen, m);
    nres_cuda(_MIPP_ m, w->a.a.a);
    hlen += bInt;
    
    w->miller = FALSE;
    w->unitary = TRUE;
    mirkill_cuda(m);
}


#endif

#ifndef sm9_utils_c
#define sm9_utils_c

__device__ unsigned char hfc[2] = {0x01,0x02};

__device__ miracl* GenMiracl_cuda(int secLevel){
    miracl *mr_mip;
	if (!sm9init) return NULL;
    mr_mip = mr_first_alloc_cuda();
#ifdef MR_OS_THREADS
    miracl *mr_mip = get_mip_cuda();
#endif
    _MIPPO_ = mirsys_cuda(128,0);
    mr_mip->TWIST = TWIST;
    
    ecurve_init_cuda(_MIPP_ sm9a,sm9b,sm9q,MR_PROJECTIVE);
    
    return mr_mip;
}

__device__ void CloseMiracl_cuda(_MIPDO_)
{
    if( _MIPPO_ )
        mirexit_cuda(_MIPPO_);
}

__device__ int ceilfunc_cuda( double v )
{
    int vi;
    vi = (int)v;
    if( (v - vi) > 0 )
        vi+=1;
    return vi;
}

__device__ int floorfunc_cuda( double v )
{
    int vi;
    vi = (int)v;
    return vi;
}

__device__ int inverseUcBuf_cuda(unsigned char *H, int HLen)
{
    unsigned char tmpChar;
    int i,loop;
    
    loop = HLen/2;
    
    for( i= 0;i<loop;i++){
        tmpChar = H[i];
        H[i] = H[ HLen -1 -i];
        H[ HLen -1 -i] = tmpChar;
    }
    
    return 0;
}

__device__ int integer_to_bytes_cuda( int bitLen, unsigned char *xbuf, int xbufLen )
{
    unsigned char intBuf[4], *ptmpBuf;
    
    ptmpBuf = (unsigned char *)&bitLen;
    
    memcpy( intBuf, ptmpBuf, sizeof(int));
    memcpy(xbuf, intBuf, xbufLen>sizeof(int)?sizeof(int):xbufLen);
#ifdef  MR_LITTLE_ENDIAN
    inverseUcBuf_cuda( xbuf, xbufLen);
#endif
    
#ifdef  MR_BIG_ENDIAN
#endif
    
    return 0;
}

// h function. h=1: h1; h=2:h2
__device__ big Hfun_cuda(_MIPD_ char *zbuf, int zbufLen, int secLevel,int h)
{
    int ct = 0x00000001;
    int v = SM3_HASH_LEN*8;
    int hlentmp = ceilfunc_cuda((double)secLevel*5/4)*8;
    int hlentmp2;
    int	loopCount = ceilfunc_cuda((double)hlentmp/v);
    int i,HaLen, HaFLen;
    char kbuf[50];
    big t,N;
    SM3_CTX sm3;
	unsigned char s[SM3_HASH_LEN], ctBuf[4];


	t = mirvar_cuda(_MIPP_ 0);
    N = mirvar_cuda(_MIPP_ 0);
    
    HaLen = 0;
    for( i=0; i<loopCount; i++){
        SM3Init_cuda(&sm3);
        SM3Update_cuda( &sm3, hfc+(h-1), 1);
        SM3Update_cuda( &sm3, (unsigned char *)zbuf, zbufLen);
        integer_to_bytes_cuda( ct, ctBuf, sizeof(ctBuf));
        SM3Update_cuda( &sm3, ctBuf, sizeof(ct));
        SM3Final_cuda( s, &sm3 );
        if( i == loopCount -1) {
            if(loopCount * v == hlentmp) {
                memcpy(kbuf+HaLen, s, SM3_HASH_LEN);
                HaLen += SM3_HASH_LEN;
            }else{
                hlentmp2 = floorfunc_cuda((double) hlentmp/v);
                
                HaFLen = (hlentmp - v*(int)hlentmp2)/8;
                memcpy(kbuf+HaLen, s, HaFLen);
                HaLen += HaFLen;
            }
        }else{
            memcpy(kbuf+HaLen, s, SM3_HASH_LEN);
            HaLen += SM3_HASH_LEN;
        }
        ct++;
    }
    bytes_to_big_cuda(_MIPP_ HaLen, (char *)kbuf, t);
    decr_cuda(_MIPP_ sm9n, 1, N);
    divide_cuda(_MIPP_ t, N, N);
    incr_cuda(_MIPP_ t, 1, t);
    mirkill_cuda(N);
    return t;
}

__device__ int kdf_cuda(char *zbuf, int zbufLen, int klen, char *kbuf)
{
    int ct = 0x00000001;
    int v = SM3_HASH_LEN*8;
    int hlentmp = ceilfunc_cuda( (double)klen/v);
    int	loopCount = (int)hlentmp;
    int i,HaLen, HaFLen;
    SM3_CTX sm3;
    unsigned char s[SM3_HASH_LEN], ctBuf[4];
    
    HaLen = 0;
    for( i=0; i<loopCount; i++){
        SM3Init_cuda(&sm3);
        
        
        SM3Update_cuda( &sm3, (unsigned char *)zbuf, zbufLen);
        
        integer_to_bytes_cuda( ct, ctBuf, sizeof(ctBuf));
        
        
        SM3Update_cuda( &sm3, ctBuf, sizeof(ct));
        
        SM3Final_cuda( s, &sm3 );
        if( i == loopCount -1) {
            if(loopCount * v == klen) {
                memcpy(kbuf+HaLen, s, SM3_HASH_LEN);
                HaLen += SM3_HASH_LEN;
            }else{
                hlentmp = floorfunc_cuda((double) klen/v);
                
                HaFLen = (klen - v*(int)hlentmp)/8;
                memcpy(kbuf+HaLen, s, HaFLen);
                HaLen += HaFLen;
            }
        }else{
            memcpy(kbuf+HaLen, s, SM3_HASH_LEN);
            HaLen += SM3_HASH_LEN;
        }
        ct++;
    }
    
    return 0;
}

__device__ int MAC_cuda(unsigned char* key, unsigned int keylen, unsigned char* mes, unsigned int meslen, unsigned char *mac){
    
    SM3_CTX sm3;
    
    SM3Init_cuda(&sm3);
    SM3Update_cuda( &sm3, mes, meslen);
    SM3Update_cuda( &sm3, key, keylen);
    SM3Final_cuda( mac, &sm3 );

    return 0;
}

__device__ int xorAlgor_cuda( unsigned char *bufIn, int ilen, unsigned char *bufKey, unsigned char * bufOut)
{
    int i;
    for(i=0;i<ilen;i++)
    {
        bufOut[i] = bufIn[i]^bufKey[i];
    }
    return 0;
}

#endif

#ifndef sm9_setup_c
#define sm9_setup_c
// __device__ BOOL sm9init = FALSE;
// __device__ BOOL sm9sign = FALSE;
// __device__ BOOL sm9encrypt = FALSE;
// __device__ BOOL sm9keyexchange = FALSE;

// __device__ int   sm9len = 32;
// // curve parameter
// __device__ big   sm9q;
// __device__ big   sm9a;
// __device__ big   sm9b;
// __device__ big   sm9n;

// // precomputed parameters
// __device__ big   sm9t;
// __device__ zzn2  sm9X;

// __device__ ecn   p1G1;
// __device__ ecn   ppG1;
// __device__ ecn   keG1;

// __device__ ecn2  p2G2;
// __device__ ecn2  ppG2;

// __device__ zzn12 gGt;
// __device__ zzn12 eGt;
// __device__ zzn12 kGt;

// __device__ BOOL  TWIST;

// // hid of algorithm
// //01 signature       algorithm
// //02 keyexchange     algorithm
// //03 encrypt         algorithm
// __device__ unsigned char hid[3] = {0x01,0x02,0x03};


//=======================

//parameters of standard sm9

__device__ BOOL twist = SM_SEXTIC_M;

__device__ char ct[32+1] = {"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x60\x00\x00\x00\x00\x58\xF9\x8A"};

__device__ char cq[32+1] = {"\xB6\x40\x00\x00\x02\xA3\xA6\xF1\xD6\x03\xAB\x4F\xF5\x8E\xC7\x45\x21\xF2\x93\x4B\x1A\x7A\xEE\xDB\xE5\x6F\x9B\x27\xE3\x51\x45\x7D"};

__device__ char ca[32+1] = {"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"};

__device__ char cb[32+1] = {"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05"};

__device__ char cn[32+1] = {"\xB6\x40\x00\x00\x02\xA3\xA6\xF1\xD6\x03\xAB\x4F\xF5\x8E\xC7\x44\x49\xF2\x93\x4B\x18\xEA\x8B\xEE\xE5\x6E\xE1\x9C\xD6\x9E\xCF\x25"};

__device__ char cxp1[32+1] = {"\x93\xDE\x05\x1D\x62\xBF\x71\x8F\xF5\xED\x07\x04\x48\x7D\x01\xD6\xE1\xE4\x08\x69\x09\xDC\x32\x80\xE8\xC4\xE4\x81\x7C\x66\xDD\xDD"};

__device__ char cyp1[32+1] = {"\x21\xFE\x8D\xDA\x4F\x21\xE6\x07\x63\x10\x65\x12\x5C\x39\x5B\xBC\x1C\x1C\x00\xCB\xFA\x60\x24\x35\x0C\x46\x4C\xD7\x0A\x3E\xA6\x16"};

__device__ char cxq1[32+1] = {"\x37\x22\x75\x52\x92\x13\x0B\x08\xD2\xAA\xB9\x7F\xD3\x4E\xC1\x20\xEE\x26\x59\x48\xD1\x9C\x17\xAB\xF9\xB7\x21\x3B\xAF\x82\xD6\x5B"};

__device__ char cxq2[32+1] = {"\x85\xAE\xF3\xD0\x78\x64\x0C\x98\x59\x7B\x60\x27\xB4\x41\xA0\x1F\xF1\xDD\x2C\x19\x0F\x5E\x93\xC4\x54\x80\x6C\x11\xD8\x80\x61\x41"};

__device__ char cyq1[32+1] = {"\xA7\xCF\x28\xD5\x19\xBE\x3D\xA6\x5F\x31\x70\x15\x3D\x27\x8F\xF2\x47\xEF\xBA\x98\xA7\x1A\x08\x11\x62\x15\xBB\xA5\xC9\x99\xA7\xC7"};

__device__ char cyq2[32+1] = {"\x17\x50\x9B\x09\x2E\x84\x5C\x12\x66\xBA\x0D\x26\x2C\xBE\xE6\xED\x07\x36\xA9\x6F\xA3\x47\xC8\xBD\x85\x6D\xC7\x6B\x84\xEB\xEB\x96"};

__device__ char cmasterkey[32+1] = {"\x00\x01\x30\xE7\x84\x59\xD7\x85\x45\xCB\x54\xC5\x87\xE0\x2C\xF4\x80\xCE\x0B\x66\x34\x0F\x31\x9F\x34\x8A\x1D\x5B\x1F\x2D\xC5\xF4"};

__device__ char sm9Xx[32+1] = {"\x3f\x23\xea\x58\xe5\x72\x0b\xdb\x84\x3c\x6c\xfa\x9c\x08\x67\x49\x47\xc5\xc8\x6e\x0d\xdd\x04\xed\xa9\x1d\x83\x54\x37\x7b\x69\x8b"};
//================================

__device__ flash gao(miracl *mr_mip, int aaa)
{
    printf("nb!");
    return NULL;
}

#define PRE_CALC (1<<7)

Comp a[(TEST_CUDA*PRE_CALC)<<2],b[(TEST_CUDA*PRE_CALC)<<2];
int r[(TEST_CUDA*PRE_CALC)<<3];

void SM9_Signature_PreCalc(unsigned int N, SM9_SSK *sk, SM9_Sign *sign)
{
    
    int l = 0, M = (N*PRE_CALC>>2) << 1;
    
    for(int i=0;i<N;i++)
        for(int j=0;j<PRE_CALC>>2;j++)
            a[i+j*N].r = sk[i].secLevel;
   for(int i=0;i<N;i++)
        for(int j=0;j<PRE_CALC>>2;j++)
            b[i+j*N].r = sign[i].secLevel;

    int n = (N - 1)*PRE_CALC>>2, m = n * 2;
    for (n = 1; n <= m; n *= 2)
        l++;
    for(int i=0;i<n;i++) r[i]=(r[i/2]/2)|((i&1)<<(l-1));
    FFT_cuda(a, r, 1, n);
    FFT_cuda(b, r, 1, n);
    for(int i=0;i<=n;i++) a[i]=a[i]*b[i];
    FFT_cuda(a, r, -1, n);
    for(int i=0;i<N;i++) sign[i].secLevel=(unsigned int)round(a[i].r/n);

    return;
}

void SM9_Verify_PreCalc(unsigned int N, SM9_PK *pk, SM9_Sign *sign)
{
    // Comp a[N<<2],b[N<<2];
    // int l = 0, M = N << 1;
    // int r[M<<2];
    
    // for(int i=0;i<N;i++)
    //     a[i].r = pk[i].keylen;
    // for(int i=0;i<N;i++)
    //     b[i].r = sign[i].secLevel;

    // int n = N - 1, m = n * 2;
    // for (n = 1; n <= m; n *= 2)
    //     l++;
    // for(int i=0;i<n;i++) r[i]=(r[i/2]/2)|((i&1)<<(l-1));
    // //FFT_cuda(a, r, 1, n);
    // //FFT_cuda(b, r, 1, n);
    // for(int i=0;i<=n;i++) a[i]=a[i]*b[i];
    // //FFT_cuda(a, r, -1, n);
    // for(int i=0;i<N;i++) sign[i].secLevel=(unsigned int)round(a[i].r/n);

    // return;
    int l = 0, M = (N*PRE_CALC) << 1;
    
    for(int i=0;i<N;i++)
        for(int j=0;j<PRE_CALC;j++)
            a[i+j*N].r = pk[i].keylen;
   for(int i=0;i<N;i++)
        for(int j=0;j<PRE_CALC;j++)
            b[i+j*N].r = sign[i].secLevel;

    int n = (N - 1)*PRE_CALC, m = n * 2;
    for (n = 1; n <= m; n *= 2)
        l++;
    for(int i=0;i<n;i++) r[i]=(r[i/2]/2)|((i&1)<<(l-1));
    FFT_cuda(a, r, 1, n);
    FFT_cuda(b, r, 1, n);
    for(int i=0;i<=n;i++) a[i]=a[i]*b[i];
    FFT_cuda(a, r, -1, n);
    for(int i=0;i<N;i++) sign[i].secLevel=(unsigned int)round(a[i].r/n);

    return;
}

__device__ int SM9_Init_cuda(unsigned int curve, BOOL TWIST_TYPE,unsigned int seclevel, unsigned char* t, unsigned char* q, unsigned char* a, unsigned char* b, unsigned char* n, unsigned char* xp1, unsigned char* yp1, unsigned char* xq1, unsigned char* xq2, unsigned char* yq1, unsigned char* yq2){
    miracl *mr_mip = mr_first_alloc_cuda();
    zzn2 xx;
    zzn2 yy;
	big x;
    big y;
#ifdef MR_OS_THREADS
    miracl *mr_mip = get_mip_cuda();
#endif
    mr_mip = mirsys_cuda(128,0);

      sm9q = mirvar_cuda(_MIPP_ 0);
      sm9a = mirvar_cuda(_MIPP_ 0);
      sm9b = mirvar_cuda(_MIPP_ 0);
      sm9n = mirvar_cuda(_MIPP_ 0);
      sm9t = mirvar_cuda(_MIPP_ 0);

        zzn2_mirvar_cuda(_MIPP_ &sm9X);

        p1G1 = *(epoint_init_cuda(_MIPPO_));
        ppG1 = *(epoint_init_cuda(_MIPPO_));
        keG1 = *(epoint_init_cuda(_MIPPO_));

        ecn2_mirvar_cuda(_MIPP_ &p2G2);
        ecn2_mirvar_cuda(_MIPP_ &ppG2);

        zzn12_mirvar_cuda(_MIPP_ &gGt);
        zzn12_mirvar_cuda(_MIPP_ &eGt);
        zzn12_mirvar_cuda(_MIPP_ &kGt);
        //========================

        zzn2_mirvar_cuda(_MIPP_ &xx);
        zzn2_mirvar_cuda(_MIPP_ &yy);

        x = mirvar_cuda(_MIPP_ 0);
        y = mirvar_cuda(_MIPP_ 0);

        if (curve){
            sm9len = seclevel;
            TWIST = TWIST_TYPE;
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)t, sm9t);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)q, sm9q);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)a, sm9a);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)b, sm9b);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)n, sm9n);

            mr_mip->TWIST = TWIST;
            ecurve_init_cuda(_MIPP_ sm9a,sm9b,sm9q,MR_PROJECTIVE);

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)xp1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)yp1, y);

            if (!epoint_set_cuda(_MIPP_ x, y, 1, &p1G1)) {
                mirkill_cuda(sm9q);
                mirkill_cuda(sm9a);
                mirkill_cuda(sm9b);
                mirkill_cuda(sm9n);
                mirkill_cuda(sm9t);

                zzn2_kill_cuda(_MIPP_ &sm9X);
    //            epoint_free_cuda(&p1G1);
    //            epoint_free_cuda(&ppG1);
    //            epoint_free_cuda(&keG1);
                ecn2_kill_cuda(_MIPP_ &p2G2);
                ecn2_kill_cuda(_MIPP_ &ppG2);
                zzn12_kill_cuda(_MIPP_ &gGt);
                zzn12_kill_cuda(_MIPP_ &eGt);
                zzn12_kill_cuda(_MIPP_ &kGt);

                zzn2_kill_cuda(_MIPP_ &xx);
                zzn2_kill_cuda(_MIPP_ &yy);
                mirkill_cuda(x);
                mirkill_cuda(y);

                CloseMiracl_cuda(_MIPPO_);
                return NOT_ON_G1;
            }

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)xq1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)xq2, y);
            zzn2_from_bigs_cuda(_MIPP_ x, y, &xx);

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)yq1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)yq2, y);
            zzn2_from_bigs_cuda(_MIPP_ x, y, &yy);

            if(!ecn2_set_cuda(_MIPP_ &xx, &yy, &p2G2)){
                mirkill_cuda(sm9q);
                mirkill_cuda(sm9a);
                mirkill_cuda(sm9b);
                mirkill_cuda(sm9n);
                mirkill_cuda(sm9t);

                zzn2_kill_cuda(_MIPP_ &sm9X);
    //            epoint_free_cuda(&p1G1);
    //            epoint_free_cuda(&ppG1);
    //            epoint_free_cuda(&keG1);
                ecn2_kill_cuda(_MIPP_ &p2G2);
                ecn2_kill_cuda(_MIPP_ &ppG2);
                zzn12_kill_cuda(_MIPP_ &gGt);
                zzn12_kill_cuda(_MIPP_ &eGt);
                zzn12_kill_cuda(_MIPP_ &kGt);

                zzn2_kill_cuda(_MIPP_ &xx);
                zzn2_kill_cuda(_MIPP_ &yy);
                mirkill_cuda(x);
                mirkill_cuda(y);

                CloseMiracl_cuda(_MIPPO_);
                return NOT_ON_G2;
            }
            set_frobenius_constant_cuda(_MIPP_ &sm9X);
        }else{
            sm9len = 32;
            TWIST = twist;
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)ct, sm9t);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cq, sm9q);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)ca, sm9a);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cb, sm9b);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cn, sm9n);

            mr_mip->TWIST = TWIST;
            ecurve_init_cuda(_MIPP_ sm9a,sm9b,sm9q,MR_PROJECTIVE);

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cxp1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cyp1, y);

            if (!epoint_set_cuda(_MIPP_ x, y, 1, &p1G1)) {
                mirkill_cuda(sm9q);
                mirkill_cuda(sm9a);
                mirkill_cuda(sm9b);
                mirkill_cuda(sm9n);
                mirkill_cuda(sm9t);

                zzn2_kill_cuda(_MIPP_ &sm9X);
     //           epoint_free_cuda(&p1G1);
     //           epoint_free_cuda(&ppG1);
     //           epoint_free_cuda(&keG1);
                ecn2_kill_cuda(_MIPP_ &p2G2);
                ecn2_kill_cuda(_MIPP_ &ppG2);
                zzn12_kill_cuda(_MIPP_ &gGt);
                zzn12_kill_cuda(_MIPP_ &eGt);
                zzn12_kill_cuda(_MIPP_ &kGt);

                zzn2_kill_cuda(_MIPP_ &xx);
                zzn2_kill_cuda(_MIPP_ &yy);
                mirkill_cuda(x);
                mirkill_cuda(y);

                CloseMiracl_cuda(_MIPPO_);
                return NOT_ON_G1;
            }

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cxq1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cxq2, y);
            zzn2_from_bigs_cuda(_MIPP_ x, y, &xx);

            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cyq1, x);
            bytes_to_big_cuda(_MIPP_ sm9len, (char *)cyq2, y);
            zzn2_from_bigs_cuda(_MIPP_ x, y, &yy);

            if(!ecn2_set_cuda(_MIPP_ &xx, &yy, &p2G2)){
                mirkill_cuda(sm9q);
                mirkill_cuda(sm9a);
                mirkill_cuda(sm9b);
                mirkill_cuda(sm9n);
                mirkill_cuda(sm9t);

                zzn2_kill_cuda(_MIPP_ &sm9X);
     //           epoint_free_cuda(&p1G1);
     //           epoint_free_cuda(&ppG1);
     //           epoint_free_cuda(&keG1);
                ecn2_kill_cuda(_MIPP_ &p2G2);
                ecn2_kill_cuda(_MIPP_ &ppG2);
                zzn12_kill_cuda(_MIPP_ &gGt);
                zzn12_kill_cuda(_MIPP_ &eGt);
                zzn12_kill_cuda(_MIPP_ &kGt);

                zzn2_kill_cuda(_MIPP_ &xx);
                zzn2_kill_cuda(_MIPP_ &yy);
                mirkill_cuda(x);
                mirkill_cuda(y);

                CloseMiracl_cuda(_MIPPO_);
                return NOT_ON_G2;
            }
            bytes_to_big_cuda(_MIPP_ sm9len, sm9Xx, x);
            zzn2_from_big_cuda(_MIPP_ x, &sm9X);
        }
        zzn2_kill_cuda(_MIPP_ &xx);
        zzn2_kill_cuda(_MIPP_ &yy);
        mirkill_cuda(x);
        mirkill_cuda(y);

        CloseMiracl_cuda(_MIPPO_);
    return 0;
}

__device__ void SM9_Free_cuda(){
    if (sm9init == FALSE) return;
    miracl *mr_mip = mr_first_alloc_cuda();
    mirkill_cuda(sm9q);
    mirkill_cuda(sm9a);
    mirkill_cuda(sm9b);
    mirkill_cuda(sm9n);
    mirkill_cuda(sm9t);
    
    zzn2_kill_cuda(_MIPP_ &sm9X);
 //   epoint_free_cuda(&p1G1);
//    epoint_free_cuda(&ppG1);
//    epoint_free_cuda(&keG1);
    ecn2_kill_cuda(_MIPP_ &p2G2);
    ecn2_kill_cuda(_MIPP_ &ppG2);
    zzn12_kill_cuda(_MIPP_ &gGt);
    zzn12_kill_cuda(_MIPP_ &eGt);
    zzn12_kill_cuda(_MIPP_ &kGt);
    CloseMiracl_cuda(_MIPPO_);
    sm9init = FALSE;
    sm9sign = FALSE;
    sm9encrypt = FALSE;
    sm9keyexchange = FALSE;
    
    return;
}

__device__ unsigned char* SM9_Set_Sign_cuda(unsigned char* x1, unsigned char* x2, unsigned char* y1, unsigned char* y2, unsigned char* gGtchar){
    miracl *mr_mip;
    big x,y;
    zzn2 xx,yy;
    unsigned char *gc;
	if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return NULL;
    }
    mr_mip = GenMiracl_cuda(sm9len);
    
    x = mirvar_cuda(_MIPP_ 0);
    y = mirvar_cuda(_MIPP_ 0);
    zzn2_mirvar_cuda(_MIPP_ &xx);
    zzn2_mirvar_cuda(_MIPP_ &yy);
    
    bytes_to_big_cuda(_MIPP_ sm9len, (char *)x2, x);
    bytes_to_big_cuda(_MIPP_ sm9len, (char *)x1, y);
    zzn2_from_bigs_cuda(_MIPP_ x, y, &xx);
    
    bytes_to_big_cuda(_MIPP_ sm9len, (char *)y2, x);
    bytes_to_big_cuda(_MIPP_ sm9len, (char *)y1, y);
    zzn2_from_bigs_cuda(_MIPP_ x, y, &yy);
    
    if(!ecn2_set_cuda(_MIPP_ &xx, &yy, &ppG2)){
        mirkill_cuda(x);
        mirkill_cuda(y);
        zzn2_kill_cuda(_MIPP_ &xx);
        zzn2_kill_cuda(_MIPP_ &yy);
        CloseMiracl_cuda(_MIPPO_);
        return NULL;
    };
    
    if(gGtchar == NULL){
        sm9sign = ecap_cuda(_MIPP_ &ppG2, &p1G1, sm9t, &sm9X, &gGt);
        gc = (unsigned char*)malloc_cuda(sizeof(unsigned char)*(12*sm9len));
        zzn12_tochar_cuda(_MIPP_ &gGt, gc,sm9len);
    }else{
        zzn12_fromchar_cuda(_MIPP_ &gGt, gGtchar,sm9len);
        sm9sign = TRUE;
        gc = gGtchar;
    }
    mirkill_cuda(x);
    mirkill_cuda(y);
    zzn2_kill_cuda(_MIPP_ &xx);
    zzn2_kill_cuda(_MIPP_ &yy);
    CloseMiracl_cuda(_MIPPO_);
    return gc;
}

__device__ void SM9_Close_Sign_cuda(){
    sm9init = FALSE;
}

__device__ unsigned char* SM9_Set_Encrypt_cuda(unsigned char* x, unsigned char* y, unsigned char* eGtchar){
    miracl *mr_mip;
    big a,b;
    unsigned char *gc;
	if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return NULL;
    }
    mr_mip = GenMiracl_cuda(sm9len);

    a = mirvar_cuda(_MIPP_ 0);
    b = mirvar_cuda(_MIPP_ 0);
    
    bytes_to_big_cuda(_MIPP_ sm9len, (char *) x, a);
    bytes_to_big_cuda(_MIPP_ sm9len, (char *) y, b);
    
    if(!epoint_set_cuda(_MIPP_ a, b, 1, &ppG1)){
        mirkill_cuda(a);
        mirkill_cuda(b);
        CloseMiracl_cuda(_MIPPO_);
        return NULL;
    }
    
    if (eGtchar == NULL){
        sm9encrypt = ecap_cuda(_MIPP_ &p2G2, &ppG1, sm9t, &sm9X, &eGt);
        gc = (unsigned char*)malloc_cuda(sizeof(unsigned char)*12*sm9len);
        zzn12_tochar_cuda(_MIPP_ &eGt, gc,sm9len);
    }else{
        zzn12_fromchar_cuda(_MIPP_ &eGt, eGtchar,sm9len);
        sm9encrypt = TRUE;
        gc = eGtchar;
    }
    mirkill_cuda(a);
    mirkill_cuda(b);
    CloseMiracl_cuda(_MIPPO_);
    return gc;
}

__device__ void SM9_Close_Encrypt_cuda(){
    sm9encrypt = FALSE;
}

__device__ unsigned char* SM9_Set_KeyExchange_cuda(unsigned char* x, unsigned char* y,unsigned char* kGtchar){
    miracl *mr_mip;
    big a,b;
    unsigned char* gc;
    if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return NULL;
    }
    mr_mip = GenMiracl_cuda(sm9len);
    a = mirvar_cuda(_MIPP_ 0);
    b = mirvar_cuda(_MIPP_ 0);
    
    bytes_to_big_cuda(_MIPP_ sm9len, (char *) x, a);
    bytes_to_big_cuda(_MIPP_ sm9len, (char *) y, b);
    
    if (!epoint_set_cuda(_MIPP_ a, b, 1, &keG1)){
        mirkill_cuda(a);
        mirkill_cuda(b);
        CloseMiracl_cuda(_MIPPO_);
        return NULL;
    }
    if (kGtchar == NULL){
        
        sm9keyexchange = ecap_cuda(_MIPP_ &p2G2, &keG1, sm9t, &sm9X, &kGt);
        gc = (unsigned char*)malloc_cuda(sizeof(unsigned char)*12*sm9len);
        zzn12_tochar_cuda(_MIPP_ &kGt, gc,sm9len);
    }else{
        zzn12_fromchar_cuda(_MIPP_ &kGt, kGtchar,sm9len);
        sm9keyexchange = TRUE;
        gc = kGtchar;
    }
    mirkill_cuda(a);
    mirkill_cuda(b);
    CloseMiracl_cuda(_MIPPO_);
    return gc;
}

__device__ void SM9_Close_KeyExchange_cuda(){
    sm9keyexchange = FALSE;
}

__device__ SM9_MSK SM9_MSK_New_cuda(int secLevel,unsigned char* w){
    SM9_MSK msk;
    msk.keylen = secLevel;
    msk.msk = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    memcpy(msk.msk, w, secLevel);
    return msk;
}

__device__ SM9_MSPK SM9_MSPK_New_cuda(int secLevel){
    SM9_MSPK mpk;
    mpk.secLevel = secLevel;
    mpk.x1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    mpk.x2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    mpk.y1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    mpk.y2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return mpk;
}

__device__ SM9_MCPK SM9_MCPK_New_cuda(int secLevel){
    SM9_MCPK mpk;
    mpk.secLevel = secLevel;
    mpk.x = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    mpk.y = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return mpk;
}

__device__ SM9_PK SM9_PK_New_cuda(int len,unsigned char* w){
    SM9_PK pk;
    pk.keylen = len;
    pk.pk = (unsigned char *)malloc_cuda(sizeof(char)*len);
    memcpy(pk.pk, w, len);
    return pk;
}

__device__ SM9_SSK SM9_SSK_New_cuda(int secLevel){
    SM9_SSK sk;
    sk.secLevel = secLevel;
    sk.x = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.y = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return sk;
}

__device__ SM9_CSK SM9_CSK_New_cuda(int secLevel){
    SM9_CSK sk;
    sk.secLevel = secLevel;
    sk.x1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.x2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.y1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.y2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return sk;
}

__device__ SM9_Sign SM9_Sign_New_cuda(int secLevel){
    SM9_Sign sign;
    sign.secLevel = secLevel;
    sign.type = 0x04;
    sign.h = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sign.xs = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sign.ys = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return sign;
}

__device__ SM9_Cipher SM9_Cipher_New_cuda(int secLevel){
    SM9_Cipher cip;
    cip.secLevel = secLevel;
    cip.cplen = 0;
    cip.x = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    cip.y = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    cip.c3 = (unsigned char *)malloc_cuda(sizeof(char)*SM3_HASH_LEN);
    return cip;
}

__device__ SM9_MKPK SM9_MKPK_New_cuda(int secLevel){
    SM9_MKPK mpk;
    mpk.secLevel = secLevel;
    mpk.x = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    mpk.y = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return mpk;
}

__device__ SM9_KSK SM9_KSK_New_cuda(int secLevel){
    SM9_KSK sk;
    sk.secLevel = secLevel;
    sk.x1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.x2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.y1 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    sk.y2 = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return sk;
}

__device__ SM9_Send SM9_Send_New_cuda(int secLevel){
    SM9_Send se;
    se.secLevel = secLevel;
    se.x = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    se.y = (unsigned char *)malloc_cuda(sizeof(char)*secLevel);
    return se;
}

__device__ void SM9_MSK_Free_cuda(SM9_MSK *msk){
    free(msk->msk);
}

__device__ void SM9_MSPK_Free_cuda(SM9_MSPK *mpk){
    free(mpk->x1);
    free(mpk->x2);
    free(mpk->y1);
    free(mpk->y2);
}

__device__ void SM9_MCPK_Free_cuda(SM9_MCPK *mpk){
    free(mpk->x);
    free(mpk->y);
}

__device__ void SM9_PK_Free_cuda(SM9_PK *pk){
    free(pk->pk);
}

__device__ void SM9_SSK_Free_cuda(SM9_SSK *sk){
    free(sk->x);
    free(sk->y);
}

__device__ void SM9_CSK_Free_cuda(SM9_CSK *sk){
    free(sk->x1);
    free(sk->x2);
    free(sk->y1);
    free(sk->y2);
}

__device__ void SM9_MKPK_Free_cuda(SM9_MKPK *mpk){
    free(mpk->x);
    free(mpk->y);
}

__device__ void SM9_KSK_Free_cuda(SM9_KSK *sk){
    free(sk->x1);
    free(sk->x2);
    free(sk->y1);
    free(sk->y2);
}

__device__ void SM9_Sign_Free_cuda(SM9_Sign *s){
    free(s->h);
    free(s->xs);
    free(s->ys);
}

__device__ void SM9_Cipher_Free_cuda(SM9_Cipher *c){
    free(c->c3);
    free(c->x);
    free(c->y);
    if(!c->cplen) free(c->cp);
}

__device__ void SM9_Send_Free_cuda(SM9_Send *s){
    free(s->x);
    free(s->y);
}

__device__ BOOL SM9_GenMSignPubKey_cuda(SM9_MSK *msk, SM9_MSPK *mspk){
    miracl *mr_mip;
    big mk;
	if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return !sm9init;
    }
    mr_mip = GenMiracl_cuda(mspk->secLevel);
	mk = mirvar_cuda(_MIPP_ 0);
    bytes_to_big_cuda(_MIPP_ mspk->secLevel, (char *)msk->msk, mk);
    
    ecn2_copy_cuda(&p2G2, &ppG2);
    ecn2_mul_cuda(_MIPP_ mk, &ppG2);
    ecn2_norm_cuda(_MIPP_ &ppG2);
    
    redc_cuda(_MIPP_ ppG2.x.a, mk);
    big_to_bytes_cuda(_MIPP_ mspk->secLevel, mk, (char *)mspk->x2, TRUE);
    
    redc_cuda(_MIPP_ ppG2.x.b, mk);
    big_to_bytes_cuda(_MIPP_ mspk->secLevel, mk, (char *)mspk->x1, TRUE);
    
    redc_cuda(_MIPP_ ppG2.y.a, mk);
    big_to_bytes_cuda(_MIPP_ mspk->secLevel, mk, (char *)mspk->y2, TRUE);
    
    redc_cuda(_MIPP_ ppG2.y.b, mk);
    big_to_bytes_cuda(_MIPP_ mspk->secLevel, mk, (char *)mspk->y1, TRUE);
    
    mirkill_cuda(mk);
    CloseMiracl_cuda(_MIPPO_);
    return 0;
}

__device__ BOOL SM9_GenMEncryptPubKey_cuda(SM9_MSK *msk, SM9_MCPK *mcpk){
    miracl *mr_mip;
    big mk;
	if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return !sm9init;
    }
    mr_mip = GenMiracl_cuda(mcpk->secLevel);
	mk = mirvar_cuda(_MIPP_ 0);
    bytes_to_big_cuda(_MIPP_ mcpk->secLevel, (char *)msk->msk, mk);
    
    ecurve_mult_cuda(_MIPP_ mk, &p1G1, &ppG1);
    epoint_norm_cuda(_MIPP_ &ppG1);

    redc_cuda(_MIPP_ ppG1.X, mk);
    big_to_bytes_cuda(_MIPP_ mcpk->secLevel, mk, (char *)mcpk->x, TRUE);
    redc_cuda(_MIPP_ ppG1.Y, mk);
    big_to_bytes_cuda(_MIPP_ mcpk->secLevel, mk, (char *)mcpk->y, TRUE);
    
    mirkill_cuda(mk);
    CloseMiracl_cuda(_MIPPO_);
    return 0;
}

__device__ BOOL SM9_GenMKeyExchangePubKey_cuda(SM9_MSK *msk, SM9_MKPK *mcpk){
    miracl *mr_mip;
    big mk;
    
    if (!sm9init){
    //    printf("the sm9 lib is not init, please run SM9_INIT function");
        return !sm9init;
    }
    mr_mip = GenMiracl_cuda(mcpk->secLevel);
    mk = mirvar_cuda(_MIPP_ 0);
    
    bytes_to_big_cuda(_MIPP_ mcpk->secLevel, (char *)msk->msk, mk);
    
    ecurve_mult_cuda(_MIPP_ mk, &p1G1, &keG1);
    epoint_norm_cuda(_MIPP_ &keG1);
    
    redc_cuda(_MIPP_ keG1.X, mk);
    big_to_bytes_cuda(_MIPP_ mcpk->secLevel, mk, (char *)mcpk->x, TRUE);
    redc_cuda(_MIPP_ keG1.Y, mk);
    big_to_bytes_cuda(_MIPP_ mcpk->secLevel, mk, (char *)mcpk->y, TRUE);
    
    mirkill_cuda(mk);
    CloseMiracl_cuda(_MIPPO_);
    return 0;
}
#endif

#ifndef sm9_signature_c
#define sm9_signature_c


__device__ BOOL SM9_GenSignSecKey_cuda(SM9_SSK *sk, SM9_PK *pk,SM9_MSK *msk){
    miracl *mr_mip;
    unsigned char *id;
    big ssk,k;
    ecn *ppk;
    if (!sm9sign){
    //    printf("the sm9 sign lib is not init, please run SM9_SET_SIGN function\n");
        return LIB_NOT_INIT;
    }
    
    
    mr_mip = GenMiracl_cuda(sk->secLevel);
    id = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(pk->keylen+1));
    memcpy(id, pk->pk, pk->keylen);
    memcpy(id+pk->keylen, hid, 1);
    
    k = mirvar_cuda(_MIPP_ 0);
    ppk = epoint_init_cuda(_MIPPO_);
    
    
    
    ssk = Hfun_cuda(_MIPP_ (char *)id,pk->keylen+1,sk->secLevel,1);
    
    bytes_to_big_cuda(_MIPP_ msk->keylen, (char *)msk->msk, k);
    
    add_cuda(_MIPP_ ssk, k, ssk);
    divide_cuda(_MIPP_ ssk, sm9n, sm9n);
    
    xgcd_cuda(_MIPP_ ssk, sm9n, ssk, ssk, ssk);
    multiply_cuda(_MIPP_ ssk,k,ssk);
    divide_cuda(_MIPP_ ssk, sm9n, sm9n);
    
    ecurve_mult_cuda(_MIPP_ ssk, &p1G1, ppk);
    epoint_norm_cuda(_MIPP_ ppk);
    
    epoint_get_cuda(_MIPP_ ppk, ssk, k);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, ssk, (char *)sk->x, TRUE);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, k, (char *)sk->y, TRUE);
    
    mirkill_cuda(ssk);
    mirkill_cuda(k);
    epoint_free_cuda(ppk);
    free(id);
    CloseMiracl_cuda(_MIPPO_);
    return 0;
}

__device__ int SM9_Signature_cuda(unsigned char* mes,unsigned int meslen,unsigned char* ran,SM9_SSK *sk, SM9_Sign *sign){
    miracl *mr_mip;
	big x,y,r,zero,h;
	zzn12 w;
	ecn *pa;
	int mwlen;
    unsigned char *mw;
	if (!sm9sign){
    //    printf("the sm9 sign lib is not init, please run SM9_SET_SIGN function\n");
        return LIB_NOT_INIT;
    }
    mr_mip = GenMiracl_cuda(sm9len);
    x = mirvar_cuda(_MIPP_ 0);
    y = mirvar_cuda(_MIPP_ 0);
    r = mirvar_cuda(_MIPP_ 0);
	h = mirvar_cuda(_MIPP_ 0);
	zero = mirvar_cuda(_MIPP_ 0);
    
    zzn12_mirvar_cuda(_MIPP_ &w);
    
    
    pa = epoint_init_cuda(_MIPPO_);
    
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)sk->x, x);
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)sk->y, y);
    
    if (!epoint_set_cuda(_MIPP_ x, y, 1, pa)){
        zzn12_kill_cuda(_MIPP_ &w);
        mirkill_cuda(r);
        mirkill_cuda(x);
        mirkill_cuda(y);
        mirkill_cuda(h);
        epoint_free_cuda(pa);
        CloseMiracl_cuda(_MIPPO_);
        return  NOT_ON_G1;
    }
    
    zzn12_copy_cuda(&gGt, &w);
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)ran, r);
    zzn12_pow_cuda(_MIPP_ &w, r);
    
    mwlen = meslen+sk->secLevel*12;
    mw = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(mwlen));
    
    memcpy(mw, mes, meslen);
    
    zzn12_tochar_cuda(_MIPP_ &w, mw+meslen,sm9len);
    
    h = Hfun_cuda(_MIPP_ (char *)mw, mwlen, sk->secLevel, 2);
    
    subtract_cuda(_MIPP_ r, h, r);
    divide_cuda(_MIPP_ r, sm9n, sm9n);
    
    
    
    if (mr_compare_cuda(zero, r)==0){
        zzn12_kill_cuda(_MIPP_ &w);
        mirkill_cuda(r);
        mirkill_cuda(x);
        mirkill_cuda(y);
        mirkill_cuda(h);
        mirkill_cuda(zero);
        free(mw);
        epoint_free_cuda(pa);
        CloseMiracl_cuda(_MIPPO_);
        return  SIGN_ZERO_ERROR;
    }
    
    ecurve_mult_cuda(_MIPP_ r, pa, pa);
    epoint_norm_cuda(_MIPP_ pa);
    
    epoint_get_cuda(_MIPP_ pa, x, y);
    
    big_to_bytes_cuda(_MIPP_ sk->secLevel, x, (char *)sign->xs, TRUE);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, y, (char *)sign->ys, TRUE);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, h, (char *)sign->h, TRUE);
    
    zzn12_kill_cuda(_MIPP_ &w);
    mirkill_cuda(r);
    mirkill_cuda(x);
    mirkill_cuda(y);
    mirkill_cuda(h);
    mirkill_cuda(zero);
    free(mw);
    epoint_free_cuda(pa);
    CloseMiracl_cuda(_MIPPO_);
    return 0;
}

__device__ int SM9_Verify_cuda(unsigned char *mes,unsigned int meslen, SM9_Sign *sign, SM9_PK *pk, SM9_MSPK *mpk){
    miracl *mr_mip;
	
	
    
    int re;
    
    big h,x,y,h1,h2;
    ecn *S;
    ecn2 pp;
    ecn2 P;
    zzn12 t;
    zzn12 g;
    
    unsigned char *id;

	 int mwlen;
    unsigned char *mw;
    
    if (!sm9sign){
    //    printf("the sm9 sign lib is not init, please run SM9_SET_SIGN function\n");
        return LIB_NOT_INIT;
    }
	mr_mip = GenMiracl_cuda(sm9len);
    h = mirvar_cuda(_MIPP_ 0);
    x = mirvar_cuda(_MIPP_ 0);
    y = mirvar_cuda(_MIPP_ 0);
    S = epoint_init_cuda(_MIPPO_);
    ecn2_mirvar_cuda(_MIPP_ &pp);
    ecn2_mirvar_cuda(_MIPP_ &P);
    zzn12_mirvar_cuda(_MIPP_ &t);
    zzn12_mirvar_cuda(_MIPP_ &g);
    
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->h, h);
    
    if((mr_compare_cuda(h, sm9n) >= 0) || (mr_compare_cuda(h, x)) < 0){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        return VERIFY_ERROR_1;
    }
    
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->xs, x);
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->ys, y);
    
    if(!(epoint_set_cuda(_MIPP_ x, y, 1, S))){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        return NOT_ON_G1;
    }
    
    if (mpk == NULL){
        zzn12_copy_cuda(&gGt, &g);
        ecn2_copy_cuda(&ppG2, &pp);
    }else{
        //todo
        zzn12_copy_cuda(&gGt, &g);
        ecn2_copy_cuda(&ppG2, &pp);
    }
    ecn2_copy_cuda(&p2G2, &P);
    
    zzn12_pow_cuda(_MIPP_ &g, h);
    
    id = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(pk->keylen+1));
    memcpy(id, pk->pk, pk->keylen);
    memcpy(id+pk->keylen, hid, 1);
    
    
    h1 = Hfun_cuda(_MIPP_ (char *)id, pk->keylen+1, sign->secLevel, 1);
    
    
    
    ecn2_mul_cuda(_MIPP_ h1, &P);
    
    ecn2_add_cuda(_MIPP_ &pp, &P);
    
    //   ecn2_norm(_MIPP_ &P);
    
    if(!ecap_cuda(_MIPP_ &P, S, sm9t, &sm9X, &t)){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        free(id);
        return VERIFY_ERROR_2;
    }
    
    zzn12_mul_cuda(_MIPP_ &t, &g, &t);
    
    
    
   
    mwlen = meslen+sign->secLevel*12;
    mw = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(mwlen));
    memcpy(mw, mes, meslen);
    zzn12_tochar_cuda(_MIPP_ &t, mw+meslen,sm9len);
    
    
    h2 = Hfun_cuda(_MIPP_ (char *)mw, mwlen, sign->secLevel, 2);
    
    re = mr_compare_cuda(h2, h);
    if (re!=0) re=VERIFY_ERROR_3;
    mirkill_cuda(h);
    mirkill_cuda(h1);
    mirkill_cuda(h2);
    mirkill_cuda(x);
    mirkill_cuda(y);
    epoint_free_cuda(S);
    zzn12_kill_cuda(_MIPP_ &t);
    zzn12_kill_cuda(_MIPP_ &g);
    ecn2_kill_cuda(_MIPP_ &pp);
    ecn2_kill_cuda(_MIPP_ &P);
    free(id);
    free(mw);
    CloseMiracl_cuda(_MIPPO_);
    return re;
}


#endif

#ifndef sm9_cuda_cu
#define sm9_cuda_cu

__global__ void SM9_Signature_Init_Cuda(unsigned int N, unsigned char *cks, unsigned int cks_len, unsigned char *id, unsigned int id_len, unsigned char *rand, unsigned int rand_len, unsigned char *msg, unsigned int msg_len, SM9_PK *pk, SM9_SSK *sk, SM9_Sign *sign)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=N) return;
    cks += index * cks_len;
    id += index*id_len;
    rand += index*rand_len;
    msg += index*msg_len;
    pk += index;
    sk += index;
    sign += index;
    
    SM9_Init_cuda(0, 0, 32, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    sm9init=TRUE;
    // printf("index: %d\n", index);
    SM9_MSK msk = SM9_MSK_New_cuda(cks_len, cks); // 申明一个签名主密钥
    SM9_MSPK mspk = SM9_MSPK_New_cuda(cks_len); //申明一个主签名公钥

    SM9_GenMSignPubKey_cuda(&msk, &mspk); // 生成签名主公钥

    unsigned char *gg;

    gg = SM9_Set_Sign_cuda(mspk.x1, mspk.x2, mspk.y1, mspk.y2, NULL); // 启动签名lib

    if(gg==NULL)
    {
        printf("index: %d, init sign lib error\n",index);
        return;
    }

    *pk = SM9_PK_New_cuda(id_len, id); // 申明一个签名公钥
    *sk = SM9_SSK_New_cuda(cks_len);  // 申明一个签名私钥

    SM9_GenSignSecKey_cuda(sk, pk, &msk); // 由公钥（id）生成签名私钥

    *sign = SM9_Sign_New_cuda(cks_len); // 申明一个签名体

    //printf("secLevel: %d\ntype: %d\n", sign->secLevel, sign->type);

    return;
}

int SM9_Signature_Init_GPU(unsigned int N, unsigned char *cks_list, unsigned int cks_len, unsigned char *id_list, unsigned int id_len, unsigned char *rand_list, unsigned int rand_len, unsigned char *msg_list, unsigned int msg_len, SM9_PK *pk_list, SM9_SSK *sk_list, SM9_Sign *sign_list)
{
    unsigned char *cks_cuda;
    unsigned char *id_cuda;
    unsigned char *rand_cuda;
    unsigned char *msg_cuda;
    SM9_PK *pk_cuda;
    SM9_SSK *sk_cuda;
    SM9_Sign *sign_cuda;

    cudaMalloc((void **)&cks_cuda, N * cks_len * sizeof(unsigned char));
    cudaMalloc((void **)&id_cuda, N * id_len * sizeof(unsigned char));
    cudaMalloc((void **)&rand_cuda, N * rand_len * sizeof(unsigned char));
    cudaMalloc((void **)&msg_cuda, N * msg_len * sizeof(unsigned char));
    cudaMalloc((void **)&pk_cuda, N * sizeof(SM9_PK));
    cudaMalloc((void **)&sk_cuda, N * sizeof(SM9_SSK));
    cudaMalloc((void **)&sign_cuda, N * sizeof(SM9_Sign));

    cudaMemcpy(cks_cuda, cks_list, N * cks_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(id_cuda, id_list, N * id_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(rand_cuda, rand_list, N * rand_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(msg_cuda, msg_list, N * msg_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(pk_cuda, pk_list, N * sizeof(SM9_PK), cudaMemcpyHostToDevice);
    cudaMemcpy(sk_cuda, sk_list, N * sizeof(SM9_SSK), cudaMemcpyHostToDevice);
    cudaMemcpy(sign_cuda, sign_list, N * sizeof(SM9_Sign), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = (N + blockSize - 1) / blockSize;

    SM9_Signature_Init_Cuda<<<numBlock, blockSize>>>(N, cks_cuda, cks_len, id_cuda, id_len, rand_cuda, rand_len, msg_cuda, msg_len, pk_cuda, sk_cuda, sign_cuda);

    // Wait for GPU to finish before accessing on host
    // CPU需要等待cuda上的代码运行完毕，才能对数据进行读取
    cudaDeviceSynchronize();

    cudaMemcpy(cks_list, cks_cuda, N * cks_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(id_list, id_cuda, N * id_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(rand_list, rand_cuda, N * rand_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(msg_list, msg_cuda, N * msg_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(pk_list, pk_cuda, N * sizeof(SM9_PK), cudaMemcpyDeviceToHost);
    cudaMemcpy(sk_list, sk_cuda, N * sizeof(SM9_SSK), cudaMemcpyDeviceToHost);
    cudaMemcpy(sign_list, sign_cuda, N * sizeof(SM9_Sign), cudaMemcpyDeviceToHost);
    cudaFree(cks_cuda);
    cudaFree(id_cuda);
    cudaFree(rand_cuda);
    cudaFree(msg_cuda);
    cudaFree(pk_cuda);
    cudaFree(sk_cuda);
    cudaFree(sign_cuda);
    
    return 0;
}

__global__ void SM9_Signature_Cuda(unsigned int N, unsigned char* mes,\
    unsigned int meslen,unsigned char* ran,SM9_SSK *sk, SM9_Sign *sign){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=N) return;
    // printf("index: %d\n", index);
    mes += index * meslen;ran += index * sk->secLevel;
    sk += index;sign += index;miracl * mr_mip;
    big x,y,r,zero_cuda,h;zzn12 w;ecn *pa;int mwlen;unsigned char *mw;
    if (!sm9sign) return; // LIB_NOT_INIT;

    mr_mip = GenMiracl_cuda(sm9len);x = mirvar_cuda(_MIPP_ 0);
    y = mirvar_cuda(_MIPP_ 0);r = mirvar_cuda(_MIPP_ 0);
    h = mirvar_cuda(_MIPP_ 0);zero_cuda = mirvar_cuda(_MIPP_ 0);
    zzn12_mirvar_cuda(_MIPP_ &w);pa = epoint_init_cuda(_MIPPO_);
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)sk->x, x);
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)sk->y, y);
    if (!epoint_set_cuda(_MIPP_ x, y, 1, pa)){
        zzn12_kill_cuda(_MIPP_ &w); mirkill_cuda(r); mirkill_cuda(x);
        mirkill_cuda(y); mirkill_cuda(h);
        epoint_free_cuda(pa);CloseMiracl_cuda(_MIPPO_);
        return;//  NOT_ON_G1;
    }

    zzn12_copy_cuda(&gGt, &w);
    bytes_to_big_cuda(_MIPP_ sk->secLevel, (char *)ran, r);zzn12_pow_cuda(_MIPP_ &w, r);
    
    mwlen = meslen+sk->secLevel*12;
    mw = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(mwlen));
    memcpy(mw, mes, meslen);zzn12_tochar_cuda(_MIPP_ &w, mw+meslen,sm9len);
    
    h = Hfun_cuda(_MIPP_ (char *)mw, mwlen, sk->secLevel, 2);
    subtract_cuda(_MIPP_ r, h, r);divide_cuda(_MIPP_ r, sm9n, sm9n);
    
    if (mr_compare_cuda(zero_cuda, r)==0){
        zzn12_kill_cuda(_MIPP_ &w);mirkill_cuda(r);mirkill_cuda(x);mirkill_cuda(y);mirkill_cuda(h);
        mirkill_cuda(zero_cuda);free(mw);epoint_free_cuda(pa);CloseMiracl_cuda(_MIPPO_);
        return;//  SIGN_ZERO_ERROR;
    }
    
    ecurve_mult_cuda(_MIPP_ r, pa, pa);epoint_norm_cuda(_MIPP_ pa);epoint_get_cuda(_MIPP_ pa, x, y);
    
    big_to_bytes_cuda(_MIPP_ sk->secLevel, x, (char *)sign->xs, TRUE);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, y, (char *)sign->ys, TRUE);
    big_to_bytes_cuda(_MIPP_ sk->secLevel, h, (char *)sign->h, TRUE);
    
    zzn12_kill_cuda(_MIPP_ &w);
    mirkill_cuda(r);mirkill_cuda(x);mirkill_cuda(y);mirkill_cuda(h);mirkill_cuda(zero_cuda);
    free(mw);
    epoint_free_cuda(pa);
    CloseMiracl_cuda(_MIPPO_);
    return;
}

int SM9_Signature_GPU(unsigned int N, unsigned char *mes, unsigned int meslen,\
    unsigned char *ran, unsigned int ranlen, SM9_SSK *sk, SM9_Sign *sign)
{
    //SM9_Signature_PreCalc(N, sk, sign);

    unsigned char *mes_cuda;
    unsigned char *ran_cuda;
    SM9_SSK *sk_cuda;
    SM9_Sign *sign_cuda;

    cudaMalloc((void **)&mes_cuda, N * meslen * sizeof(unsigned char));
    cudaMalloc((void **)&ran_cuda, N * ranlen * sizeof(unsigned char));
    cudaMalloc((void **)&sk_cuda, N * sizeof(SM9_SSK));
    cudaMalloc((void **)&sign_cuda, N * sizeof(SM9_Sign));

    cudaMemcpy(mes_cuda, mes, N * meslen * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(ran_cuda, ran, N * ranlen * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(sk_cuda, sk, N * sizeof(SM9_SSK), cudaMemcpyHostToDevice);
    cudaMemcpy(sign_cuda, sign, N * sizeof(SM9_Sign), cudaMemcpyHostToDevice);
    
    int blockSize = 1024;
    int numBlock = (N + blockSize - 1) / blockSize;

    SM9_Signature_Cuda<<<numBlock, blockSize>>>(N, mes_cuda, meslen, ran_cuda, sk_cuda, sign_cuda);

    // Wait for GPU to finish before accessing on host
    // CPU需要等待cuda上的代码运行完毕，才能对数据进行读取
    cudaDeviceSynchronize();

    cudaMemcpy(sign, sign_cuda, N * sizeof(SM9_Sign), cudaMemcpyDeviceToHost);

    cudaFree(mes_cuda);
    cudaFree(ran_cuda);
    cudaFree(sk_cuda);
    cudaFree(sign_cuda);
    
    return 0;
}

__global__ void SM9_Verify_Cuda(unsigned int N, unsigned char *mes, unsigned int meslen, SM9_PK *pk, SM9_Sign *sign)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=N) return;
    mes += index*meslen;
    pk += index;
    sign += index;

    SM9_MSPK *mpk=NULL;

    miracl *mr_mip;

    int re;
    
    big h,x,y,h1,h2;
    ecn *S;
    ecn2 pp;
    ecn2 P;
    zzn12 t;
    zzn12 g;
    
    unsigned char *id;
    int mwlen;
    unsigned char *mw;
    
    if (!sm9sign){
    //    printf("the sm9 sign lib is not init, please run SM9_SET_SIGN function\n");
        return;// LIB_NOT_INIT;
    }
	mr_mip = GenMiracl_cuda(sm9len);
    h = mirvar_cuda(_MIPP_ 0);
    x = mirvar_cuda(_MIPP_ 0);
    y = mirvar_cuda(_MIPP_ 0);
    S = epoint_init_cuda(_MIPPO_);
    ecn2_mirvar_cuda(_MIPP_ &pp);
    ecn2_mirvar_cuda(_MIPP_ &P);
    zzn12_mirvar_cuda(_MIPP_ &t);
    zzn12_mirvar_cuda(_MIPP_ &g);
    
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->h, h);
    
    if((mr_compare_cuda(h, sm9n) >= 0) || (mr_compare_cuda(h, x)) < 0){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        return ;//VERIFY_ERROR_1;
    }
    
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->xs, x);
    bytes_to_big_cuda(_MIPP_ sign->secLevel, (char *)sign->ys, y);
    
    if(!(epoint_set_cuda(_MIPP_ x, y, 1, S))){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        return ;//NOT_ON_G1;
    }
    
    if (mpk == NULL){
        zzn12_copy_cuda(&gGt, &g);
        ecn2_copy_cuda(&ppG2, &pp);
    }else{
        //todo
        zzn12_copy_cuda(&gGt, &g);
        ecn2_copy_cuda(&ppG2, &pp);
    }
    ecn2_copy_cuda(&p2G2, &P);
    
    zzn12_pow_cuda(_MIPP_ &g, h);
    
    id = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(pk->keylen+1));
    memcpy(id, pk->pk, pk->keylen);
    memcpy(id+pk->keylen, hid, 1);
    
    
    h1 = Hfun_cuda(_MIPP_ (char *)id, pk->keylen+1, sign->secLevel, 1);
    
    ecn2_mul_cuda(_MIPP_ h1, &P);
    
    ecn2_add_cuda(_MIPP_ &pp, &P);
    
    //   ecn2_norm(_MIPP_ &P);
    
    if(!ecap_cuda(_MIPP_ &P, S, sm9t, &sm9X, &t)){
        mirkill_cuda(h);
        mirkill_cuda(x);
        mirkill_cuda(y);
        epoint_free_cuda(S);
        zzn12_kill_cuda(_MIPP_ &t);
        zzn12_kill_cuda(_MIPP_ &g);
        ecn2_kill_cuda(_MIPP_ &pp);
        ecn2_kill_cuda(_MIPP_ &P);
        free(id);
        return ;//VERIFY_ERROR_2;
    }
    
    zzn12_mul_cuda(_MIPP_ &t, &g, &t);
   
    mwlen = meslen+sign->secLevel*12;
    mw = (unsigned char *)malloc_cuda(sizeof(unsigned char)*(mwlen));
    memcpy(mw, mes, meslen);
    zzn12_tochar_cuda(_MIPP_ &t, mw+meslen,sm9len);
    
    h2 = Hfun_cuda(_MIPP_ (char *)mw, mwlen, sign->secLevel, 2);
    
    re = mr_compare_cuda(h2, h);
    if (re!=0) re=VERIFY_ERROR_3;
    mirkill_cuda(h);
    mirkill_cuda(h1);
    mirkill_cuda(h2);
    mirkill_cuda(x);
    mirkill_cuda(y);
    epoint_free_cuda(S);
    zzn12_kill_cuda(_MIPP_ &t);
    zzn12_kill_cuda(_MIPP_ &g);
    ecn2_kill_cuda(_MIPP_ &pp);
    ecn2_kill_cuda(_MIPP_ &P);
    free(id);
    free(mw);
    CloseMiracl_cuda(_MIPPO_);
    
}

int SM9_Verify_GPU(unsigned int N, unsigned char *msg_list, unsigned int msg_len, SM9_PK *pk_list, SM9_Sign *sign_list)
{
    //SM9_Verify_PreCalc(N, pk_list, sign_list);

    unsigned char *msg_cuda;
    SM9_PK *pk_cuda;
    SM9_Sign *sign_cuda;

    cudaMalloc((void **)&msg_cuda, N * msg_len * sizeof(unsigned char));//printf("...");
    cudaMalloc((void **)&pk_cuda, N * sizeof(SM9_PK));
    cudaMalloc((void **)&sign_cuda, N * sizeof(SM9_Sign));

    cudaMemcpy(msg_cuda, msg_list, N * msg_len * sizeof(unsigned char), cudaMemcpyHostToDevice);//printf("...");
    cudaMemcpy(pk_cuda, pk_list, N * sizeof(SM9_PK), cudaMemcpyHostToDevice);
    cudaMemcpy(sign_cuda, sign_list, N * sizeof(SM9_Sign), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = (N + blockSize - 1) / blockSize;//printf("...");

    SM9_Verify_Cuda<<<numBlock, blockSize>>>(N, msg_cuda, msg_len, pk_cuda, sign_cuda);

    // Wait for GPU to finish before accessing on host
    // CPU需要等待cuda上的代码运行完毕，才能对数据进行读取
    cudaDeviceSynchronize();//printf("...");

    cudaMemcpy(msg_list, msg_cuda, N * msg_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);//printf("...");
    cudaMemcpy(pk_list, pk_cuda, N * sizeof(SM9_PK), cudaMemcpyDeviceToHost);
    cudaMemcpy(sign_list, sign_cuda, N * sizeof(SM9_Sign), cudaMemcpyDeviceToHost);

    cudaFree(msg_cuda);//printf("...");
    cudaFree(pk_cuda);
    cudaFree(sign_cuda);

    return 0;
}

int main_GPU()
{
    clock_t start, end;
    double time;

    int res;

    printf("\n=============gpu-test-begin==============\n");
    //===========签名用到的参数==========
    unsigned char cks[32+1] = "\x86\xDC\xD6\x4F\xEB\x81\xA7\x19\x63\x59\x59\xF1\xA5\xC2\xF9\x88\xBD\x39\x43\x1B\x08"
                             "\xA8\x63\xF0\x42\x8D\x21\xDF\xFA\xF2\xBF\x89";
    unsigned char id[5+1] = {"\x41\x6C\x69\x63\x65"};
    unsigned char rand[32+1] = {"\x1A\x23\x29\x77\xBA\x9F\xA2\xD1\xC5\x58\xF2\xD4\x67\xFE\x7B\xE7\x04\x05\x41\x26\x73"
                               "\xF8\xBE\x64\x9B\xBD\xD4\xA0\x95\xBE\x1B\x4B"};
    unsigned char msg[20+1] = {"\x43\x68\x69\x6E\x65\x73\x65\x20\x49\x42\x53\x20\x73\x74\x61\x6E\x64\x61\x72\x64"};

    unsigned char cks_list[TEST_CUDA][32];
    unsigned char id_list[TEST_CUDA][5];
    unsigned char rand_list[TEST_CUDA][32];
    unsigned char msg_list[TEST_CUDA][20];

    for (int i = 0; i < TEST_CUDA;i++)
    {
        memcpy((unsigned char*)cks_list + i * 32, cks, 32 * sizeof(unsigned char));
        memcpy((unsigned char*)id_list + i * 5, id, 5 * sizeof(unsigned char));
        memcpy((unsigned char*)rand_list + i * 32, rand, 32 * sizeof(unsigned char));
        memcpy((unsigned char*)msg_list + i * 20, msg, 20 * sizeof(unsigned char));
    }

    SM9_PK pk_list[TEST_CUDA];
    SM9_SSK sk_list[TEST_CUDA];
    SM9_Sign sign_list[TEST_CUDA];

    SM9_Signature_Init_GPU(TEST_CUDA, (unsigned char *)cks_list, 32, (unsigned char *)id_list, 5, (unsigned char *)rand_list, 32, (unsigned char *)msg_list, 20, pk_list, sk_list, sign_list);

    start = clock();
    SM9_Signature_GPU(TEST_CUDA, (unsigned char *)msg_list, 20, (unsigned char *)rand_list, 32, sk_list, sign_list);
    end = clock();
    time = (double)(clock() - start) / (double)CLOCKS_PER_SEC *1000;
    printf("cuda sign %d time is %lf ms\n", TEST_CUDA, time);


    start = clock();
    SM9_Verify_GPU(TEST_CUDA, (unsigned char *)msg_list, 20, pk_list, sign_list);
    end = clock();
    time = (double)(clock() - start) / (double)CLOCKS_PER_SEC *1000;
    printf("cuda verify %d time is %lf ms\n", TEST_CUDA, time);

    printf("==============gpu-test-end===============\n\n");
    return 0;
}

#endif