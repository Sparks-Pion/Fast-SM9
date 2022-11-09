#ifndef __SM9_CUDA_DEF_CUH__
#define __SM9_CUDA_DEF_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "../sm9/sm9_algorithm.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*mirdef.cuh-begin*/
    #ifndef mirdef_cuh
    #define mirdef_cuh
    #define MR_GENERIC_MT
    #define MR_LITTLE_ENDIAN
    #define MIRACL 64
    #define mr_utype long
    #define mr_unsign64 unsigned long
    #define MR_IBITS 32
    #define MR_LBITS 64
    #define mr_unsign32 unsigned int
    #define MR_FLASH 52
    #define MAXBASE ((mr_small)1<<(MIRACL-1))
    #define MR_BITSINCHAR 8
    #endif
    /*mirdef.cuh-end*/

    /*miracl.cuh-begin*/
    #ifndef miracl_cuh
    #define miracl_cuh
    /* Some modifiable defaults... */
    /* Use a smaller buffer if space is limited, don't be so wasteful! */
    #ifdef MR_STATIC
    #define MR_DEFAULT_BUFFER_SIZE 260
    #else
    #define MR_DEFAULT_BUFFER_SIZE 1024
    #endif

    /* see mrgf2m.c */

    #ifndef MR_KARATSUBA
    #define MR_KARATSUBA 2
    #endif

    #ifndef MR_DOUBLE_BIG

    #ifdef MR_KCM
    #ifdef MR_FLASH
        #define MR_SPACES 32
    #else
        #define MR_SPACES 31
    #endif
    #else
    #ifdef MR_FLASH
    #define MR_SPACES 28
    #else
    #define MR_SPACES 27
    #endif
    #endif

    #else

    #ifdef MR_KCM
    #ifdef MR_FLASH
        #define MR_SPACES 44
    #else
        #define MR_SPACES 43
    #endif
    #else
    #ifdef MR_FLASH
        #define MR_SPACES 40
    #else
        #define MR_SPACES 39
    #endif
    #endif

    #endif

    /* To avoid name clashes - undefine this */

    /* #define compare mr_compare */

    #ifdef MR_AVR
    #include <avr/pgmspace.h>
    #endif

    /* size of bigs and elliptic curve points for memory allocation from stack or heap */

    #define MR_ROUNDUP(a,b) ((a)-1)/(b)+1

    #define MR_SL sizeof(long)

    #ifdef MR_STATIC

    #define MR_SIZE (((sizeof(struct bigtype)+(MR_STATIC+2)*sizeof(mr_utype))-1)/MR_SL+1)*MR_SL
    #define MR_BIG_RESERVE(n) ((n)*MR_SIZE+MR_SL)

    #ifdef MR_AFFINE_ONLY
    #define MR_ESIZE (((sizeof(epoint)+MR_BIG_RESERVE(2))-1)/MR_SL+1)*MR_SL
    #else
    #define MR_ESIZE (((sizeof(epoint)+MR_BIG_RESERVE(3))-1)/MR_SL+1)*MR_SL
    #endif
    #define MR_ECP_RESERVE(n) ((n)*MR_ESIZE+MR_SL)

    #define MR_ESIZE_A (((sizeof(epoint)+MR_BIG_RESERVE(2))-1)/MR_SL+1)*MR_SL
    #define MR_ECP_RESERVE_A(n) ((n)*MR_ESIZE_A+MR_SL)


    #endif

    /* useful macro to convert size of big in words, to size of required structure */

    #define mr_size(n) (((sizeof(struct bigtype)+((n)+2)*sizeof(mr_utype))-1)/MR_SL+1)*MR_SL
    #define mr_big_reserve(n,m) ((n)*mr_size(m)+MR_SL)

    #define mr_esize_a(n) (((sizeof(epoint)+mr_big_reserve(2,(n)))-1)/MR_SL+1)*MR_SL
    #define mr_ecp_reserve_a(n,m) ((n)*mr_esize_a(m)+MR_SL)

    #ifdef MR_AFFINE_ONLY
    #define mr_esize(n) (((sizeof(epoint)+mr_big_reserve(2,(n)))-1)/MR_SL+1)*MR_SL 
    #else
    #define mr_esize(n) (((sizeof(epoint)+mr_big_reserve(3,(n)))-1)/MR_SL+1)*MR_SL
    #endif
    #define mr_ecp_reserve(n,m) ((n)*mr_esize(m)+MR_SL)


    /* if basic library is static, make sure and use static C++ */

    #ifdef MR_STATIC
    #ifndef BIGS
    #define BIGS MR_STATIC
    #endif
    #ifndef ZZNS
    #define ZZNS MR_STATIC
    #endif
    #ifndef GF2MS
    #define GF2MS MR_STATIC
    #endif
    #endif

    #ifdef __ia64__
    #if MIRACL==64
    #define MR_ITANIUM
    #include <ia64intrin.h>
    #endif
    #endif

    #ifdef _M_X64
    #ifdef _WIN64
    #if MIRACL==64
    #define MR_WIN64
    #include <intrin.h>
    #endif
    #endif
    #endif

    #ifndef MR_NO_FILE_IO
    #include <stdio.h>
    #endif
    /* error returns */

    #define MR_ERR_BASE_TOO_BIG       1
    #define MR_ERR_DIV_BY_ZERO        2
    #define MR_ERR_OVERFLOW           3
    #define MR_ERR_NEG_RESULT         4
    #define MR_ERR_BAD_FORMAT         5
    #define MR_ERR_BAD_BASE           6
    #define MR_ERR_BAD_PARAMETERS     7
    #define MR_ERR_OUT_OF_MEMORY      8
    #define MR_ERR_NEG_ROOT           9
    #define MR_ERR_NEG_POWER         10
    #define MR_ERR_BAD_ROOT          11
    #define MR_ERR_INT_OP            12
    #define MR_ERR_FLASH_OVERFLOW    13
    #define MR_ERR_TOO_BIG           14
    #define MR_ERR_NEG_LOG           15
    #define MR_ERR_DOUBLE_FAIL       16
    #define MR_ERR_IO_OVERFLOW       17
    #define MR_ERR_NO_MIRSYS         18
    #define MR_ERR_BAD_MODULUS       19
    #define MR_ERR_NO_MODULUS        20
    #define MR_ERR_EXP_TOO_BIG       21
    #define MR_ERR_NOT_SUPPORTED     22
    #define MR_ERR_NOT_DOUBLE_LEN    23
    #define MR_ERR_NOT_IRREDUC       24
    #define MR_ERR_NO_ROUNDING       25
    #define MR_ERR_NOT_BINARY        26
    #define MR_ERR_NO_BASIS          27
    #define MR_ERR_COMPOSITE_MODULUS 28
    #define MR_ERR_DEV_RANDOM        29

    /* some useful definitions */

    #define forever for(;;)

    #define mr_abs(x)  ((x)<0? (-(x)) : (x))

    #ifndef TRUE
    #define TRUE 1
    #endif
    #ifndef FALSE
    #define FALSE 0
    #endif

    #define OFF 0
    #define ON 1
    #define PLUS 1
    #define MINUS (-1)

    #define M1 (MIRACL-1)
    #define M2 (MIRACL-2)
    #define M3 (MIRACL-3)
    #define M4 (MIRACL-4)
    #define TOPBIT ((mr_small)1<<M1)
    #define SECBIT ((mr_small)1<<M2)
    #define THDBIT ((mr_small)1<<M3)
    #define M8 (MIRACL-8)

    #define MR_MAXDEPTH 24
    /* max routine stack depth */
    /* big and flash variables consist of an encoded length, *
    * and an array of mr_smalls containing the digits       */

    #ifdef MR_COUNT_OPS
    extern __device__ int fpm2,fpi2,fpc,fpa,fpx;
    #endif

    typedef int BOOL;

    #define MR_BYTE unsigned char

    #ifdef MR_BITSINCHAR
    #if MR_BITSINCHAR == 8
    #define MR_TOBYTE(x) ((MR_BYTE)(x))
    #else
    #define MR_TOBYTE(x) ((MR_BYTE)((x)&0xFF))
    #endif
    #else
    #define MR_TOBYTE(x) ((MR_BYTE)(x))
    #endif

    #ifdef MR_FP

    typedef mr_utype mr_small;
    #ifdef mr_dltype
    typedef mr_dltype mr_large;
    #endif

    #define MR_DIV(a,b)    (modf((a)/(b),&dres),dres)

    #ifdef MR_FP_ROUNDING

    /* slightly dicey - for example the optimizer might remove the MAGIC ! */

        #define MR_LROUND(a)   ( ( (a) + MR_MAGIC ) - MR_MAGIC )
    #else
        #define MR_LROUND(a)   (modfl((a),&ldres),ldres)
    #endif

    #define MR_REMAIN(a,b) ((a)-(b)*MR_DIV((a),(b)))

    #else

    typedef unsigned mr_utype mr_small;
    #ifdef mr_dltype
    typedef unsigned mr_dltype mr_large;
    #endif
    #ifdef mr_qltype
    typedef unsigned mr_qltype mr_vlarge;
    #endif

    #define MR_DIV(a,b)    ((a)/(b))
    #define MR_REMAIN(a,b) ((a)%(b))
    #define MR_LROUND(a)   ((a))
    #endif


    /* It might be wanted to change this to unsigned long */

    typedef unsigned int mr_lentype;

    struct bigtype
    {
        mr_lentype len;
        mr_small *w;
    };

    typedef struct bigtype *big;
    typedef big zzn;

    typedef big flash;

    #define MR_MSBIT ((mr_lentype)1<<(MR_IBITS-1))

    #define MR_OBITS (MR_MSBIT-1)

    #if MIRACL >= MR_IBITS
    #define MR_TOOBIG (1<<(MR_IBITS-2))
    #else
    #define MR_TOOBIG (1<<(MIRACL-1))
    #endif

    #ifdef  MR_FLASH
    #define MR_EBITS (8*sizeof(double) - MR_FLASH)
    /* no of Bits per double exponent */
    #define MR_BTS 16
    #define MR_MSK 0xFFFF

    #endif

    /* Default Hash function output size in bytes */
    #define MR_HASH_BYTES     32

    /* Marsaglia & Zaman Random number generator */
    /*         constants      alternatives       */
    #define NK   37           /* 21 */
    #define NJ   24           /*  6 */
    #define NV   14           /*  8 */

    /* Use smaller values if memory is precious */

    #ifdef mr_dltype

    #ifdef MR_LITTLE_ENDIAN 
    #define MR_BOT 0
    #define MR_TOP 1
    #endif
    #ifdef MR_BIG_ENDIAN
    #define MR_BOT 1
    #define MR_TOP 0
    #endif

    union doubleword
    {
        mr_large d;
        mr_small h[2];
    };

    #endif

    /* chinese remainder theorem structures */

    typedef struct {
        big *C;
        big *V;
        big *M;
        int NP;
    } big_chinese;

    typedef struct {
        mr_utype *C;
        mr_utype *V;
        mr_utype *M;
        int NP;
    } small_chinese;

    /* Cryptographically strong pseudo-random number generator */

    typedef struct {
        mr_unsign32 ira[NK];  /* random number...   */
        int         rndptr;   /* ...array & pointer */
        mr_unsign32 borrow;
        int pool_ptr;
        char pool[MR_HASH_BYTES];    /* random pool */
    } csprng;

    /* secure hash Algorithm structure */

    typedef struct {
        mr_unsign32 length[2];
        mr_unsign32 h[8];
        mr_unsign32 w[80];
    } sha256;

    typedef sha256 sha;

    #ifdef mr_unsign64

    typedef struct {
        mr_unsign64 length[2];
        mr_unsign64 h[8];
        mr_unsign64 w[80];
    } sha512;

    typedef sha512 sha384;

    typedef struct {
        mr_unsign64 length;
        mr_unsign64 S[5][5];
        int rate,len;
    } sha3;

    #endif

    /* Symmetric Encryption algorithm structure */

    #define MR_ECB   0
    #define MR_CBC   1
    #define MR_CFB1  2
    #define MR_CFB2  3
    #define MR_CFB4  5
    #define MR_PCFB1 10
    #define MR_PCFB2 11
    #define MR_PCFB4 13
    #define MR_OFB1  14
    #define MR_OFB2  15
    #define MR_OFB4  17
    #define MR_OFB8  21
    #define MR_OFB16 29

    typedef struct {
        int Nk,Nr;
        int mode;
        mr_unsign32 fkey[60];
        mr_unsign32 rkey[60];
        char f[16];
    } aes;

    /* AES-GCM suppport. See mrgcm.c */

    #define GCM_ACCEPTING_HEADER 0
    #define GCM_ACCEPTING_CIPHER 1
    #define GCM_NOT_ACCEPTING_MORE 2
    #define GCM_FINISHED 3
    #define GCM_ENCRYPTING 0
    #define GCM_DECRYPTING 1

    typedef struct {
        mr_unsign32 table[128][4]; /* 2k bytes */
        MR_BYTE stateX[16];
        MR_BYTE Y_0[16];
        mr_unsign32 counter;
        mr_unsign32 lenA[2],lenC[2];
        int status;
        aes a;
    } gcm;

    /* Elliptic curve point status */

    #define MR_EPOINT_GENERAL    0
    #define MR_EPOINT_NORMALIZED 1
    #define MR_EPOINT_INFINITY   2

    #define MR_NOTSET     0
    #define MR_PROJECTIVE 0
    #define MR_AFFINE     1
    #define MR_BEST       2
    #define MR_TWIST      8

    #define MR_OVER       0
    #define MR_ADD        1
    #define MR_DOUBLE     2

    /* Twist type */

    #define MR_QUADRATIC 2
    #define MR_CUBIC_M   0x3A
    #define MR_CUBIC_D   0x3B
    #define MR_QUARTIC_M 0x4A
    #define MR_QUARTIC_D 0x4B
    #define MR_SEXTIC_M  0x6A
    #define MR_SEXTIC_D  0x6B


    /* Fractional Sliding Windows for ECC - how much precomputation storage to use ? */
    /* Note that for variable point multiplication there is an optimal value 
    which can be reduced if space is short. For fixed points its a matter of 
    how much ROM is available to store precomputed points.
    We are storing the k points (P,3P,5P,7P,...,[2k-1].P) */

    /* These values can be manually tuned for optimal performance... */

    #ifdef MR_SMALL_EWINDOW
    #define MR_ECC_STORE_N  3   /* point store for ecn  variable point multiplication */
    #define MR_ECC_STORE_2M 3   /* point store for ec2m variable point multiplication */
    #define MR_ECC_STORE_N2 3   /* point store for ecn2 variable point multiplication */
    #else
    #define MR_ECC_STORE_N  8   /* 8/9 is close to optimal for 256 bit exponents */
    #define MR_ECC_STORE_2M 9
    #define MR_ECC_STORE_N2 8
    #endif

    /*#define MR_ECC_STORE_N2_PRECOMP MR_ECC_STORE_N2 */
    /* Might want to make this bigger.. */

    /* If multi-addition is of m points, and s precomputed values are required, this is max of m*s (=4.10?) */
    #define MR_MAX_M_T_S 64

    /* Elliptic Curve epoint structure. Uses projective (X,Y,Z) co-ordinates */

    typedef struct {
        int marker;
        big X;
        big Y;
    #ifndef MR_AFFINE_ONLY
        big Z;
    #endif
    } epoint;


    /* Structure for Comb method for finite *
    field exponentiation with precomputation */

    typedef struct {
    #ifdef MR_STATIC
        const mr_small *table;
    #else
        mr_small *table;
    #endif
        big n;
        int window;
        int max;
    } brick;

    /* Structure for Comb method for elliptic *
    curve exponentiation with precomputation  */

    typedef struct {
    #ifdef MR_STATIC
        const mr_small *table;
    #else
        mr_small *table;
    #endif
        big a,b,n;
        int window;
        int max;
    } ebrick;

    typedef struct {
    #ifdef MR_STATIC
        const mr_small *table;
    #else
        mr_small *table;
    #endif
        big a6,a2;
        int m,a,b,c;
        int window;
        int max;
    } ebrick2;

    typedef struct
    {
        big a;
        big b;
    } zzn2;

    typedef struct
    {
        zzn2 a;
        zzn2 b;
        BOOL unitary;
    } zzn4;

    typedef struct
    {
        int marker;
        zzn2 x;
        zzn2 y;
    #ifndef MR_AFFINE_ONLY
        zzn2 z;
    #endif

    } ecn2;

    typedef struct
    {
        big a;
        big b;
        big c;
    } zzn3;

    typedef struct
    {
        zzn2 a;
        zzn2 b;
        zzn2 c;
    } zzn6_3x2;

    /* main MIRACL instance structure */

    /* ------------------------------------------------------------------------*/

    typedef struct {
        mr_small base;       /* number base     */
        mr_small apbase;     /* apparent base   */
        int   pack;          /* packing density */
        int   lg2b;          /* bits in base    */
        mr_small base2;      /* 2^mr_lg2b          */
        BOOL (*user)(void);  /* pointer to user supplied function */

        int   nib;           /* length of bigs  */
    #ifndef MR_STRIPPED_DOWN
        int   depth;                 /* error tracing ..*/
        int   trace[MR_MAXDEPTH];    /* .. mechanism    */
    #endif
        BOOL  check;         /* overflow check  */
        BOOL  fout;          /* Output to file   */
        BOOL  fin;           /* Input from file  */
        BOOL  active;

    #ifndef MR_NO_FILE_IO

        FILE  *infile;       /* Input file       */
        FILE  *otfile;       /* Output file      */

    #endif


    #ifndef MR_NO_RAND
        mr_unsign32 ira[NK];  /* random number...   */
        int         rndptr;   /* ...array & pointer */
        mr_unsign32 borrow;
    #endif

        /* Montgomery constants */
        mr_small ndash;
        big modulus;
        big pR;
        BOOL ACTIVE;
        BOOL MONTY;

        /* Elliptic Curve details   */
    #ifndef MR_NO_SS
        BOOL SS;               /* True for Super-Singular  */
    #endif
    #ifndef MR_NOKOBLITZ
        BOOL KOBLITZ;          /* True for a Koblitz curve */
    #endif
    #ifndef MR_AFFINE_ONLY
        int coord;
    #endif
        int Asize,Bsize;

        int M,AA,BB,CC;     /* for GF(2^m) curves */

    /*
    mr_small pm,mask;
    int e,k,Me,m;       for GF(p^m) curves */


    #ifndef MR_STATIC

        int logN;           /* constants for fast fourier fft multiplication */
        int nprimes,degree;
        mr_utype *prime,*cr;
        mr_utype *inverse,**roots;
        small_chinese chin;
        mr_utype const1,const2,const3;
        mr_small msw,lsw;
        mr_utype **s1,**s2;   /* pre-computed tables for polynomial reduction */
        mr_utype **t;         /* workspace */
        mr_utype *wa;
        mr_utype *wb;
        mr_utype *wc;

    #endif

        BOOL same;
        BOOL first_one;
        BOOL debug;

        big w0;            /* workspace bigs  */
        big w1,w2,w3,w4;
        big w5,w6,w7;
        big w8,w9,w10,w11;
        big w12,w13,w14,w15;
        big sru;
        big one;

    #ifdef MR_KCM
        big big_ndash;
    big ws,wt;
    #endif

        big A,B;

    /* User modifiables */

    #ifndef MR_SIMPLE_IO
        int  IOBSIZ;       /* size of i/o buffer */
    #endif
        BOOL ERCON;        /* error control   */
        int  ERNUM;        /* last error code */
        int  NTRY;         /* no. of tries for probablistic primality testing   */
    #ifndef MR_SIMPLE_IO
        int  INPLEN;       /* input length               */
    #ifndef MR_SIMPLE_BASE
        int  IOBASE;       /* base for input and output */

    #endif
    #endif
    #ifdef MR_FLASH
        BOOL EXACT;        /* exact flag      */
        BOOL RPOINT;       /* =ON for radix point, =OFF for fractions in output */
    #endif
    #ifndef MR_STRIPPED_DOWN
        BOOL TRACER;       /* turns trace tracker on/off */
    #endif

    #ifdef MR_STATIC
        const int *PRIMES;                      /* small primes array         */
    #ifndef MR_SIMPLE_IO
    char IOBUFF[MR_DEFAULT_BUFFER_SIZE];    /* i/o buffer    */
    #endif
    #else
        int *PRIMES;        /* small primes array         */
    #ifndef MR_SIMPLE_IO
        char *IOBUFF;       /* i/o buffer    */
    #endif
    #endif

    #ifdef MR_FLASH
        int   workprec;
        int   stprec;        /* start precision */

        int RS,RD;
        double D;

        double db,n,p;
        int a,b,c,d,r,q,oldn,ndig;
        mr_small u,v,ku,kv;

        BOOL last,carryon;
        flash pi;

    #endif

    #ifdef MR_FP_ROUNDING
        mr_large inverse_base;
    #endif

    #ifndef MR_STATIC
        char *workspace;
    #else
        char workspace[MR_BIG_RESERVE(MR_SPACES)];
    #endif

        int TWIST; /* set to twisted curve */
        int qnr;    /* a QNR -1 for p=3 mod 4, -2 for p=5 mod 8, 0 otherwise */
        int cnr;    /* a cubic non-residue */
        int pmod8;
        int pmod9;
        BOOL NO_CARRY;
    } miracl;

    /* ------------------------------------------------------------------------*/


    #ifndef MR_GENERIC_MT

    #ifdef MR_WINDOWS_MT
    #define MR_OS_THREADS
    #endif

    #ifdef MR_UNIX_MT
    #define MR_OS_THREADS
    #endif

    #ifdef MR_OPENMP_MT
    #define MR_OS_THREADS
    #endif


    #ifndef MR_OS_THREADS

    extern __device__ miracl *mr_mip;  /* pointer to MIRACL's only global variable */

    #endif

    #endif

    #ifdef MR_GENERIC_MT

    #ifdef MR_STATIC
    #define MR_GENERIC_AND_STATIC
    #endif

    #define _MIPT_  miracl *,
    #define _MIPTO_ miracl *
    #define _MIPD_  miracl *mr_mip,
    #define _MIPDO_ miracl *mr_mip
    #define _MIPP_  mr_mip,
    #define _MIPPO_ mr_mip

    #else

    #define _MIPT_    
    #define _MIPTO_  void  
    #define _MIPD_    
    #define _MIPDO_  void  
    #define _MIPP_    
    #define _MIPPO_    

    #endif

    /* Preamble and exit code for MIRACL routines. *
    * Not used if MR_STRIPPED_DOWN is defined     */

    #ifdef MR_STRIPPED_DOWN
    #define MR_OUT
    #define MR_IN(N)
    #else
    #define MR_OUT  mr_mip->depth--;
    #define MR_IN(N) mr_mip->depth++; if (mr_mip->depth<MR_MAXDEPTH) {mr_mip->trace[mr_mip->depth]=(N); if (mr_mip->TRACER) mr_track_cuda(_MIPPO_); }
    #endif

    /* Function definitions  */

    /* Group 0 - Internal routines */

    extern __device__ void  mr_berror_cuda(_MIPT_ int);
    extern __device__ mr_small mr_shiftbits_cuda(mr_small,int);
    extern __device__ mr_small mr_setbase_cuda(_MIPT_ mr_small);
    extern __device__ void  mr_track_cuda(_MIPTO_ );
    extern __device__ void  mr_lzero_cuda(big);
    extern __device__ BOOL  mr_notint_cuda(flash);
    extern __device__ int   mr_lent_cuda(flash);
    extern __device__ void  mr_padd_cuda(_MIPT_ big,big,big);
    extern __device__ void  mr_psub_cuda(_MIPT_ big,big,big);
    extern __device__ void  mr_pmul_cuda(_MIPT_ big,mr_small,big);
    #ifdef MR_FP_ROUNDING
    extern __device__ mr_large mr_invert_cuda(mr_small);
    extern __device__ mr_small imuldiv_cuda(mr_small,mr_small,mr_small,mr_small,mr_large,mr_small *);
    extern __device__ mr_small mr_sdiv_cuda(_MIPT_ big,mr_small,mr_large,big);
    #else
    extern __device__ mr_small mr_sdiv_cuda(_MIPT_ big,mr_small,big);
    extern __device__ void mr_and_cuda(big,big,big);
    extern __device__ void mr_xor_cuda(big,big,big);
    #endif
    extern __device__ void  mr_shift_cuda(_MIPT_ big,int,big);
    extern __device__ miracl *mr_first_alloc_cuda(void);
    extern __device__ void  *mr_alloc_cuda(_MIPT_ int,int);
    extern __device__ void  mr_free_cuda(void *);
    extern __device__ void  set_user_function_cuda(_MIPT_ BOOL (*)(void));
    extern __device__ void  set_io_buffer_size_cuda(_MIPT_ int);
    extern __device__ int   mr_testbit_cuda(_MIPT_ big,int);
    extern __device__ void  mr_addbit_cuda(_MIPT_ big,int);
    extern __device__ int   recode_cuda(_MIPT_ big ,int ,int ,int );
    extern __device__ int   mr_window_cuda(_MIPT_ big,int,int *,int *,int);
    extern __device__ int   mr_window2_cuda(_MIPT_ big,big,int,int *,int *);
    extern __device__ int   mr_naf_window_cuda(_MIPT_ big,big,int,int *,int *,int);

    extern __device__ int   mr_fft_init_cuda(_MIPT_ int,big,big,BOOL);
    extern __device__ void  mr_dif_fft_cuda(_MIPT_ int,int,mr_utype *);
    extern __device__ void  mr_dit_fft_cuda(_MIPT_ int,int,mr_utype *);
    extern __device__ void  fft_reset_cuda(_MIPTO_);

    extern __device__ int   mr_poly_mul_cuda(_MIPT_ int,big*,int,big*,big*);
    extern __device__ int   mr_poly_sqr_cuda(_MIPT_ int,big*,big*);
    extern __device__ void  mr_polymod_set_cuda(_MIPT_ int,big*,big*);
    extern __device__ int   mr_poly_rem_cuda(_MIPT_ int,big *,big *);

    extern __device__ int   mr_ps_big_mul_cuda(_MIPT_ int,big *,big *,big *);
    extern __device__ int   mr_ps_zzn_mul_cuda(_MIPT_ int,big *,big *,big *);

    extern __device__ mr_small muldiv_cuda(mr_small,mr_small,mr_small,mr_small,mr_small *);
    extern __device__ mr_small muldvm_cuda(mr_small,mr_small,mr_small,mr_small *);
    extern __device__ mr_small muldvd_cuda(mr_small,mr_small,mr_small,mr_small *);
    extern __device__ void  muldvd2_cuda(mr_small,mr_small,mr_small *,mr_small *);

    extern __device__ flash mirvar_mem_variable_cuda(char *,int,int);
    extern __device__ epoint* epoint_init_mem_variable_cuda(_MIPT_ char *,int,int);

    /* Group 1 - General purpose, I/O and basic arithmetic routines  */

    extern __device__ unsigned int   igcd_cuda(unsigned int,unsigned int);
    extern __device__ unsigned long  lgcd_cuda(unsigned long,unsigned long);
    extern __device__ mr_small sgcd_cuda(mr_small,mr_small);
    extern __device__ unsigned int   isqrt_cuda(unsigned int,unsigned int);
    extern __device__ unsigned long  mr_lsqrt_cuda(unsigned long,unsigned long);
    extern __device__ void  irand_cuda(_MIPT_ mr_unsign32);
    extern __device__ mr_small brand_cuda(_MIPTO_ );
    extern __device__ void  zero_cuda(flash);
    extern __device__ void  convert_cuda(_MIPT_ int,big);
    extern __device__ void  uconvert_cuda(_MIPT_ unsigned int,big);
    extern __device__ void  lgconv_cuda(_MIPT_ long,big);
    extern __device__ void  ulgconv_cuda(_MIPT_ unsigned long,big);
    extern __device__ void  tconvert_cuda(_MIPT_ mr_utype,big);

    #ifdef mr_dltype
    extern __device__ void  dlconv_cuda(_MIPT_ mr_dltype,big);
    #endif

    extern __device__ flash mirvar_cuda(_MIPT_ int);
    extern __device__ flash mirvar_mem_cuda(_MIPT_ char *,int);
    extern __device__ void  mirkill_cuda(big);
    extern __device__ void  *memalloc_cuda(_MIPT_ int);
    extern __device__ void  memkill_cuda(_MIPT_ char *,int);
    extern __device__ void  mr_init_threading_cuda(void);
    extern __device__ void  mr_end_threading_cuda(void);
    extern __device__ miracl *get_mip_cuda(void );
    extern __device__ void  set_mip_cuda(miracl *);
    #ifdef MR_GENERIC_AND_STATIC
    extern __device__ miracl *mirsys_cuda(miracl *,int,mr_small);
    #else
    extern __device__ miracl *mirsys_cuda(int,mr_small);
    #endif
    extern __device__ miracl *mirsys_basic_cuda(miracl *,int,mr_small);
    extern __device__ void  mirexit_cuda(_MIPTO_ );
    extern __device__ int   exsign_cuda(flash);
    extern __device__ void  insign_cuda(int,flash);
    extern __device__ int   getdig_cuda(_MIPT_ big,int);
    extern __device__ int   numdig_cuda(_MIPT_ big);
    extern __device__ void  putdig_cuda(_MIPT_ int,big,int);
    extern __device__ void  copy_cuda(flash,flash);
    extern __device__ void  negify_cuda(flash,flash);
    extern __device__ void  absol_cuda(flash,flash);
    extern __device__ int   size_cuda(big);
    extern __device__ int   mr_compare_cuda(big,big);
    extern __device__ void  add_cuda(_MIPT_ big,big,big);
    extern __device__ void  subtract_cuda(_MIPT_ big,big,big);
    extern __device__ void  incr_cuda(_MIPT_ big,int,big);
    extern __device__ void  decr_cuda(_MIPT_ big,int,big);
    extern __device__ void  premult_cuda(_MIPT_ big,int,big);
    extern __device__ int   subdiv_cuda(_MIPT_ big,int,big);
    extern __device__ BOOL  subdivisible_cuda(_MIPT_ big,int);
    extern __device__ int   remain_cuda(_MIPT_ big,int);
    extern __device__ void  bytes_to_big_cuda(_MIPT_ int,const char *,big);
    extern __device__ int   big_to_bytes_cuda(_MIPT_ int,big,char *,BOOL);
    extern __device__ mr_small normalise_cuda(_MIPT_ big,big);
    extern __device__ void  multiply_cuda(_MIPT_ big,big,big);
    extern __device__ void  fft_mult_cuda(_MIPT_ big,big,big);
    extern __device__ BOOL  fastmultop_cuda(_MIPT_ int,big,big,big);
    extern __device__ void  divide_cuda(_MIPT_ big,big,big);
    extern __device__ BOOL  divisible_cuda(_MIPT_ big,big);
    extern __device__ void  mad_cuda(_MIPT_ big,big,big,big,big,big);
    extern __device__ int   instr_cuda(_MIPT_ flash,char *);
    extern __device__ int   otstr_cuda(_MIPT_ flash,char *);
    extern __device__ int   cinstr_cuda(_MIPT_ flash,char *);
    extern __device__ int   cotstr_cuda(_MIPT_ flash,char *);
    extern __device__ epoint* epoint_init_cuda(_MIPTO_ );
    extern __device__ epoint* epoint_init_mem_cuda(_MIPT_ char *,int);
    extern __device__ void* ecp_memalloc_cuda(_MIPT_ int);
    __device__ void ecp_memkill_cuda(_MIPT_ char *,int);
    __device__ BOOL init_big_from_rom_cuda(big,int,const mr_small *,int ,int *);
    __device__ BOOL init_point_from_rom_cuda(epoint *,int,const mr_small *,int,int *);

    #ifndef MR_NO_FILE_IO

    extern __device__ int   innum_cuda(_MIPT_ flash,FILE *);
    extern __device__ int   otnum_cuda(_MIPT_ flash,FILE *);
    extern __device__ int   cinnum_cuda(_MIPT_ flash,FILE *);
    extern __device__ int   cotnum_cuda(_MIPT_ flash,FILE *);

    #endif

    /* Group 2 - Advanced arithmetic routines */

    extern __device__ mr_small smul_cuda(mr_small,mr_small,mr_small);
    extern __device__ mr_small spmd_cuda(mr_small,mr_small,mr_small);
    extern __device__ mr_small invers_cuda(mr_small,mr_small);
    extern __device__ mr_small sqrmp_cuda(mr_small,mr_small);
    extern __device__ int      jac_cuda(mr_small,mr_small);

    extern __device__ void  gprime_cuda(_MIPT_ int);
    extern __device__ int   jack_cuda(_MIPT_ big,big);
    extern __device__ int   egcd_cuda(_MIPT_ big,big,big);
    extern __device__ int   xgcd_cuda(_MIPT_ big,big,big,big,big);
    extern __device__ int   invmodp_cuda(_MIPT_ big,big,big);
    extern __device__ int   logb2_cuda(_MIPT_ big);
    extern __device__ int   hamming_cuda(_MIPT_ big);
    extern __device__ void  expb2_cuda(_MIPT_ int,big);
    extern __device__ void  bigbits_cuda(_MIPT_ int,big);
    extern __device__ void  expint_cuda(_MIPT_ int,int,big);
    extern __device__ void  sftbit_cuda(_MIPT_ big,int,big);
    extern __device__ void  power_cuda(_MIPT_ big,long,big,big);
    extern __device__ void  powmod_cuda(_MIPT_ big,big,big,big);
    extern __device__ void  powmod2_cuda(_MIPT_ big,big,big,big,big,big);
    extern __device__ void  powmodn_cuda(_MIPT_ int,big *,big *,big,big);
    extern __device__ int   powltr_cuda(_MIPT_ int,big,big,big);
    extern __device__ BOOL  double_inverse_cuda(_MIPT_ big,big,big,big,big);
    extern __device__ BOOL  multi_inverse_cuda(_MIPT_ int,big*,big,big*);
    extern __device__ void  lucas_cuda(_MIPT_ big,big,big,big,big);
    extern __device__ BOOL  nroot_cuda(_MIPT_ big,int,big);
    extern __device__ BOOL  sqroot_cuda(_MIPT_ big,big,big);
    extern __device__ void  bigrand_cuda(_MIPT_ big,big);
    extern __device__ void  bigdig_cuda(_MIPT_ int,int,big);
    extern __device__ int   trial_division_cuda(_MIPT_ big,big);
    extern __device__ BOOL  isprime_cuda(_MIPT_ big);
    extern __device__ BOOL  nxprime_cuda(_MIPT_ big,big);
    extern __device__ BOOL  nxsafeprime_cuda(_MIPT_ int,int,big,big);
    extern __device__ BOOL  crt_init_cuda(_MIPT_ big_chinese *,int,big *);
    extern __device__ void  crt_cuda(_MIPT_ big_chinese *,big *,big);
    extern __device__ void  crt_end_cuda(big_chinese *);
    extern __device__ BOOL  scrt_init_cuda(_MIPT_ small_chinese *,int,mr_utype *);
    extern __device__ void  scrt_cuda(_MIPT_ small_chinese*,mr_utype *,big);
    extern __device__ void  scrt_end_cuda(small_chinese *);
    #ifndef MR_STATIC
    extern __device__ BOOL  brick_init_cuda(_MIPT_ brick *,big,big,int,int);
    extern __device__ void  brick_end_cuda(brick *);
    #else
    extern __device__ void  brick_init_cuda(brick *,const mr_small *,big,int,int);
    #endif
    extern __device__ void  pow_brick_cuda(_MIPT_ brick *,big,big);
    #ifndef MR_STATIC
    extern __device__ BOOL  ebrick_init_cuda(_MIPT_ ebrick *,big,big,big,big,big,int,int);
    extern __device__ void  ebrick_end_cuda(ebrick *);
    #else
    extern __device__ void  ebrick_init_cuda(ebrick *,const mr_small *,big,big,big,int,int);
    #endif
    extern __device__ int   mul_brick_cuda(_MIPT_ ebrick*,big,big,big);
    #ifndef MR_STATIC
    extern __device__ BOOL  ebrick2_init_cuda(_MIPT_ ebrick2 *,big,big,big,big,int,int,int,int,int,int);
    extern __device__ void  ebrick2_end_cuda(ebrick2 *);
    #else
    extern __device__ void  ebrick2_init_cuda(ebrick2 *,const mr_small *,big,big,int,int,int,int,int,int);
    #endif
    extern __device__ int   mul2_brick_cuda(_MIPT_ ebrick2*,big,big,big);

    /* Montgomery stuff */

    extern __device__ mr_small prepare_monty_cuda(_MIPT_ big);
    extern __device__ void  kill_monty_cuda(_MIPTO_ );
    extern __device__ void  nres_cuda(_MIPT_ big,big);
    extern __device__ void  redc_cuda(_MIPT_ big,big);

    extern __device__ void  nres_negate_cuda(_MIPT_ big,big);
    extern __device__ void  nres_modadd_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_modsub_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_lazy_cuda(_MIPT_ big,big,big,big,big,big);
    extern __device__ void  nres_complex_cuda(_MIPT_ big,big,big,big);
    extern __device__ void  nres_double_modadd_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_double_modsub_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_premult_cuda(_MIPT_ big,int,big);
    extern __device__ void  nres_modmult_cuda(_MIPT_ big,big,big);
    extern __device__ int   nres_moddiv_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_dotprod_cuda(_MIPT_ int,big *,big *,big);
    extern __device__ void  nres_powmod_cuda(_MIPT_ big,big,big);
    extern __device__ void  nres_powltr_cuda(_MIPT_ int,big,big);
    extern __device__ void  nres_powmod2_cuda(_MIPT_ big,big,big,big,big);
    extern __device__ void  nres_powmodn_cuda(_MIPT_ int,big *,big *,big);
    extern __device__ BOOL  nres_sqroot_cuda(_MIPT_ big,big);
    extern __device__ void  nres_lucas_cuda(_MIPT_ big,big,big,big);
    extern __device__ BOOL  nres_double_inverse_cuda(_MIPT_ big,big,big,big);
    extern __device__ BOOL  nres_multi_inverse_cuda(_MIPT_ int,big *,big *);
    extern __device__ void  nres_div2_cuda(_MIPT_ big,big);
    extern __device__ void  nres_div3_cuda(_MIPT_ big,big);
    extern __device__ void  nres_div5_cuda(_MIPT_ big,big);

    extern __device__ void  shs_init_cuda(sha *);
    extern __device__ void  shs_process_cuda(sha *,int);
    extern __device__ void  shs_hash_cuda(sha *,char *);

    extern __device__ void  shs256_init_cuda(sha256 *);
    extern __device__ void  shs256_process_cuda(sha256 *,int);
    extern __device__ void  shs256_hash_cuda(sha256 *,char *);

    #ifdef mr_unsign64

    extern __device__ void  shs512_init_cuda(sha512 *);
    extern __device__ void  shs512_process_cuda(sha512 *,int);
    extern __device__ void  shs512_hash_cuda(sha512 *,char *);

    extern __device__ void  shs384_init_cuda(sha384 *);
    extern __device__ void  shs384_process_cuda(sha384 *,int);
    extern __device__ void  shs384_hash_cuda(sha384 *,char *);

    extern __device__ void  sha3_init_cuda(sha3 *,int);
    extern __device__ void  sha3_process_cuda(sha3 *,int);
    extern __device__ void  sha3_hash_cuda(sha3 *,char *);

    #endif

    extern __device__ BOOL  aes_init_cuda(aes *,int,int,char *,char *);
    extern __device__ void  aes_getreg_cuda(aes *,char *);
    extern __device__ void  aes_ecb_encrypt_cuda(aes *,MR_BYTE *);
    extern __device__ void  aes_ecb_decrypt_cuda(aes *,MR_BYTE *);
    extern __device__ mr_unsign32 aes_encrypt_cuda(aes *,char *);
    extern __device__ mr_unsign32 aes_decrypt_cuda(aes *,char *);
    extern __device__ void  aes_reset_cuda(aes *,int,char *);
    extern __device__ void  aes_end_cuda(aes *);

    extern __device__ void  gcm_init_cuda(gcm *,int,char *,int,char *);
    extern __device__ BOOL  gcm_add_header_cuda(gcm *,char *,int);
    extern __device__ BOOL  gcm_add_cipher_cuda(gcm *,int,char *,int,char *);
    extern __device__ void  gcm_finish_cuda(gcm *,char *);

    extern __device__ void FPE_encrypt_cuda(int ,aes *,mr_unsign32 ,mr_unsign32 ,char *,int);
    extern __device__ void FPE_decrypt_cuda(int ,aes *,mr_unsign32 ,mr_unsign32 ,char *,int);

    extern __device__ void  strong_init_cuda(csprng *,int,char *,mr_unsign32);
    extern __device__ int   strong_rng_cuda(csprng *);
    extern __device__ void  strong_bigrand_cuda(_MIPT_ csprng *,big,big);
    extern __device__ void  strong_bigdig_cuda(_MIPT_ csprng *,int,int,big);
    extern __device__ void  strong_kill_cuda(csprng *);

    /* special modular multipliers */

    extern __device__ void  comba_mult_cuda(big,big,big);
    extern __device__ void  comba_square_cuda(big,big);
    extern __device__ void  comba_redc_cuda(_MIPT_ big,big);
    extern __device__ void  comba_modadd_cuda(_MIPT_ big,big,big);
    extern __device__ void  comba_modsub_cuda(_MIPT_ big,big,big);
    extern __device__ void  comba_double_modadd_cuda(_MIPT_ big,big,big);
    extern __device__ void  comba_double_modsub_cuda(_MIPT_ big,big,big);
    extern __device__ void  comba_negate_cuda(_MIPT_ big,big);
    extern __device__ void  comba_add_cuda(big,big,big);
    extern __device__ void  comba_sub_cuda(big,big,big);
    extern __device__ void  comba_double_add_cuda(big,big,big);
    extern __device__ void  comba_double_sub_cuda(big,big,big);

    extern __device__ void  comba_mult2_cuda(_MIPT_ big,big,big);

    extern __device__ void  fastmodmult_cuda(_MIPT_ big,big,big);
    extern __device__ void  fastmodsquare_cuda(_MIPT_ big,big);

    extern __device__ void  kcm_mul_cuda(_MIPT_ big,big,big);
    extern __device__ void  kcm_sqr_cuda(_MIPT_ big,big);
    extern __device__ void  kcm_redc_cuda(_MIPT_ big,big);

    extern __device__ void  kcm_multiply_cuda(_MIPT_ int,big,big,big);
    extern __device__ void  kcm_square_cuda(_MIPT_ int,big,big);
    extern __device__ BOOL  kcm_top_cuda(_MIPT_ int,big,big,big);

    /* elliptic curve stuff */

    extern __device__ BOOL point_at_infinity_cuda(epoint *);

    extern __device__ void mr_jsf_cuda(_MIPT_ big,big,big,big,big,big);

    extern __device__ void ecurve_init_cuda(_MIPT_ big,big,big,int);
    extern __device__ int  ecurve_add_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ int  ecurve_sub_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ void ecurve_double_add_cuda(_MIPT_ epoint *,epoint *,epoint *,epoint *,big *,big *);
    extern __device__ void ecurve_multi_add_cuda(_MIPT_ int,epoint **,epoint **);
    extern __device__ void ecurve_double_cuda(_MIPT_ epoint*);
    extern __device__ int  ecurve_mult_cuda(_MIPT_ big,epoint *,epoint *);
    extern __device__ void ecurve_mult2_cuda(_MIPT_ big,epoint *,big,epoint *,epoint *);
    extern __device__ void ecurve_multn_cuda(_MIPT_ int,big *,epoint**,epoint *);

    extern __device__ BOOL epoint_x_cuda(_MIPT_ big);
    extern __device__ BOOL epoint_set_cuda(_MIPT_ big,big,int,epoint*);
    extern __device__ int  epoint_get_cuda(_MIPT_ epoint*,big,big);
    extern __device__ void epoint_getxyz_cuda(_MIPT_ epoint *,big,big,big);
    extern __device__ BOOL epoint_norm_cuda(_MIPT_ epoint *);
    extern __device__ BOOL epoint_multi_norm_cuda(_MIPT_ int,big *,epoint **);
    extern __device__ void epoint_free_cuda(epoint *);
    extern __device__ void epoint_copy_cuda(epoint *,epoint *);
    extern __device__ BOOL epoint_comp_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ void epoint_negate_cuda(_MIPT_ epoint *);

    extern __device__ BOOL ecurve2_init_cuda(_MIPT_ int,int,int,int,big,big,BOOL,int);
    extern __device__ big  ecurve2_add_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ big  ecurve2_sub_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ void ecurve2_multi_add_cuda(_MIPT_ int,epoint **,epoint **);
    extern __device__ void ecurve2_mult_cuda(_MIPT_ big,epoint *,epoint *);
    extern __device__ void ecurve2_mult2_cuda(_MIPT_ big,epoint *,big,epoint *,epoint *);
    extern __device__ void ecurve2_multn_cuda(_MIPT_ int,big *,epoint**,epoint *);

    extern __device__ epoint* epoint2_init_cuda(_MIPTO_ );
    extern __device__ BOOL epoint2_set_cuda(_MIPT_ big,big,int,epoint*);
    extern __device__ int  epoint2_get_cuda(_MIPT_ epoint*,big,big);
    extern __device__ void epoint2_getxyz_cuda(_MIPT_ epoint *,big,big,big);
    extern __device__ int  epoint2_norm_cuda(_MIPT_ epoint *);
    extern __device__ void epoint2_free_cuda(epoint *);
    extern __device__ void epoint2_copy_cuda(epoint *,epoint *);
    extern __device__ BOOL epoint2_comp_cuda(_MIPT_ epoint *,epoint *);
    extern __device__ void epoint2_negate_cuda(_MIPT_ epoint *);

    /* GF(2) stuff */

    extern __device__ BOOL prepare_basis_cuda(_MIPT_ int,int,int,int,BOOL);
    extern __device__ int parity2_cuda(big);
    extern __device__ BOOL multi_inverse2_cuda(_MIPT_ int,big *,big *);
    extern __device__ void add2_cuda(big,big,big);
    extern __device__ void incr2_cuda(big,int,big);
    extern __device__ void reduce2_cuda(_MIPT_ big,big);
    extern __device__ void multiply2_cuda(_MIPT_ big,big,big);
    extern __device__ void modmult2_cuda(_MIPT_ big,big,big);
    extern __device__ void modsquare2_cuda(_MIPT_ big,big);
    extern __device__ void power2_cuda(_MIPT_ big,int,big);
    extern __device__ void sqroot2_cuda(_MIPT_ big,big);
    extern __device__ void halftrace2_cuda(_MIPT_ big,big);
    extern __device__ BOOL quad2_cuda(_MIPT_ big,big);
    extern __device__ BOOL inverse2_cuda(_MIPT_ big,big);
    extern __device__ void karmul2_cuda(int,mr_small *,mr_small *,mr_small *,mr_small *);
    extern __device__ void karmul2_poly_cuda(_MIPT_ int,big *,big *,big *,big *);
    extern __device__ void karmul2_poly_upper_cuda(_MIPT_ int,big *,big *,big *,big *);
    extern __device__ void gf2m_dotprod_cuda(_MIPT_ int,big *,big *,big);
    extern __device__ int  trace2_cuda(_MIPT_ big);
    extern __device__ void rand2_cuda(_MIPT_ big);
    extern __device__ void gcd2_cuda(_MIPT_ big,big,big);
    extern __device__ int degree2_cuda(big);

    /* zzn2 stuff */
    extern __device__ void zzn2_mirvar_cuda(_MIPD_ zzn2 *w);
    extern __device__ void zzn2_kill_cuda(_MIPD_ zzn2*w);
    extern __device__ BOOL zzn2_iszero_cuda(zzn2 *);
    extern __device__ BOOL zzn2_isunity_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_from_int_cuda(_MIPT_ int,zzn2 *);
    extern __device__ void zzn2_from_ints_cuda(_MIPT_ int,int,zzn2 *);
    extern __device__ void zzn2_copy_cuda(zzn2 *,zzn2 *);
    extern __device__ void zzn2_zero_cuda(zzn2 *);
    extern __device__ void zzn2_negate_cuda(_MIPT_ zzn2 *,zzn2 *);
    extern __device__ void zzn2_conj_cuda(_MIPT_ zzn2 *,zzn2 *);
    extern __device__ void zzn2_add_cuda(_MIPT_ zzn2 *,zzn2 *,zzn2 *);
    extern __device__ void zzn2_sub_cuda(_MIPT_ zzn2 *,zzn2 *,zzn2 *);
    extern __device__ void zzn2_smul_cuda(_MIPT_ zzn2 *,big,zzn2 *);
    extern __device__ void zzn2_mul_cuda(_MIPT_ zzn2 *,zzn2 *,zzn2 *);
    extern __device__ void zzn2_sqr_cuda(_MIPT_ zzn2 *,zzn2 *);
    extern __device__ void zzn2_inv_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_timesi_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_pow_cuda(_MIPD_ zzn2 *x, big e, zzn2*w);
    extern __device__ void zzn2_powl_cuda(_MIPT_ zzn2 *,big,zzn2 *);
    extern __device__ void zzn2_from_zzns_cuda(big,big,zzn2 *);
    extern __device__ void zzn2_from_bigs_cuda(_MIPT_ big,big,zzn2 *);
    extern __device__ void zzn2_from_zzn_cuda(big,zzn2 *);
    extern __device__ void zzn2_from_big_cuda(_MIPT_ big, zzn2 *);
    extern __device__ void zzn2_sadd_cuda(_MIPT_ zzn2 *,big,zzn2 *);
    extern __device__ void zzn2_ssub_cuda(_MIPT_ zzn2 *,big,zzn2 *);
    extern __device__ void zzn2_div2_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_div3_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_div5_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_imul_cuda(_MIPT_ zzn2 *,int,zzn2 *);
    extern __device__ BOOL zzn2_compare_cuda(zzn2 *,zzn2 *);
    extern __device__ void zzn2_txx_cuda(_MIPT_ zzn2 *);
    extern __device__ void zzn2_txd_cuda(_MIPT_ zzn2 *);
    extern __device__ BOOL zzn2_sqrt_cuda(_MIPT_ zzn2 *,zzn2 *);
    extern __device__ BOOL zzn2_qr_cuda(_MIPT_ zzn2 *);
    extern __device__ BOOL zzn2_multi_inverse_cuda(_MIPT_ int,zzn2 *,zzn2 *);


    /* zzn3 stuff */

    extern __device__ void zzn3_set_cuda(_MIPT_ int,big);
    extern __device__ BOOL zzn3_iszero_cuda(zzn3 *);
    extern __device__ BOOL zzn3_isunity_cuda(_MIPT_ zzn3 *);
    extern __device__ void zzn3_from_int_cuda(_MIPT_ int,zzn3 *);
    extern __device__ void zzn3_from_ints_cuda(_MIPT_ int,int,int,zzn3 *);
    extern __device__ void zzn3_copy_cuda(zzn3 *,zzn3 *);
    extern __device__ void zzn3_zero_cuda(zzn3 *);
    extern __device__ void zzn3_negate_cuda(_MIPT_ zzn3 *,zzn3 *);
    extern __device__ void zzn3_powq_cuda(_MIPT_ zzn3 *,zzn3 *);
    extern __device__ void zzn3_add_cuda(_MIPT_ zzn3 *,zzn3 *,zzn3 *);
    extern __device__ void zzn3_sub_cuda(_MIPT_ zzn3 *,zzn3 *,zzn3 *);
    extern __device__ void zzn3_smul_cuda(_MIPT_ zzn3 *,big,zzn3 *);
    extern __device__ void zzn3_mul_cuda(_MIPT_ zzn3 *,zzn3 *,zzn3 *);
    extern __device__ void zzn3_inv_cuda(_MIPT_ zzn3 *);
    extern __device__ void zzn3_timesi_cuda(_MIPT_ zzn3 *);
    extern __device__ void zzn3_timesi2_cuda(_MIPT_ zzn3 *);
    extern __device__ void zzn3_powl_cuda(_MIPT_ zzn3 *,big,zzn3 *);
    extern __device__ void zzn3_from_zzns_cuda(big,big,big,zzn3 *);
    extern __device__ void zzn3_from_bigs_cuda(_MIPT_ big,big,big,zzn3 *);
    extern __device__ void zzn3_from_zzn_cuda(big,zzn3 *);
    extern __device__ void zzn3_from_zzn_1_cuda(big,zzn3 *);
    extern __device__ void zzn3_from_zzn_2_cuda(big,zzn3 *);
    extern __device__ void zzn3_from_big_cuda(_MIPT_ big, zzn3 *);
    extern __device__ void zzn3_sadd_cuda(_MIPT_ zzn3 *,big,zzn3 *);
    extern __device__ void zzn3_ssub_cuda(_MIPT_ zzn3 *,big,zzn3 *);
    extern __device__ void zzn3_div2_cuda(_MIPT_ zzn3 *);
    extern __device__ void zzn3_imul_cuda(_MIPT_ zzn3 *,int,zzn3 *);
    extern __device__ BOOL zzn3_compare_cuda(zzn3 *,zzn3 *);

    /* zzn4 stuff */
    extern __device__ void zzn4_mirvar_cuda(_MIPD_ zzn4 *w);
    extern __device__ void zzn4_kill_cuda(_MIPD_ zzn4 *w);
    extern __device__ BOOL zzn4_iszero_cuda(zzn4 *);
    extern __device__ BOOL zzn4_isunity_cuda(_MIPT_ zzn4 *);
    extern __device__ void zzn4_from_int_cuda(_MIPT_ int,zzn4 *);
    extern __device__ void zzn4_copy_cuda(zzn4 *,zzn4 *);
    extern __device__ void zzn4_zero_cuda(zzn4 *);
    extern __device__ void zzn4_negate_cuda(_MIPT_ zzn4 *,zzn4 *);
    extern __device__ void zzn4_powq_cuda(_MIPT_ zzn2 *,zzn4 *);
    extern __device__ void zzn4_add_cuda(_MIPT_ zzn4 *,zzn4 *,zzn4 *);
    extern __device__ void zzn4_sub_cuda(_MIPT_ zzn4 *,zzn4 *,zzn4 *);
    extern __device__ void zzn4_smul_cuda(_MIPT_ zzn4 *,zzn2 *,zzn4 *);
    extern __device__ void zzn4_sqr_cuda(_MIPT_ zzn4 *,zzn4 *);
    extern __device__ void zzn4_mul_cuda(_MIPT_ zzn4 *,zzn4 *,zzn4 *);
    extern __device__ void zzn4_inv_cuda(_MIPT_ zzn4 *);
    extern __device__ void zzn4_timesi_cuda(_MIPT_ zzn4 *);
    extern __device__ void zzn4_tx_cuda(_MIPT_ zzn4 *);
    extern __device__ void zzn4_from_zzn2s_cuda(zzn2 *,zzn2 *,zzn4 *);
    extern __device__ void zzn4_from_zzn2_cuda(zzn2 *,zzn4 *);
    extern __device__ void zzn4_from_zzn2h_cuda(zzn2 *,zzn4 *);
    extern __device__ void zzn4_from_zzn_cuda(big,zzn4 *);
    extern __device__ void zzn4_from_big_cuda(_MIPT_ big , zzn4 *);
    extern __device__ void zzn4_sadd_cuda(_MIPT_ zzn4 *,zzn2 *,zzn4 *);
    extern __device__ void zzn4_ssub_cuda(_MIPT_ zzn4 *,zzn2 *,zzn4 *);
    extern __device__ void zzn4_div2_cuda(_MIPT_ zzn4 *);
    extern __device__ void zzn4_conj_cuda(_MIPT_ zzn4 *,zzn4 *);
    extern __device__ void zzn4_imul_cuda(_MIPT_ zzn4 *,int,zzn4 *);
    extern __device__ void zzn4_lmul_cuda(_MIPT_ zzn4 *,big,zzn4 *);
    extern __device__ BOOL zzn4_compare_cuda(zzn4 *,zzn4 *);

    /* ecn2 stuff */
    extern __device__ void ecn2_mirvar_cuda(_MIPD_ ecn2* q);
    extern __device__ void ecn2_kill_cuda(_MIPD_ ecn2* q);
    extern __device__ BOOL ecn2_iszero_cuda(ecn2 *);
    extern __device__ void ecn2_copy_cuda(ecn2 *,ecn2 *);
    extern __device__ void ecn2_zero_cuda(ecn2 *);
    extern __device__ BOOL ecn2_compare_cuda(_MIPT_ ecn2 *,ecn2 *);
    extern __device__ void ecn2_norm_cuda(_MIPT_ ecn2 *);
    extern __device__ void ecn2_get_cuda(_MIPT_ ecn2 *,zzn2 *,zzn2 *,zzn2 *);
    extern __device__ void ecn2_getxy_cuda(ecn2 *,zzn2 *,zzn2 *);
    extern __device__ void ecn2_getx_cuda(ecn2 *,zzn2 *);
    extern __device__ void ecn2_getz_cuda(_MIPT_ ecn2 *,zzn2 *);
    extern __device__ void ecn2_rhs_cuda(_MIPT_ zzn2 *,zzn2 *);
    extern __device__ BOOL ecn2_set_cuda(_MIPT_ zzn2 *,zzn2 *,ecn2 *);
    extern __device__ BOOL ecn2_setx_cuda(_MIPT_ zzn2 *,ecn2 *);
    extern __device__ void ecn2_setxyz_cuda(_MIPT_ zzn2 *,zzn2 *,zzn2 *,ecn2 *);
    extern __device__ void ecn2_negate_cuda(_MIPT_ ecn2 *,ecn2 *);
    extern __device__ BOOL ecn2_add3_cuda(_MIPT_ ecn2 *,ecn2 *,zzn2 *,zzn2 *,zzn2 *);
    extern __device__ BOOL ecn2_add2_cuda(_MIPT_ ecn2 *,ecn2 *,zzn2 *,zzn2 *);
    extern __device__ BOOL ecn2_add1_cuda(_MIPT_ ecn2 *,ecn2 *,zzn2 *);
    extern __device__ BOOL ecn2_add_cuda(_MIPT_ ecn2 *,ecn2 *);
    extern __device__ BOOL ecn2_sub_cuda(_MIPT_ ecn2 *,ecn2 *);
    extern __device__ BOOL ecn2_add_sub_cuda(_MIPT_ ecn2 *,ecn2 *,ecn2 *,ecn2 *);
    extern __device__ int ecn2_mul2_jsf_cuda(_MIPT_ big,ecn2 *,big,ecn2 *,ecn2 *);
    extern __device__ int ecn2_mul_cuda(_MIPT_ big,ecn2 *);
    extern __device__ void ecn2_psi_cuda(_MIPT_ zzn2 *,ecn2 *);
    extern __device__ BOOL ecn2_multi_norm_cuda(_MIPT_ int ,zzn2 *,ecn2 *);
    extern __device__ int ecn2_mul4_gls_v_cuda(_MIPT_ big *,int,ecn2 *,big *,ecn2 *,zzn2 *,ecn2 *);
    extern __device__ int ecn2_muln_engine_cuda(_MIPT_ int,int,int,int,big *,big *,big *,big *,ecn2 *,ecn2 *,ecn2 *);
    extern __device__ void ecn2_precomp_gls_cuda(_MIPT_ int,BOOL,ecn2 *,zzn2 *,ecn2 *);
    extern __device__ int ecn2_mul2_gls_cuda(_MIPT_ big *,ecn2 *,zzn2 *,ecn2 *);
    extern __device__ void ecn2_precomp_cuda(_MIPT_ int,BOOL,ecn2 *,ecn2 *);
    extern __device__ int ecn2_mul2_cuda(_MIPT_ big,int,ecn2 *,big,ecn2 *,ecn2 *);
    #ifndef MR_STATIC
    extern __device__ BOOL ecn2_brick_init_cuda(_MIPT_ ebrick *,zzn2 *,zzn2 *,big,big,big,int,int);
    extern __device__ void ecn2_brick_end_cuda(ebrick *);
    #else
    extern __device__ void ebrick_init_cuda(ebrick *,const mr_small *,big,big,big,int,int);
    #endif
    extern __device__ void ecn2_mul_brick_gls_cuda(_MIPT_ ebrick *B,big *,zzn2 *,zzn2 *,zzn2 *);
    extern __device__ void ecn2_multn_cuda(_MIPT_ int,big *,ecn2 *,ecn2 *);
    extern __device__ void ecn2_mult4_cuda(_MIPT_ big *,ecn2 *,ecn2 *);
    /* Group 3 - Floating-slash routines      */

    #ifdef MR_FLASH
    extern __device__ void  fpack_cuda(_MIPT_ big,big,flash);
    extern __device__ void  numer_cuda(_MIPT_ flash,big);
    extern __device__ void  denom_cuda(_MIPT_ flash,big);
    extern __device__ BOOL  fit_cuda(big,big,int);
    extern __device__ void  build_cuda(_MIPT_ flash,int (*)(_MIPT_ big,int));
    extern __device__ void  mround_cuda(_MIPT_ big,big,flash);
    extern __device__ void  flop_cuda(_MIPT_ flash,flash,int *,flash);
    extern __device__ void  fmul_cuda(_MIPT_ flash,flash,flash);
    extern __device__ void  fdiv_cuda(_MIPT_ flash,flash,flash);
    extern __device__ void  fadd_cuda(_MIPT_ flash,flash,flash);
    extern __device__ void  fsub_cuda(_MIPT_ flash,flash,flash);
    extern __device__ int   fcomp_cuda(_MIPT_ flash,flash);
    extern __device__ void  fconv_cuda(_MIPT_ int,int,flash);
    extern __device__ void  frecip_cuda(_MIPT_ flash,flash);
    extern __device__ void  ftrunc_cuda(_MIPT_ flash,big,flash);
    extern __device__ void  fmodulo_cuda(_MIPT_ flash,flash,flash);
    extern __device__ void  fpmul_cuda(_MIPT_ flash,int,int,flash);
    extern __device__ void  fincr_cuda(_MIPT_ flash,int,int,flash);
    extern __device__ void  dconv_cuda(_MIPT_ double,flash);
    extern __device__ double fdsize_cuda(_MIPT_ flash);
    extern __device__ void  frand_cuda(_MIPT_ flash);

    /* Group 4 - Advanced Flash routines */

    extern __device__ void  fpower_cuda(_MIPT_ flash,int,flash);
    extern __device__ BOOL  froot_cuda(_MIPT_ flash,int,flash);
    extern __device__ void  fpi_cuda(_MIPT_ flash);
    extern __device__ void  fexp_cuda(_MIPT_ flash,flash);
    extern __device__ void  flog_cuda(_MIPT_ flash,flash);
    extern __device__ void  fpowf_cuda(_MIPT_ flash,flash,flash);
    extern __device__ void  ftan_cuda(_MIPT_ flash,flash);
    extern __device__ void  fatan_cuda(_MIPT_ flash,flash);
    extern __device__ void  fsin_cuda(_MIPT_ flash,flash);
    extern __device__ void  fasin_cuda(_MIPT_ flash,flash);
    extern __device__ void  fcos_cuda(_MIPT_ flash,flash);
    extern __device__ void  facos_cuda(_MIPT_ flash,flash);
    extern __device__ void  ftanh_cuda(_MIPT_ flash,flash);
    extern __device__ void  fatanh_cuda(_MIPT_ flash,flash);
    extern __device__ void  fsinh_cuda(_MIPT_ flash,flash);
    extern __device__ void  fasinh_cuda(_MIPT_ flash,flash);
    extern __device__ void  fcosh_cuda(_MIPT_ flash,flash);
    extern __device__ void  facosh_cuda(_MIPT_ flash,flash);
    #endif


    /* Test predefined Macros to determine compiler type, and hopefully 
    selectively use fast in-line assembler (or other compiler specific
    optimisations. Note I am unsure of Microsoft version numbers. So I 
    suspect are Microsoft.

    Note: It seems to be impossible to get the 16-bit Microsoft compiler
    to allow inline 32-bit op-codes. So I suspect that INLINE_ASM == 2 will
    never work with it. Pity. 

    #define INLINE_ASM 1    -> generates 8086 inline assembly
    #define INLINE_ASM 2    -> generates mixed 8086 & 80386 inline assembly,
                            so you can get some benefit while running in a 
                            16-bit environment on 32-bit hardware (DOS, Windows
                            3.1...)
    #define INLINE_ASM 3    -> generate true 80386 inline assembly - (Using DOS 
                            extender, Windows '95/Windows NT)
                            Actually optimised for Pentium

    #define INLINE_ASM 4    -> 80386 code in the GNU style (for (DJGPP)

    Small, medium, compact and large memory models are supported for the
    first two of the above.
                            
    */

    /* To allow for inline assembly */

    #ifdef __GNUC__
    #define ASM __asm__ __volatile__
    #endif

    #ifdef __TURBOC__
    #define ASM asm
    #endif

    #ifdef _MSC_VER
    #define ASM _asm
    #endif

    #ifndef MR_NOASM

    /* Win64 - inline the time critical function */
    #ifndef MR_NO_INTRINSICS
    #ifdef MR_WIN64
    #define muldvd(a,b,c,rp) (*(rp)=_umul128((a),(b),&(tm)),*(rp)+=(c),tm+=(*(rp)<(c)),tm)
            #define muldvd2(a,b,c,rp) (tr=_umul128((a),(b),&(tm)),tr+=(*(c)),tm+=(tr<(*(c))),tr+=(*(rp)),tm+=(tr<(*(rp))),*(rp)=tr,*(c)=tm)
    #endif

    /* Itanium - inline the time-critical functions */

    #ifdef MR_ITANIUM
    #define muldvd(a,b,c,rp)  (tm=_m64_xmahu((a),(b),(c)),*(rp)=_m64_xmalu((a),(b),(c)),tm)
            #define muldvd2(a,b,c,rp) (tm=_m64_xmalu((a),(b),(*(c))),*(c)=_m64_xmahu((a),(b),(*(c))),tm+=*(rp),*(c)+=(tm<*(rp)),*(rp)=tm)
    #endif
    #endif
    /*

    SSE2 code. Works as for itanium - but in fact it is slower than the regular code so not recommended
    Would require a call to emmintrin.h or xmmintrin.h, and an __m128i variable tm to be declared in effected 
    functions. But it works!

        #define muldvd(a,b,c,rp)  (tm=_mm_add_epi64(_mm_mul_epu32(_mm_cvtsi32_si128((a)),_mm_cvtsi32_si128((b))),_mm_cvtsi32_si128((c))),*(rp)=_mm_cvtsi128_si32(tm),_mm_cvtsi128_si32(_mm_shuffle_epi32(tm,_MM_SHUFFLE(3,2,0,1))) )
        #define muldvd2(a,b,c,rp) (tm=_mm_add_epi64(_mm_add_epi64(_mm_mul_epu32(_mm_cvtsi32_si128((a)),_mm_cvtsi32_si128((b))),_mm_cvtsi32_si128(*(c))),_mm_cvtsi32_si128(*(rp))),*(rp)=_mm_cvtsi128_si32(tm),*(c)=_mm_cvtsi128_si32( _mm_shuffle_epi32(tm,_MM_SHUFFLE(3,2,0,1))  )
    */

    /* Borland C/Turbo C */

    #ifdef __TURBOC__
    #ifndef __HUGE__
            #if defined(__COMPACT__) || defined(__LARGE__)
                #define MR_LMM
            #endif

            #if MIRACL==16
                #define INLINE_ASM 1
            #endif

            #if __TURBOC__>=0x410
                #if MIRACL==32
    #if defined(__SMALL__) || defined(__MEDIUM__) || defined(__LARGE__) || defined(__COMPACT__)
                        #define INLINE_ASM 2
                    #else
                        #define INLINE_ASM 3
                    #endif
                #endif
            #endif
        #endif
    #endif

    /* Microsoft C */

    #ifdef _MSC_VER
    #ifndef M_I86HM
            #if defined(M_I86CM) || defined(M_I86LM)
                #define MR_LMM
            #endif
            #if _MSC_VER>=600
                #if _MSC_VER<1200
                    #if MIRACL==16
                        #define INLINE_ASM 1
                    #endif
                #endif
            #endif
            #if _MSC_VER>=1000
                #if _MSC_VER<1500
                    #if MIRACL==32
                        #define INLINE_ASM 3
                    #endif
                #endif
            #endif     
        #endif
    #endif

    /* DJGPP GNU C */

    #ifdef __GNUC__
    #ifdef i386
    #if MIRACL==32
                #define INLINE_ASM 4
            #endif
    #endif
    #endif

    #endif



    /* 
    The following contribution is from Tielo Jongmans, Netherlands
    These inline assembler routines are suitable for Watcom 10.0 and up 

    Added into miracl.h.  Notice the override of the original declarations 
    of these routines, which should be removed.

    The following pragma is optional, it is dangerous, but it saves a 
    calling sequence
    */

    /*

    #pragma off (check_stack);

    extern unsigned int muldiv(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *);
    #pragma aux muldiv=                 \
        "mul     edx"                \
        "add     eax,ebx"            \
        "adc     edx,0"              \
        "div     ecx"                \
        "mov     [esi],edx"          \
        parm [eax] [edx] [ebx] [ecx] [esi]   \
        value [eax]                     \
        modify [eax edx];

    extern unsigned int muldvm(unsigned int, unsigned int, unsigned int, unsigned int *);
    #pragma aux muldvm=                 \
            "div     ebx"               \
            "mov     [ecx],edx"         \
        parm [edx] [eax] [ebx] [ecx]    \
        value [eax]                     \
        modify [eax edx];

    extern unsigned int muldvd(unsigned int, unsigned int, unsigned int, unsigned int *);
    #pragma aux muldvd=                 \
            "mul     edx"               \
            "add     eax,ebx"           \
            "adc     edx,0"             \
            "mov     [ecx],eax"         \
            "mov     eax,edx"           \
        parm [eax] [edx] [ebx] [ecx]    \
        value [eax]                     \
        modify [eax edx];

    */
    #endif
    /*miracl.cuh-end*/

    /*smzzn12.cuh-begin*/
    #ifndef smzzn12_cuh
    #define smzzn12_cuh
    typedef struct
    {
        zzn4 a;
        zzn4 b;
        zzn4 c;
        BOOL unitary;
        BOOL miller;
    } zzn12;


    extern __device__ void zzn12_mark_miller_cuda(_MIPD_ zzn12* );
    extern __device__ void zzn12_mark_regular_cuda(_MIPD_ zzn12* );
    extern __device__ void zzn12_mark_unitary_cuda(_MIPD_ zzn12* );
    extern __device__ void zzn12_mirvar_cuda(_MIPD_ zzn12*);
    extern __device__ void zzn12_kill_cuda(_MIPD_ zzn12*);
    extern __device__ BOOL zzn12_iszero_cuda(_MIPD_ zzn12 *);
    extern __device__ BOOL zzn12_isunity_cuda(_MIPT_ zzn12 *);
    extern __device__ void zzn12_from_int_cuda(_MIPT_ int,zzn12 *);
    extern __device__ void zzn12_copy_cuda(zzn12 *,zzn12 *);
    extern __device__ void zzn12_zero_cuda(zzn12 *);
    extern __device__ void zzn12_negate_cuda(_MIPT_ zzn12 *,zzn12 *);
    extern __device__ void zzn12_powq_cuda(_MIPT_ zzn12 *,zzn2 *);
    extern __device__ void zzn12_pow_cuda(_MIPT_ zzn12 *,big );
    extern __device__ void zzn12_add_cuda(_MIPT_ zzn12 *,zzn12 *,zzn12 *);
    extern __device__ void zzn12_sub_cuda(_MIPT_ zzn12 *,zzn12 *,zzn12 *);
    extern __device__ void zzn12_smul_cuda(_MIPT_ zzn12 *,zzn4 *,zzn12 *);
    extern __device__ void zzn12_sqr_cuda(_MIPT_ zzn12 *,zzn12 *);
    extern __device__ void zzn12_mul_cuda(_MIPT_ zzn12 *,zzn12 *,zzn12 *);
    extern __device__ void zzn12_inv_cuda(_MIPT_ zzn12 *);
    extern __device__ void zzn12_timesi_cuda(_MIPT_ zzn12 *);
    extern __device__ void zzn12_tx_cuda(_MIPT_ zzn12 *);
    extern __device__ void zzn12_from_zzn4s_cuda(zzn4 *,zzn4 *,zzn4 *,zzn12 *);
    extern __device__ void zzn12_from_zzn4_cuda(zzn4 *,zzn12 *);
    extern __device__ void zzn12_div2_cuda(_MIPT_ zzn12 *);
    extern __device__ void zzn12_conj_cuda(_MIPT_ zzn12 *,zzn12 *);
    extern __device__ void zzn12_imul_cuda(_MIPT_ zzn12 *,int,zzn12 *);
    extern __device__ void zzn12_lmul_cuda(_MIPT_ zzn12 *,big,zzn12 *);
    extern __device__ BOOL zzn12_compare_cuda(zzn12 *,zzn12 *);

    __device__ void zzn12_tochar_cuda(_MIPD_ zzn12* ,unsigned char *,unsigned int bInt);
    __device__ void zzn12_fromchar_cuda(_MIPD_ zzn12* ,unsigned char *,unsigned int bInt);
    #endif
    /*smzzn12.cuh-end*/

    /*sm9_utils.cuh-begin*/
    #ifndef sm9_utils_cuh
    #define sm9_utils_cuh
    typedef epoint ecn; // edit by andrew song
    extern __device__ BOOL sm9init;    //sm9算法库状态
    extern __device__ BOOL sm9sign;    //sm9签名模块状态
    extern __device__ BOOL sm9encrypt; //sm9加解密模块状态
    extern __device__ BOOL sm9keyexchange; //sm9密钥协商模块状态

    extern __device__ int   sm9len = 32;  //sm9算法的安全参数，标准中是32（bytes）

    extern __device__ big   sm9q;    //sm9椭圆曲线基域素数
    extern __device__ big   sm9a;    //sm9椭圆曲线函数参数a
    extern __device__ big   sm9b;    //sm9椭圆曲线函数参数b
    extern __device__ big   sm9n;    //sm9基域上椭圆曲线的阶

    extern __device__ big   sm9t;    //sm9椭圆曲线复乘参数t
    extern __device__ zzn2  sm9X;    //sm9椭圆曲线pairing预计算参数

    extern __device__ ecn   p1G1;    //sm9椭圆曲线G1群生成元
    extern __device__ ecn   ppG1;    //sm9算法加密主公钥
    extern __device__ ecn   keG1;    //sm9算法密钥协商主公钥

    extern __device__ ecn2  p2G2;    //sm9椭圆曲线G2群生成元
    extern __device__ ecn2  ppG2;    //sm9算法签名主公钥

    extern __device__ zzn12 gGt;     //sm9算法签名模块预计算结果
    extern __device__ zzn12 eGt;     //sm9算法加解密模块预计算结果
    extern __device__ zzn12 kGt;     //sm9算法密钥协商模块预计算结果

    extern __device__ BOOL  TWIST;   //sm9算法TWIST类型，标准为MR_SEXTIC_M
    
    extern __device__ unsigned char hid[3] = {0x01,0x02,0x03}; // sm9算法标识，hid[0]表示签名算法，hid[1]表示密钥协商算法，hid[2]表示加解密算法

    __device__ miracl* GenMiracl_cuda(int secLevel);  //初始化一个mircal库
    __device__ void CloseMiracl_cuda(_MIPDO_);        //关闭一个mircal库

    //sm9算法中辅助算法H1、H2
    //@zbuf 输入字符串
    //@zbuflen 字符串长度
    //@secLevel 安全参数
    //@h 函数选择参数，当h=1时为函数H1，当h=2时为函数H2
    //输出一个大整数
    __device__ big Hfun_cuda(_MIPD_ char *zbuf, int zbufLen, int secLevel,int h);

    //sm9算法中kdf算法
    //@zbuf 输入字符串
    //@zbulen 字符串长度
    //@klen 预期输出长度
    //@kbuf 输出字符串
    //输出 0
    __device__ int kdf_cuda(char *zbuf, int zbufLen, int klen, char *kbuf);

    //基于SM3的MAC算法
    //@key 输入密钥
    //@keylen 密钥长度
    //@msg   消息
    //@msglen 消息长度
    //@mac 输出的mac值
    //输出 0
    __device__ int MAC_cuda(unsigned char* key, unsigned int keylen, unsigned char* msg, unsigned int msglen, unsigned char *mac);

    //XOR加密函数
    //@bufIn 输入字符串
    //@ilen  输入字符串长度
    //@bufKey XOR加密密钥
    //@bufOut 加密结果
    //输出 0
    __device__ int xorAlgor_cuda( unsigned char *bufIn, int ilen, unsigned char *bufKey, unsigned char * bufOut);

    __device__ BOOL ecap_cuda(_MIPD_ ecn2 *P,ecn *Q,big x,zzn2 *X,zzn12 *r);
    __device__ void set_frobenius_constant_cuda(_MIPD_ zzn2 *x);
    #endif
    /*sm9_utils.cuh-end*/

    /*print_out.cuh-begin*/
    #ifndef print_out_cuh
    #define print_out_cuh
    __device__ void print_hex_cuda(unsigned char *pbuf, int len); //打印一个字符串pbuf，长度为len Byte
    __device__ void print_big_cuda(_MIPD_ big x, int len);       //打印一个大整数x，长度为len Byte
    __device__ void print_zzn2_cuda(_MIPD_ zzn2 *w,int len);     //打印一个二阶扩域数w，长度为len Byte
    __device__ void print_zzn12_cuda(_MIPD_ zzn12 *w,int len);   //打印一个十二阶扩域w的最后一个分量，长度为len Byte
    __device__ void print_ecn2_cuda(_MIPD_ ecn2 *q, int len);    //打印一个扩域上的椭圆曲线点q，分量长度为len Byte
    __device__ void print_ecn_cuda(_MIPD_ ecn *q, int len);      //打印椭圆曲线的一个点q，分量长度为len Byte
    #endif
    /*print_out.cuh-end*/

    /*sm3.cuh-begin*/
    #ifndef sm3_cuh
    #define sm3_cuh
    #define SM3_HASH_LEN  32

    typedef struct {
        unsigned int state[8];
        unsigned long long tc;
        unsigned int bc;
        unsigned char buffer[64];
    } SM3_CTX;   //SM3算法的基本结构体


    //SM3算法三件套，首先调用SM3Init，然后调用SM3Update（可重复调用）逐步将消息读入，最后调用SM3Final得到SM3摘要值digest

    __device__ void SM3Init_cuda(SM3_CTX* context);
    __device__ void SM3Update_cuda(SM3_CTX* context, unsigned char* data, unsigned int len);
    __device__ void SM3Final_cuda(unsigned char digest[32], SM3_CTX* context);


    //SM3算法一步到位，输入消息和长度，得到摘要digest
    __device__ void SM3_cuda(unsigned char* data, unsigned int len, unsigned char digest[32]);
    #endif
    /*sm3.cuh-end*/

    /*sm9_algorithm.cuh-begin*/
    #ifndef sm9_algorithm_cuh
    #define sm_algorithm_cuh
    // #include <stdio.h>
    // //#include <stdlib.h>
    // #include <string.h>

    // #define KDF_XOR 0
    // #define KDF_SM4 1

    // #define AKE_SENDER 0
    // #define AKE_RECEIVER 1


    // //define error code
    // #define LIB_NOT_INIT    1
    // #define NOT_ON_G1       2
    // #define NOT_ON_G2       3
    // #define SIGN_ZERO_ERROR 4
    // #define VERIFY_ERROR_1  5
    // #define VERIFY_ERROR_2  6
    // #define VERIFY_ERROR_3  7
    // #define PAIRING_ERROR   8
    // #define BLOCK_ERROR     9
    // #define DECRYPT_ERROR   10

    // /* define Twist type */

    // #define SM_QUADRATIC 2
    // #define SM_CUBIC_M   0x3A
    // #define SM_CUBIC_D   0x3B
    // #define SM_QUARTIC_M 0x4A
    // #define SM_QUARTIC_D 0x4B
    // #define SM_SEXTIC_M  0x6A
    // #define SM_SEXTIC_D  0x6B

    // typedef struct sm9_msk_st {
    //     unsigned int keylen;
    //     unsigned char *msk;
    // }SM9_MSK;   //  主密钥，数据格式上不区分签名、加解密或密钥协商

    // typedef struct sm9_mspk_st {
    //     unsigned int secLevel;
    //     unsigned char *x1;
    //     unsigned char *x2;
    //     unsigned char *y1;
    //     unsigned char *y2;
    // }SM9_MSPK; // 签名主密钥，代表扩域上椭圆曲线的一个点

    // typedef struct sm9_mcpk_st {
    //     unsigned int secLevel;
    //     unsigned char *x;
    //     unsigned char *y;
    // }SM9_MCPK;  //  加密主密钥，代表基域上椭圆曲线的一个点

    // typedef struct sm9_mkpk_st {
    //     unsigned int secLevel;
    //     unsigned char *x;
    //     unsigned char *y;
    // }SM9_MKPK; //  密钥协商主密钥，代表基域上椭圆曲线的一个点

    // typedef struct sm9_pubkey_st {
    //     unsigned int keylen;
    //     unsigned char *pk;
    // }SM9_PK;  // 用户公钥（ID）

    // typedef struct sm9_signseckey_st {
    //     unsigned int secLevel;
    //     unsigned char *x;
    //     unsigned char *y;
    // }SM9_SSK; //用户签名私钥，表示基域上椭圆曲线的一个点

    // typedef struct sm9_cipherseckey_st {
    //     unsigned int secLevel;
    //     unsigned char *x1;
    //     unsigned char *x2;
    //     unsigned char *y1;
    //     unsigned char *y2;
    // }SM9_CSK; // 用户加解密私钥，表示扩域上椭圆曲线的一个点

    // typedef struct sm9_keyexseckey_st {
    //     unsigned int secLevel;
    //     unsigned char *x1;
    //     unsigned char *x2;
    //     unsigned char *y1;
    //     unsigned char *y2;
    // }SM9_KSK; //用户密钥协商私钥，表示扩域上椭圆曲线的一个点

    // typedef struct sm9_sign_st {
    //     unsigned int secLevel;
    //     unsigned char *h;
    //     unsigned char *xs;
    //     unsigned char *ys;
    //     unsigned char type;
    // } SM9_Sign; // 签名，h表示一段hash值，xs，ys表示基域上椭圆曲线的一个点

    // typedef struct sm9_cipher_st {
    //     unsigned int secLevel;
    //     unsigned long entype;
    //     unsigned char* x;
    //     unsigned char* y;
    //     unsigned char* c3;
    //     unsigned char* cp;
    //     unsigned int cplen;
    // } SM9_Cipher;  // 密文，（x,y）表示基域上椭圆曲线的一个点，C3表示验证信息，cp表示xor或者对称加密密文，entype表示采用sm4加密或xor加密

    // typedef struct sm9_sender_st {
    //     unsigned int secLevel;
    //     unsigned char *x;
    //     unsigned char *y;
    // }SM9_Send;  //sm9密钥协商中第一阶段（发送阶段）数据结构

    struct Comp
    {
        double r,i;
        //Comp(double x=0,double y=0):r(x),i(y){}
        friend Comp operator + (Comp a,Comp b)
        {
            Comp tmp;
            tmp.r = a.r + b.r;
            tmp.i = a.i + b.i;
            return tmp;
        }
        friend Comp operator - (Comp a,Comp b)
        {
            Comp tmp;
            tmp.r = a.r - b.r;
            tmp.i = a.i - b.i;
            return tmp;
        }
        friend Comp operator * (Comp a,Comp b)
        {
            Comp tmp;
            tmp.r = a.r * b.r - a.i * b.i;
            tmp.i = a.r * b.i + a.i * b.r;
            return tmp;
        }
    }; // sm9签名验签辅助计算


    //初始化sm9算法
    //@seclevel 表示算法安全参数，（byte长度，标准为32）
    //@t sm9复承参数t
    //@q sm9基域素数q
    //@a sm9椭圆曲线方程参数a
    //@b sm9椭圆曲线方程参数b
    //@n sm9基域上椭圆曲线的阶n
    //@xp1 sm9 G1群生成元x坐标
    //@yp1 sm9 G1群生成元y坐标
    //@xq1 sm9 G2群生成元x坐标分量
    //@xq2 sm9 G2群生成元x坐标分量
    //@yq1 sm9 G2群生成元y坐标分量
    //@yq2 sm9 G2群生成元y坐标分量
    //@curve 等于0时，配置内置标准参数，等于1时配置前面的参数
    //@TWIST_TYPE 椭圆曲线twist类型，标准为MR_SEXTIC_M
    //返回值
    //0 设置成功
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_Init_cuda(unsigned int curve, int TWIST_TYPE, unsigned int seclevel,unsigned char* t, unsigned char* q, unsigned char* a, unsigned char* b, unsigned char* n, unsigned char* xp1, unsigned char* yp1, unsigned char* xq1, unsigned char* xq2, unsigned char* yq1, unsigned char* yq2);

    //关闭SM9算法，清除缓存
    __device__ void SM9_Free_cuda();

    //初始化sm9签名模块，完成预计算
    //@x1 sm9签名主公钥的x分量1
    //@x2 sm9签名主公钥的x分量2
    //@y1 sm9签名主公钥的y分量1
    //@y1 sm9签名主公钥的y分量2
    //@gGtchar 初始化SM9需要用到的预计算结果
    //返回值 预计算结果
    //第一次调用gc = SM9_Set_Sign(x1,x2,y1,y2,NULL)，以后可以调用SM9_Set_Sign(NULL,NULL,NULL,NULL,gc)完成初始化
    //可将gc写入文件或数据库，以后从文件或数据库中读取
    unsigned char* SM9_Set_Sign_cuda(unsigned char* x1, unsigned char* x2, unsigned char* y1, unsigned char* y2, unsigned char* gGtchar);

    //关闭sm9签名模块
    __device__ void SM9_Close_Sign_cuda();

    //初始化sm9加密模块，完成预计算
    //@x sm9加密主公钥的x分量
    //@y sm9加密主公钥的y分量
    //@eGtchar 初始化SM9需要用到的预计算结果
    //返回值 预计算结果
    //第一次调用gc = SM9_Set_Encrypt(x,y,NULL)，以后可以调用SM9_Set_Encrypt(NULL,NULL,gc)完成初始化
    //可将gc写入文件或数据库，以后从文件或数据库中读取
    __device__ unsigned char* SM9_Set_Encrypt_cuda(unsigned char* x, unsigned char* y, unsigned char* eGtchar);

    //关闭sm9加密模块
    __device__ void SM9_Close_Encrypt_cuda();

    //初始化sm9密钥协商模块，完成预计算
    //@x sm9密钥协商主公钥的x分量
    //@y sm9密钥协商主公钥的y分量
    //@eGtchar 初始化SM9需要用到的预计算结果
    //返回值 预计算结果
    //第一次调用gc = SM9_Set_KeyExchange(x,y,NULL)，以后可以调用SM9_Set_KeyExchange(NULL,NULL,gc)完成初始化
    //可将gc写入文件或数据库，以后从文件或数据库中读取
    __device__ unsigned char* SM9_Set_KeyExchange_cuda(unsigned char* x, unsigned char* y,unsigned char* kGtchar);

    //关闭sm9秘钥交换模块
    __device__ void SM9_Close_KeyExchange_cuda();

    //根据密钥字符串w，初始化一个主私钥
    __device__ SM9_MSK SM9_MSK_New_cuda(int secLevel,unsigned char* w);

    //根据安全参数secLevel初始化一个签名主公钥，标准为32
    __device__ SM9_MSPK SM9_MSPK_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个加密主公钥，标准为32
    __device__ SM9_MCPK SM9_MCPK_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个密钥协商主公钥，标准为32
    __device__ SM9_MKPK SM9_MKPK_New_cuda(int secLevel);

    //根据id生产一个用户公钥
    __device__ SM9_PK SM9_PK_New_cuda(int len,unsigned char* w);

    //根据安全参数secLevel初始化一个用户签名私钥，标准为32
    __device__ SM9_SSK SM9_SSK_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个用户加解密私钥，标准为32
    __device__ SM9_CSK SM9_CSK_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个用户密钥协商私钥，标准为32
    __device__ SM9_KSK SM9_KSK_New_cuda(int secLevel);

    //根据安全参数secLevel初始化一个签名结构体，标准为32
    __device__ SM9_Sign SM9_Sign_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个密文结构体，标准为32
    __device__ SM9_Cipher SM9_Cipher_New_cuda(int secLevel);
    //根据安全参数secLevel初始化一个发送结构体，标准为32
    __device__ SM9_Send SM9_Send_New_cuda(int secLevel);




    //释放一个主私钥
    __device__ void SM9_MSK_Free_cuda(SM9_MSK *msk);
    //释放一个签名主公钥
    __device__ void SM9_MSPK_Free_cuda(SM9_MSPK *mpk);
    //释放一个加解密主公钥
    __device__ void SM9_MCPK_Free_cuda(SM9_MCPK *mpk);
    //释放一个密钥协商主公钥
    __device__ void SM9_MKPK_Free_cuda(SM9_MKPK *mpk);

    //释放一个用户公钥
    __device__ void SM9_PK_Free_cuda(SM9_PK *pk);
    //释放一个用户签名私钥
    __device__ void SM9_SSK_Free_cuda(SM9_SSK *sk);
    //释放一个用户加解密私钥
    __device__ void SM9_CSK_Free_cuda(SM9_CSK *sk);
    //释放一个用户密钥协商私钥
    __device__ void SM9_KSK_Free_cuda(SM9_KSK *sk);


    //释放一个签名结构体
    __device__ void SM9_Sign_Free_cuda(SM9_Sign *s);
    //释放一个密文结构体
    __device__ void SM9_Cipher_Free_cuda(SM9_Cipher *c);
    //释放一个发送结构体
    __device__ void SM9_Send_Free_cuda(SM9_Send *s);

    //根据主私钥生成主签名公钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenMSignPubKey_cuda(SM9_MSK *msk, SM9_MSPK *mspk);

    //根据主私钥生成主加解密公钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenMEncryptPubKey_cuda(SM9_MSK *msk, SM9_MCPK *mcpk);

    //根据主私钥生成主密钥协商公钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenMKeyExchangePubKey_cuda(SM9_MSK *msk, SM9_MKPK *mcpk);

    //根据主私钥，用户公钥生成用户签名私钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenSignSecKey_cuda(SM9_SSK *sk, SM9_PK *pk,SM9_MSK *msk);

    //根据主私钥，用户公钥生成用户加解密私钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenEncryptSecKey_cuda(SM9_CSK *sk, SM9_PK *pk,SM9_MSK *msk);

    //根据主私钥，用户公钥生成用户密钥协商私钥
    //返回码
    //0 生成成功
    //1 系统未初始化
    __device__ int SM9_GenKeyExchangeSecKey_cuda(SM9_KSK *sk, SM9_PK *pk,SM9_MSK *msk);


    //=============For Signature algorithm=================
    //@mes      the message for signature
    //@meslen   the length of the message
    //@ran      the random number for sign (must be 32 char, if it's length less than 32 , plese add 0x00 at the first)
    //@sk       the secretkey for signature
    //@sign     the signature

    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_Signature_cuda(unsigned char* mes,unsigned int meslen,unsigned char* ran,SM9_SSK *sk, SM9_Sign *sign);

    //@mes      the message for signature
    //@meslen   the length of the message
    //@sing     the signature
    //@pk       the public key(id) of the signer
    //@mpk      the master public key for signtuer, please set it as NULL

    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_Verify_cuda(unsigned char *mes,unsigned int meslen, SM9_Sign *sign, SM9_PK *pk, SM9_MSPK *mpk);
    //=============For Encryption algorithm=================
    //
    //@mes      the message for encryption
    //@meslen   the length of the message
    //@KDF_ID   KDF_XOR for xor encrypt function, KDF_SM4 for SM4 and the mes will be add 0xC0 at last,
    //@ran      the random number for encrypt (must be 32 char, if it's length less than 32 , plese add 0x00 at the first)
    //@pk       the public key(id) for the receiver
    //@cip      the ciphertext

    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_Encrypt_cuda(unsigned char *mes, unsigned int meslen, unsigned int KDF_ID, unsigned char *ran, SM9_PK *pk, SM9_Cipher *cip);

    //@pk       the public key(id) for the receiver
    //@KDF_ID  KDF_XOR for xor encrypt function, KDF_SM4 for SM4 and the mes will be add 0xC0 at last,
    //@sk       the secret key for the receiver
    //@cip      the ciphertext
    //@mes      the plaintext
    //@meslen   the length of the plaintext

    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_Decrypt_cuda(SM9_PK *pk, unsigned int KDF_ID, SM9_CSK *sk, SM9_Cipher *cip,unsigned char *mes,unsigned int *meslen);
    //=============For KeyExchange algorithn================

    //@ran      the random number for sendstep (must be 32 char, if it's length less than 32 , plese add 0x00 at the first)
    //@pk       the publickey(id) of the receiver
    //@se       the send in the first step

    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_SendStep_cuda(unsigned char *ran, SM9_PK *pk, SM9_Send *se);

    //@ran      the random number used in sendstep (must be 32 char, if it's length less than 32 , plese add 0x00 at the first)
    //@S        the "send" generater by self
    //@R        the "send" from the another party
    //@SP       the publickey(id) of self
    //@RP       the publickey(id) of the receiver (another party)
    //@ksk      the secret key for keyexchange of self
    //@sklen    the length of the sk;
    //@S1       hold S1 for checkstep
    //@S2       send S2 to receiver for the checkstep
    //@SK       the exchanged key
    //@SorR     set AKE_SENDER if you are the sender party, and AKE_RECEIVER if you are the receiver party



    // 返回码
    //0 计算成功
    //1 没有完成预配置，库没有启动
    //2 G1生成元配置错误
    //3 G2生成元配置错误
    __device__ int SM9_ReceiveStep_cuda(unsigned char *ran,SM9_Send *S, SM9_Send *R,SM9_PK *SP, SM9_PK *RP, SM9_KSK *ksk, unsigned int sklen,unsigned char *S1, unsigned char *S2, unsigned char *SK,unsigned int SorR);

    //对比R与G是否一致
    //0 表示一致
    //1 表示不一致
    __device__ int SM9_CheckStep_cuda(unsigned char *R, unsigned char *G);
    #endif
    /*sm9_algorithm.cuh-end*/



#ifdef __cplusplus
}
#endif

#endif