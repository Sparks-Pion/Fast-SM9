#ifndef __SM9_CUDA_CUH__
#define __SM9_CUDA_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "../sm9/sm9_algorithm.h"

#define TEST_CUDA 20000

#ifdef __cplusplus
extern "C" {
#endif
    /*API of sm9_cuda*/

    int SM9_Signature_Init_GPU(unsigned int N, unsigned char *cks_list, unsigned int cks_len, unsigned char *id_list, unsigned int id_len, unsigned char *rand_list, unsigned int rand_len, unsigned char *msg_list, unsigned int msg_len, SM9_PK *pk_list, SM9_SSK *sk_list, SM9_Sign *sign_list);
    
    int SM9_Signature_GPU(unsigned int N, unsigned char *mes, unsigned int meslen, unsigned char *ran, unsigned int ranlen, SM9_SSK *sk, SM9_Sign *sign);
    
    int SM9_Verify_GPU(unsigned int N, unsigned char *msg_list, unsigned int msg_len, SM9_PK *pk_list, SM9_Sign *sign_list);

    int main_GPU();

#ifdef __cplusplus
}
#endif

#endif