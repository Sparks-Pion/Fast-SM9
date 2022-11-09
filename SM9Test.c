#include <stdio.h>
#include <string.h>
#include <time.h>

#define TEST 1
time_t start, end;

#include "sm9/sm9_algorithm.h"
#include "sm9/sm9_utils.h"

#include "cuda/sm9_cuda.cuh"

void test_signature();

void test_key_exchange();

void test_encryption();

int main(int argc, char **argv) {
    //===========签名测试================
    test_signature();
    //===========密钥交换================
    test_key_exchange();
    //===========加密测试================
    test_encryption();
    //===========并行测试================
    main_GPU();
    return 0;
}

void start_count_modmulti(){
//    start_zzn12_mul_cnt();
//    start_zzn4_mul_cnt();
//    start_zzn3_mul_cnt();
//    start_zzn2_mul_cnt();
//    start_ecn2_mul_cnt();
//    start_nres_modmult_cnt();
}

void print_count_modmulti() {
//    printf("=======================\n");
//    printf("zzn12_mul_cnt = %lu\n", get_zzn12_mul_cnt());
//    printf("zzn4_mul_cnt = %lu\n", get_zzn4_mul_cnt());
//    printf("zzn3_mul_cnt = %lu\n", get_zzn3_mul_cnt());
//    printf("zzn2_mul_cnt = %lu\n", get_zzn2_mul_cnt());
//    printf("ecn2_mul_cnt = %lu\n", get_ecn2_mul_cnt());
//    printf("nres_modmult_cnt = %lu\n", get_nres_modmult_cnt());
//    printf("=======================\n");
}

void test_signature() {
    start_count_time();

    SM9_Init(0, 0, 32,
             NULL, NULL, NULL, NULL, NULL,
             NULL, NULL, NULL, NULL, NULL, NULL);

    // 签名主私钥 mks = 000130E7 8459D785 45CB54C5 87E02CF4
    //                80CE0B66 340F319F 348A1D5B 1F2DC5F4
    unsigned char msk_bytes[32] = "\x00\x01\x30\xE7\x84\x59\xD7\x85\x45\xCB\x54\xC5\x87\xE0\x2C\xF4"
                                  "\x80\xCE\x0B\x66\x34\x0F\x31\x9F\x34\x8A\x1D\x5B\x1F\x2D\xC5\xF4";
    SM9_MSK msk = SM9_MSK_New(32, msk_bytes);

    // 签名主公钥 mpk =
    // ((9F64080B 3084F733 E48AFF4B 41B56501 1CE0711C 5E392CFB 0AB1B679 1B94C408,
    //   29DBA116 152D1F78 6CE843ED 24A3B573 414D2177 386A92DD 8F14D656 96EA5E32)
    //  (69850938 ABEA0112 B57329F4 47E3A0CB AD3E2FDB 1A77F335 E89E1408 D0EF1C25,
    //   41E00A53 DDA532DA 1A7CE027 B7A46F74 1006E85F 5CDFF073 0E75C05F B4E3216D))
    SM9_MSPK mpk = SM9_MSPK_New(32);
    SM9_GenMSignPubKey(&msk, &mpk);
    if (SM9_Set_Sign(mpk.x1, mpk.x2, mpk.y1, mpk.y2, NULL) == NULL) {
        puts("SM9_GenMSignPubKey error");
        exit(1);
    }

    // 实体 A 的标识 id = Alice
    unsigned char id[5] = "Alice";
    // id 生成的公钥
    SM9_PK pk = SM9_PK_New(5, id);

    // 签名私钥 sk =
    // x:A5702F05 CF131530 5E2D6EB6 4B0DEB92 3DB1A0BC F0CAFF90 523AC875 4AA69820
    // y:78559A84 4411F982 5C109F5E E3F52D72 0DD01785 392A727B B1556952 B2B013D3
    SM9_SSK sk = SM9_SSK_New(32);
    SM9_GenSignSecKey(&sk, &pk, &msk);

    SM9_Sign sign = SM9_Sign_New(32);

    // 消息 M = Chinese IBS standard
    unsigned char msg[20] = "Chinese IBS standard";
    // 随机数 r = 033C86 16B06704 813203DF D0096502 2ED15975 C662337A ED648835 DC4B1CBE
    unsigned char rand[32] = "\x00\x03\x3C\x86\x16\xB0\x67\x04\x81\x32\x03\xDF\xD0\x09\x65\x02"
                             "\x2E\xD1\x59\x75\xC6\x62\x33\x7A\xED\x64\x88\x35\xDC\x4B\x1C\xBE";
    
    //printf("\n===================signature-test-start===================\n");
    start = clock();
    start_count_modmulti();
    for (int i = 0; i < TEST; i++) {
        SM9_Signature(msg, 20, rand, &sk, &sign);
    }
    end = clock();
    printf("sign %d time is %lf ms\n", TEST, (double) (end - start) / CLOCKS_PER_SEC * 1000);
    print_count_modmulti();
    //printf("====================signature-test-end====================\n\n");

    //printf("\n===================verify-test-start===================\n");
    start = clock();
    start_count_modmulti();
    for (int i = 0; i < TEST; i++) { // 验证函数
        int res = SM9_Verify(msg, 20, &sign, &pk, NULL);
        if (res)
            printf("verify error at %d = %d\n", i, res);
    }
    end = clock();
    printf("verify %d time is %lf ms\n", TEST, (double) (end - start) / CLOCKS_PER_SEC * 1000);
    printf("average pairing time is %lf ms\n", end_count_time());
    print_count_modmulti();
    //printf("====================verify-test-end====================\n\n");

    SM9_Sign_Free(&sign);
    SM9_SSK_Free(&sk);
    SM9_PK_Free(&pk);
    SM9_MSPK_Free(&mpk);
    SM9_MSK_Free(&msk);
    SM9_Free();
};

void test_key_exchange() {
    start_count_time();

    SM9_Init(0, 0, 32,
             NULL, NULL, NULL, NULL, NULL,
             NULL, NULL, NULL, NULL, NULL, NULL);

    // 加密主私钥 mks = 0002E65B 0762D042 F51F0D23 542B13ED
    //                8CFA2E9A 0E720636 1E013A28 3905E31F
    unsigned char msk_bytes[32] = "\x00\x02\xE6\x5B\x07\x62\xD0\x42\xF5\x1F\x0D\x23\x54\x2B\x13\xED"
                                  "\x8C\xFA\x2E\x9A\x0E\x72\x06\x36\x1E\x01\x3A\x28\x39\x05\xE3\x1F";
    SM9_MSK msk = SM9_MSK_New(32, msk_bytes);

    // 加密主公钥 mpk =
    //  (91745426 68E8F14A B273C094 5C3690C6 6E5DD096 78B86F73 4C435056 7ED06283,
    //   54E598C6 BF749A3D ACC9FFFE DD9DB686 6C50457C FC7AA2A4 AD65C316 8FF74210)
    SM9_MKPK mpk = SM9_MKPK_New(32);
    SM9_GenMKeyExchangePubKey(&msk, &mpk);
    if (SM9_Set_KeyExchange(mpk.x, mpk.y, NULL) == NULL) {
        printf("SM9_Set_KeyExchange error\n");
        exit(1);
    }

    // 实体 A 的标识 id_A = Alice
    unsigned char id_A[5] = "Alice";
    SM9_PK pk_A = SM9_PK_New(5, id_A);
    SM9_KSK sk_A = SM9_KSK_New(32);
    SM9_GenKeyExchangeSecKey(&sk_A, &pk_A, &msk);

    // 实体 B 的标识 id_B = Bob
    unsigned char id_B[3] = "Bob";
    SM9_PK pk_B = SM9_PK_New(3, id_B);
    SM9_KSK sk_B = SM9_KSK_New(32);
    SM9_GenKeyExchangeSecKey(&sk_B, &pk_B, &msk);

    // 随机数
    // 取 rA为：00005879 DD1D51E1 75946F23 B1B41E93 BA31C584 AE59A426 EC1046A4 D03B06C8
    unsigned char rand_A[32 + 1] = "\x00\x00\x58\x79\xDD\x1D\x51\xE1\x75\x94\x6F\x23\xB1\xB4\x1E\x93"
                                   "\xBA\x31\xC5\x84\xAE\x59\xA4\x26\xEC\x10\x46\xA4\xD0\x3B\x06\xC8";
    // 取 rB为: 00018B98 C44BEF9F 8537FB7D 071B2C92 8B3BC65B D3D69E1E EE213564 905634FE
    unsigned char rand_B[32 + 1] = "\x00\x00\x18\xB9\x8C\x44\xBE\xF9\xF8\x53\x7F\xB7\xD0\x71\xB2\xC9"
                                   "\x28\xB3\xBC\x65\xBD\x3D\x69\xE1\xEE\x21\x35\x64\x90\x56\x34\xFE";

    SM9_Send send_A = SM9_Send_New(32);
    SM9_Send send_B = SM9_Send_New(32);
    SM9_SendStep(rand_A, &pk_B, &send_A);
    SM9_SendStep(rand_B, &pk_A, &send_B);

    unsigned char S1_A[32];
    unsigned char S2_A[32];
    unsigned char SK_A[16];
    unsigned char S1_B[32];
    unsigned char S2_B[32];
    unsigned char SK_B[16];

    //printf("\n===================key exchange-test-start===================\n");
    start = clock();
    start_count_modmulti();
    for (int i = 0; i < TEST; i++) {
        SM9_ReceiveStep(rand_A, &send_A, &send_B, &pk_A, &pk_B, &sk_A, 16, S1_A, S2_A, SK_A, AKE_SENDER);
        SM9_ReceiveStep(rand_B, &send_B, &send_A, &pk_B, &pk_A, &sk_B, 16, S1_B, S2_B, SK_B, AKE_RECEIVER);
        if (SM9_CheckStep(S1_A, S2_B))
            printf("error at step 1\n");
        if (SM9_CheckStep(S2_A, S1_B))
            printf("error at step 2\n");
    }
    end = clock();
    printf("key exchange %d time is %lf ms\n", TEST, (double) (end - start) / CLOCKS_PER_SEC * 1000);
    printf("average pairing time is %lf ms\n", end_count_time());
    print_count_modmulti();
    //printf("\n====================key exchange-test-end====================\n");

    SM9_Send_Free(&send_A);
    SM9_Send_Free(&send_B);
    SM9_KSK_Free(&sk_A);
    SM9_KSK_Free(&sk_B);
    SM9_PK_Free(&pk_A);
    SM9_PK_Free(&pk_B);
    SM9_MKPK_Free(&mpk);
    SM9_MSK_Free(&msk);
    SM9_Free();
}

void test_encryption(){
    start_count_time();

    SM9_Init(0, 0, 32,
             NULL, NULL, NULL, NULL, NULL,
             NULL, NULL, NULL, NULL, NULL, NULL);

    // 加密主私钥 mks = 0001EDEE 3778F441 F8DEA3D9 FA0ACC4E
    //                07EE36C9 3F9A0861 8AF4AD85 CEDE1C22
    unsigned char msk_bytes[32] = "\x00\x01\xED\xEE\x37\x78\xF4\x41\xF8\xDE\xA3\xD9\xFA\x0A\xCC\x4E"
                                  "\x07\xEE\x36\xC9\x3F\x9A\x08\x61\x8A\xF4\xAD\x85\xCE\xDE\x1C\x22";
    SM9_MSK msk = SM9_MSK_New(32, msk_bytes);

    // 加密主公钥 mpk =
    //  (787ED7B8 A51F3AB8 4E0A6600 3F32DA5C 720B17EC A7137D39 ABC66E3C 80A892FF,
    //   769DE617 91E5ADC4 B9FF85A3 1354900B 20287127 9A8C49DC 3F220F64 4C57A7B1)
    SM9_MCPK mpk = SM9_MCPK_New(32);
    SM9_GenMEncryptPubKey(&msk, &mpk);
    if (SM9_Set_Encrypt(mpk.x, mpk.y, NULL) == NULL) {
        printf("SM9_Set_Encrypt error\n");
        exit(1);
    }

    // 实体 B 的标识 IDB: Bob
    unsigned char id[3] = "Bob";

    // 加密公钥 pk = 426F62
    SM9_PK pk = SM9_PK_New(3, id);

    // 生成加密私钥
    SM9_CSK ck = SM9_CSK_New(32);
    SM9_GenEncryptSecKey(&ck, &pk, &msk);

    // 消息 M = Chinese IBS standard
    unsigned char msg[20] = "Chinese IBS standard";

    // 随机数 r = 0000AAC0 541779C8 FC45E3E2 CB25C12B 5D2576B2 129AE8BB 5EE2CBE5 EC9E785C
    unsigned char rand[32] = "\x00\x00\xAA\xC0\x54\x17\x79\xC8\xFC\x45\xE3\xE2\xCB\x25\xC1\x2B"
                             "\x5D\x25\x76\xB2\x12\x9A\xE8\xBB\x5E\xE2\xCB\xE5\xEC\x9E\x78\x5C";

    // 定义明文
    unsigned char plain[32];
    unsigned int plainlen;

    // 定义密文
    SM9_Cipher cipher = SM9_Cipher_New(32);

    //printf("\n===================encrypt-test-start===================\n");
    start = clock();
    start_count_modmulti();
    for (int i = 0; i < TEST; i++) {
        SM9_Encrypt(msg, 20, KDF_SM4, rand, &pk, &cipher);
    }
    end = clock();
    printf("Encrypt %d time is %lf ms\n", TEST, (double) (end - start) / CLOCKS_PER_SEC * 1000);
    print_count_modmulti();
    //printf("====================encrypt-test-end====================\n\n");

    //printf("\n===================decrypt-test-start===================\n");
    start = clock();
    start_count_modmulti();
    for (int i = 0; i < TEST; i++) {
        int res = SM9_Decrypt(&pk, KDF_SM4, &ck, &cipher, plain, &plainlen);
        if (res)
            printf("decrypt error code is %d \n", res);
    }
    end = clock();
    printf("Decrypt %d time is %lf ms\n", TEST, (double) (end - start) / CLOCKS_PER_SEC * 1000);
    printf("average pairing time is %lf ms\n", end_count_time());
    print_count_modmulti();
    //printf("====================decrypt-test-end====================\n\n");

    SM9_CSK_Free(&ck);
    SM9_PK_Free(&pk);
    SM9_MCPK_Free(&mpk);
    SM9_MSK_Free(&msk);
    SM9_Cipher_Free(&cipher);
    SM9_Free();
}