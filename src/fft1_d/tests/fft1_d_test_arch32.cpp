/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft1_d_test.h"
#include "fft1_d/fft1_d.h"

namespace Fft1DApiIntegrationTest {
    static std::vector<C2CTestCase> build_precision_cases() {
        return {
            // c2c n<=256 / SelectCore -> arch32/dft/ : dft
            {"dft_forward",      4,     1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"dft_backward",     4,     1,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"dft_config",       256,   10,  ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            // c2c stride>1 -> arch32/fft_stride/ : fft_stride
            {"stride_forward",   256,   128, ACLFFT_FORWARD,  ACLFFT_VERTICAL},
            {"stride_forward_n4096", 4096, 128, ACLFFT_FORWARD, ACLFFT_VERTICAL},
            // c2c radix-2 n>=32768 / SelectCore -> arch32/fft_n/ : fft_n
            {"fft_n_forward",    32768, 1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_n_roundtrip",  32768, 1,   0,               ACLFFT_HORIZONTAL},
            {"fft_n_config",     32768, 10,  ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            // c2c radix-2 n<32768 -> arch32/fft_b/ : fft_b
            {"fft_b_n512",       512,   2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_b_n2048",      2048,  2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_b_n16384",     16384, 2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            // c2c radix-mix -> arch32/fft_mix/ : fft_mix
            {"mix_fwd_n6",           6,     1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_fwd_n105",         105,   1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_fwd_n210",         210,   1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_bwd_n15",          15,    1,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"mix_prime_n11",        11,    1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_primepow_n121",    121,   1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_mixed_n143",       143,   1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_all4_n2310",       2310,  1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_vlarge_n46200",    46200, 1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"mix_roundtrip_n210",    210,  1,   0,               ACLFFT_HORIZONTAL},
            {"mix_bwd_n210_b4",       210,   4,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"mix_fwd_n1536_b4",     1536,   4,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
        };
    }

    // 负向/防护测试：不依赖 golden 数据，仅断言返回码与进程存活
    // issue #72: VERTICAL plan + 不支持的 n（非 2 的幂或不在 [2^8, 2^18]）应返回错误码而非异常穿透
    static void run_vertical_unsupported_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("vertical_unsupported_n64");
        aclfftHandle plan;
        aclfftResult plan_res = aclfftPlan1d(&plan, 64, ACLFFT_C2C, 128, ACLFFT_VERTICAL);
        if (plan_res != ACLFFT_SUCCESS) {
            TEST_CASE_FAIL_DETAIL(stats, "vertical_unsupported_n64", "plan creation failed");
            return;
        }
        std::vector<aclfftComplex> in(64 * 128);
        std::vector<aclfftComplex> out(64 * 128);
        aclfftResult res = aclfftExecC2C(plan, in.data(), out.data(), ACLFFT_FORWARD);
        aclfftDestroy(plan);
        if (res == ACLFFT_NOT_IMPLEMENTED) {
            TEST_CASE_PASS_DETAIL(stats, "vertical_unsupported_n64",
                                   "unsupported VERTICAL n=64 returns ACLFFT_NOT_IMPLEMENTED, no crash");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "vertical_unsupported_n64",
                                   "expected ACLFFT_NOT_IMPLEMENTED, got " + std::to_string(static_cast<int>(res)));
        }
    }

    // issue #72: 直连 fft_b/stride 入口传不支持的 n，应返回错误码而非 throw 穿透 C 边界
    static void run_host_entry_unsupported_n_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("host_entry_unsupported_n");
        std::vector<float> in(64 * 2, 0.0f);
        std::vector<float> out(64 * 2, 0.0f);
        bool b_ok = aclfftFft1DB(in.data(), out.data(), 64, 1, 1, nullptr) != ACL_SUCCESS;
        bool stride_ok = aclfftFft1DStride(in.data(), out.data(), 64, 2, 1, 1, nullptr) != ACL_SUCCESS;
        if (b_ok && stride_ok) {
            TEST_CASE_PASS_DETAIL(stats, "host_entry_unsupported_n",
                                   "unsupported n rejected with error code, no exception");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "host_entry_unsupported_n",
                                   "unexpected success for unsupported n");
        }
    }

    // issue #76: fft1_d arch32 五个导出 host 入口空指针应返回 ACL_ERROR_INVALID_PARAM 而非崩溃
    static void run_host_entry_null_pointer_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("host_entry_null_pointer");
        float dummy = 0.0f;
        bool all_ok = true;
        if (aclfftFft1DDft(nullptr, &dummy, 256, 1, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftFft1DB(nullptr, &dummy, 512, 2, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftFft1DN(nullptr, &dummy, 32768, 1, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftFft1DMix(nullptr, &dummy, 210, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftFft1DStride(nullptr, &dummy, 256, 128, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "host_entry_null_pointer",
                                   "all 5 exported entries reject null pointers with ACL_ERROR_INVALID_PARAM");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "host_entry_null_pointer",
                                   "some entry did not return ACL_ERROR_INVALID_PARAM");
        }
    }

    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        OpsFftTest::RunBatchTests(stream, stats, "fft1_d", build_precision_cases(), run_c2c_case);
        run_vertical_unsupported_test(stats);
        run_host_entry_unsupported_n_test(stats);
        run_host_entry_null_pointer_test(stats);
    }
}

REGISTER_OP_TEST(Fft1DApiIntegration)
