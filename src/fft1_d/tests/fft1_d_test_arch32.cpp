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

namespace Fft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<C2CTestCase> cases = {
            // c2c n<=256 / SelectCore -> arch32/dft/ : dft
            {"dft_forward",      4,     1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"dft_backward",     4,     1,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"dft_config",       256,   10,  ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            // c2c stride>1 -> arch32/fft_stride/ : fft_stride
            {"stride_forward",   256,   128, ACLFFT_FORWARD,  ACLFFT_VERTICAL},
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
            {"mix_roundtrip_n210",    210,   1,   0,               ACLFFT_HORIZONTAL},
            {"mix_bwd_n210_b4",       210,   4,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"mix_fwd_n1536_b4",     1536,   4,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
        };
        OpsFftTest::RunBatchTests(stream, stats, "fft1_d", cases, run_c2c_case);
    }
}

REGISTER_OP_TEST(Fft1DApiIntegration)
