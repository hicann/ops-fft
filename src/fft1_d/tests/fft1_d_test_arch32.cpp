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
            {"dft_forward",      4,     1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"dft_backward",     4,     1,   ACLFFT_BACKWARD, ACLFFT_HORIZONTAL},
            {"dft_config",       256,   10,  ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"stride_forward",   256,   128, ACLFFT_FORWARD,  ACLFFT_VERTICAL},
            {"fft_n_forward",    32768, 1,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_n_roundtrip",  32768, 1,   0,               ACLFFT_HORIZONTAL},
            {"fft_n_config",     32768, 10,  ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_b_n512",       512,   2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_b_n2048",      2048,  2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
            {"fft_b_n16384",     16384, 2,   ACLFFT_FORWARD,  ACLFFT_HORIZONTAL},
        };
        OpsFftTest::RunBatchTests(stream, stats, "fft1_d", cases, run_c2c_case);
    }
}

REGISTER_OP_TEST(Fft1DApiIntegration)
