/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file rfft1_d_test.cpp
 * @brief Rfft1D算子单元测试
 */

#include "cann_ops_fft.h"
#include "rfft1_d_test.h"
#include "utils/rfft_expected_loader.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>

void test_rfft1_d_n8(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_rfft1_d_n8");

    int64_t n = 8;  // 输入张量最后一维
    int64_t norm = 1;  // 归一化选项
    uint32_t batches = 1;  // 输入张量除了最后一维的所有维度相乘

    std::vector<float> x(n);
    std::iota(x.begin(), x.end(), 1.0f);
    std::vector<float> expected;
    RfftExpected::generateExpectedData(x, n, batches, norm, expected);

    int64_t y_size = (n / 2 + 1) * 2;
    std::vector<float> y(y_size);

    aclError result = aclfftRfft1D(x.data(), y.data(), n, norm, batches, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y, expected, y_size, 1e-4f, "array mismatch");

    TEST_CASE_PASS(stats, "test_rfft1_d_n8");
}

// Export function
namespace Rfft1DTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_rfft1_d_n8(stream, stats);
    }
}

// Auto register to test framework
REGISTER_OP_TEST(Rfft1D)
