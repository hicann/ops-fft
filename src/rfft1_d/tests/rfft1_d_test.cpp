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
#include <iostream>
#include <vector>
#include <cmath>

void test_basic_rfft1_d(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_basic_rfft1_d");

    int64_t n = 8;  // 输入张量最后一维
    int64_t norm = 1;  // 归一化选项
	uint32_t batches = 1;  // 输入张量除了最后一维的所有维度相乘

    std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<float> y(n);

    std::vector<float> expected = {
        36.0, 0.0, 
        -4.0, 9.6569, 
        -4.0, 4.0, 
        -4.0, 1.6569, 
        -4.0, 0.0
        };

    // 3. 调用CANN算子库API（需要根据具体API名称修改）
    aclError result = aclfftRfft1D(x.data(), y.data(), n, norm, batches, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y, expected, n, 1e-4f, "array mismatch");

    TEST_CASE_PASS(stats, "test_basic_rfft1_d");
}

// 导出函数
namespace Rfft1DTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_basic_rfft1_d(stream, stats);
    }
}

// 自动注册到测试框架
REGISTER_OP_TEST(Rfft1D)
