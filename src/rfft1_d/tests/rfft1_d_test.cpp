/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file rfft1_d_test.cpp
 * @brief R2C FFT API 集成测试 - 测试 aclfftExecR2C 接口调用 rfft1_d 算子
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "cann_ops_fft.h"
#include "rfft1_d_test.h"

static const uint32_t COMPLEX_PART = 2;

void test_r2c_api_basic(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_r2c_api_basic");

    // 测试参数
    const int n = 8;
    const int batch = 1;

    // 1. 创建 Plan
    aclfftHandle plan;
    aclfftResult res = aclfftPlan1d(&plan, n, ACLFFT_R2C, batch);
    TEST_ASSERT(stats, res == ACLFFT_SUCCESS, "aclfftPlan1d failed");

    // 2. 准备输入数据
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // 3. 准备输出数据
    int output_size = (n / 2) + 1;
    std::vector<aclfftComplex> output(output_size * batch);

    // 4. 调用 aclfftExecR2C（底层会处理内存拷贝）
    res = aclfftExecR2C(plan, input.data(), output.data());
    TEST_ASSERT(stats, res == ACLFFT_SUCCESS, "aclfftExecR2C failed");

    // 5. 验证结果（rfft1_d BACKWARD 模式）
    std::vector<aclfftComplex> expected = {
        {36.0, 0.0},
        {-4.0, 9.6569},
        {-4.0, 4.0},
        {-4.0, 1.6569},
        {-4.0, 0.0}
    };

    // 比较实部和虚部
    std::vector<float> output_flat(output_size * COMPLEX_PART);
    std::vector<float> expected_flat(output_size * COMPLEX_PART);
    for (int i = 0; i < output_size; ++i) {
        output_flat[i * COMPLEX_PART] = output[i].x;
        output_flat[i * COMPLEX_PART + 1] = output[i].y;
        expected_flat[i * COMPLEX_PART] = expected[i].x;
        expected_flat[i * COMPLEX_PART + 1] = expected[i].y;
    }

    TEST_ASSERT_ARRAY_NEAR(stats, output_flat, expected_flat, output_size * COMPLEX_PART, 1e-3f, "result mismatch");

    // 6. 清理
    aclfftDestroy(plan);

    TEST_CASE_PASS(stats, "test_r2c_api_basic");
}

void test_r2c_api_impulse(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_r2c_api_impulse");

    // 测试参数
    const int n = 8;
    const int batch = 1;

    // 1. 创建 Plan
    aclfftHandle plan;
    aclfftResult res = aclfftPlan1d(&plan, n, ACLFFT_R2C, batch);
    TEST_ASSERT(stats, res == ACLFFT_SUCCESS, "aclfftPlan1d failed");

    // 2. 准备输入数据（单位冲激）
    std::vector<float> input(n, 0.0f);
    input[0] = 1.0f;

    // 3. 准备输出数据
    int output_size = (n / 2) + 1;
    std::vector<aclfftComplex> output(output_size);

    // 4. 调用 aclfftExecR2C
    res = aclfftExecR2C(plan, input.data(), output.data());
    TEST_ASSERT(stats, res == ACLFFT_SUCCESS, "aclfftExecR2C failed");

    // 5. 验证结果（所有分量都应该是 1.0）
    std::vector<float> expected(output_size * COMPLEX_PART, 1.0f);  // 实部=1.0, 虚部=0.0
    expected[1] = 0.0f;  // 第一个虚部
    for (int i = 1; i < output_size; ++i) {
        expected[i * COMPLEX_PART + 1] = 0.0f;  // 其他虚部
    }

    std::vector<float> output_flat(output_size * COMPLEX_PART);
    for (int i = 0; i < output_size; ++i) {
        output_flat[i * COMPLEX_PART] = output[i].x;
        output_flat[i * COMPLEX_PART + 1] = output[i].y;
    }

    TEST_ASSERT_ARRAY_NEAR(stats, output_flat, expected, output_size * COMPLEX_PART, 0.01f, "impulse result mismatch");

    // 6. 清理
    aclfftDestroy(plan);

    TEST_CASE_PASS(stats, "test_r2c_api_impulse");
}

// 导出函数
namespace Rfft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_r2c_api_basic(stream, stats);
        test_r2c_api_impulse(stream, stats);
    }
}

// 自动注册到测试框架
REGISTER_OP_TEST(Rfft1DApiIntegration)
