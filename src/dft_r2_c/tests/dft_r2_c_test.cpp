/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file dft_r2_c_test.cpp
 * @brief Rfft1D算子单元测试
 */

#include "cann_ops_fft.h"
#include "dft_r2_c_test.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>

void test_dft_r2_c(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_dft_r2_c");

    // 1. 准备输入数据
    int batch = 32, Nfft = 256;
    const int64_t inSignal = Nfft;
    const int64_t outSignal = Nfft / 2 + 1;
    const int64_t tensorInSize = batch * inSignal;
    const int64_t tensorOutSize = batch * outSignal;

    std::vector<float> inputHostData(tensorInSize, 0);
    for (int i = 0; i < tensorInSize; i++) {
        inputHostData[i] = i;
    }
    // 输出数据缓冲区 (Complex64 需要两倍 float 空间)
    std::vector<float> outHostData(tensorOutSize * 2, 0);

    // 2. 创建 Device 侧 Tensor
    void *inputDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclError computeRet = aclFftR2C(stream, inputHostData.data(), outHostData.data(), Nfft, batch, true);

    TEST_ASSERT(stats, computeRet == ACL_SUCCESS, "return code failed");
    // TEST_ASSERT_ARRAY_NEAR(stats, y, expected, y_size, 1e-4f, "array mismatch");

    TEST_CASE_PASS(stats, "test_dft_r2_c");
}

// Export function
namespace DftR2CTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_dft_r2_c(stream, stats);
    }
}

// Auto register to test framework
REGISTER_OP_TEST(DftR2C)
