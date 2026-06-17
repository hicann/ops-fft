/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fft1_d.cpp
 * \brief
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_dft_tilingdata.h"
#include "fft1_d_dft_kernel.h"

static std::vector<float> InitRotationMatrix(int64_t nDoing)
{
    int64_t fftN = static_cast<int64_t>(nDoing);
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;
    std::vector<float> rotationMatrixHost(outSize * inSize, INIT_VALUE);
    std::vector<float> cosTable(fftN);
    std::vector<float> sinTable(fftN);
    for (int64_t i = 0; i < fftN; i++) {
        cosTable[i] = cos(K_2PI * i / fftN);
        sinTable[i] = sin(K_2PI * i / fftN);
    }
    for (int64_t i = 0; i < fftN; i++) {
        for (int64_t j = 0; j < fftN; j++) {
            rotationMatrixHost[(2 * i) * (2 * fftN) + 2 * j] = cosTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i) * (2 * fftN) + 2 * j + 1] = -sinTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i + 1) * (2 * fftN) + 2 * j] = sinTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i + 1) * (2 * fftN) + 2 * j + 1] = cosTable[(i * j) % fftN];
        }
    }
    return rotationMatrixHost;
}

static int SetTilingData(Fft1DDFTTilingData &tiling, uint32_t fftN, int32_t norm,
                         int isInverse, uint32_t coreNum, uint32_t batches)
{
    const int thresholdK = 32768;
    const int cubeDataCountPerLoopSmall = 128;
    const int cubeDataCountPerLoopLarge = 64;

    tiling.batchSize = 1;
    tiling.m = batches;
    tiling.n = fftN * DFT_SIZE_MULTIPLIER;
    tiling.k = fftN * DFT_SIZE_MULTIPLIER;
    tiling.transA = 0;
    tiling.transB = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform -> GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B) {
        if (static_cast<bool>(isInverse)) {
            tiling.transB = 1;
        }
    }
    tiling.m0 = cubeDataCountPerLoopSmall;
    tiling.n0 = cubeDataCountPerLoopSmall;
    tiling.k0 = cubeDataCountPerLoopSmall;
    if (tiling.k > thresholdK) {
        tiling.m0 = cubeDataCountPerLoopLarge;
        tiling.n0 = cubeDataCountPerLoopLarge;
        tiling.k0 = cubeDataCountPerLoopLarge;
    }
    return ACL_SUCCESS;
}

extern "C" aclError aclfftFft1DDft(float *x, float *y, uint32_t n, int32_t norm,
                                uint32_t batches, int isForward, void *stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    auto matrix = InitRotationMatrix(n);
    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();

    const uint32_t inputSize = n * batches * sizeof(float) * 2;
    const uint32_t matrixSize = matrix.size() * sizeof(float);
    const uint32_t outputSize = inputSize;

    void *dev_a = nullptr;
    void *dev_b = nullptr;
    void *dev_c = nullptr;
    void *workspace_ptr = nullptr;

    // Allocate device memory for inputs and output
    CHECK_ACL(aclrtMalloc(&dev_a, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_b, matrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_c, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspace_ptr, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // 智能指针指定删除器，在unique_ptr析构时自动调用指定的函数释放资源
    std::unique_ptr<void, AclrtFreeDeleter> d_a_guard(dev_a);
    std::unique_ptr<void, AclrtFreeDeleter> d_b_guard(dev_b);
    std::unique_ptr<void, AclrtFreeDeleter> d_c_guard(dev_c);
    std::unique_ptr<void, AclrtFreeDeleter> workspace_ptr_guard(workspace_ptr);

    // Copy inputs to device
    CHECK_ACL(aclrtMemcpy(dev_a, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_b, matrixSize, matrix.data(), matrixSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Fft1DDFTTilingData tilingData;

    // set tiling data
    int isInverse = 1 - isForward;   // 正向传播则不需要inverse
    aclError ret = SetTilingData(tilingData, n, norm, isInverse, coreNum, batches);

    // Call host-side launcher (must be implemented to actually launch kernel)

    dft<<<coreNum, nullptr, stream>>>((__gm__ float *)dev_a, (__gm__ float *)dev_b, (__gm__ float *)dev_c, 
                                      (__gm__ uint8_t *)workspace_ptr, tilingData);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy device output back to host
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_c, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
