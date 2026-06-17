/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../rfft1_d.h"
#include "common/fft_common_core.h"
#include "rfft1_d_dft_tilingdata.h"
#include "rfft1_d_dft_kernel.h"

static constexpr int MUL_DIV_TWO = 2;
static constexpr int64_t DFT_R2C_WORKSPACE_SIZE = 32;
static constexpr int CUBE_DATA_COUNT_PER_LOOP_SMALL = 128;
static constexpr int CUBE_DATA_COUNT_PER_LOOP_LARGE = 64;
static constexpr int THRESHOLD_K = 32768;

static std::vector<float> GenerateDftMatrixR2C(int64_t fftN)
{
    int64_t M = fftN;
    int64_t N = (fftN / 2 + 1) * MUL_DIV_TWO;
    std::vector<float> dftMatrix(M * N, 0.0f);

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < fftN / 2 + 1; j++) {
            double angle = -2.0 * K_PI * i * j / (fftN ? fftN : 1);
            dftMatrix[i * N + 2 * j] = static_cast<float>(cos(angle));
            dftMatrix[i * N + 2 * j + 1] = static_cast<float>(sin(angle));
        }
    }

    return dftMatrix;
}

extern "C" aclError aclfftRfft1DDft(float *x, float *y, uint32_t n, int32_t norm,
                                     uint32_t batches, int isForward, void *stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int64_t fftN = static_cast<int64_t>(n);
    int32_t M = batches;
    int32_t N = (fftN / 2 + 1) * MUL_DIV_TWO;
    int32_t K = fftN;

    Rfft1DDftTilingData tilingData;
    tilingData.batchSize = 1;
    tilingData.m = M;
    tilingData.n = N;
    tilingData.k = K;
    tilingData.transA = 0;
    tilingData.transB = 0;
    tilingData.m0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingData.n0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingData.k0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    if (K > THRESHOLD_K) {
        tilingData.m0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingData.n0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingData.k0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
    }

    std::vector<float> dftMatrix = GenerateDftMatrixR2C(fftN);

    uint32_t inputSize = batches * fftN * sizeof(float);
    uint32_t dftMatrixSize = dftMatrix.size() * sizeof(float);
    uint32_t outputSize = batches * (fftN / 2 + 1) * sizeof(float) * 2;
    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
    uint32_t tilingSize = sizeof(Rfft1DDftTilingData);

    void *dev_input = nullptr;
    void *dev_dft = nullptr;
    void *dev_output = nullptr;
    void *dev_workspace = nullptr;
    void *dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_dft, dftMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_workspace, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft);
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    std::unique_ptr<void, AclrtFreeDeleter> d_ws_guard(dev_workspace);
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft, dftMatrixSize, dftMatrix.data(), dftMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Rfft1DDftKernel::dft_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ float *)dev_input,
        (__gm__ float *)dev_dft,
        (__gm__ float *)dev_output,
        (__gm__ uint8_t *)dev_workspace,
        (__gm__ uint8_t *)dev_tiling
    );

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
