/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../irfft1_d.h"
#include "common/fft_common_core.h"
#include "irfft1_d_dft_tilingdata.h"
#include "irfft1_d_dft_kernel.h"

static int32_t FindRadixHost(int64_t n)
{
    if (n % 2 == 0) return 2;
    if (n % 3 == 0) return 3;
    if (n % 5 == 0) return 5;
    if (n % 7 == 0) return 7;
    return 0;
}

static std::vector<float> GenerateDftMatrixHost(int32_t radix, int32_t isInverse)
{
    int64_t rows = radix;
    int64_t cols = radix;
    std::vector<float> dft(2 * rows * cols, 0.0f);
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    constexpr double K_2PI = 2.0 * K_PI;

    for (int64_t u = 0; u < rows; u++) {
        for (int64_t v = 0; v < cols; v++) {
            double angle = sign * K_2PI * u * v / (radix ? radix : 1);
            dft[(2 * u) * radix + v] = static_cast<float>(cos(angle));
            dft[(2 * u + 1) * radix + v] = static_cast<float>(sin(angle));
        }
    }

    return dft;
}

static std::vector<float> GenerateTwiddleFactorsHost(int32_t radix, int64_t M, int32_t isInverse)
{
    int64_t totalElements = 2 * radix * M;
    std::vector<float> tw(totalElements, 0.0f);
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    constexpr double K_2PI = 2.0 * K_PI;
    int64_t tempN = static_cast<int64_t>(radix) * M;

    for (int64_t u = 0; u < radix; u++) {
        for (int64_t n2 = 0; n2 < M; n2++) {
            double angle = sign * K_2PI * u * n2 / tempN;
            tw[(2 * u) * M + n2] = static_cast<float>(cos(angle));
            tw[(2 * u + 1) * M + n2] = static_cast<float>(sin(angle));
        }
    }

    return tw;
}

static int SetTilingData(Irfft1DDftTilingData &tiling, int64_t fftN, int32_t isInverse,
                          uint32_t coreNum, uint32_t batches)
{
    tiling.batchSize = batches;
    tiling.fftN = fftN;
    tiling.isInverse = isInverse;

    int64_t tempN = fftN;
    int64_t tempBatch = batches;
    int64_t dftOff = 0;
    int64_t twOff = 0;
    int32_t stageCount = 0;

    while (tempN > 1 && stageCount < MAX_FFT_STAGES_TILING) {
        int32_t radix = FindRadixHost(tempN);
        if (radix == 0) {
            std::cerr << "irfft1_d_fft: cannot factorize N=" << tempN << std::endl;
            return 1;
        }

        tiling.radix_arr[stageCount] = radix;
        tiling.M_arr[stageCount] = tempN / radix;
        tiling.currentBatch_arr[stageCount] = tempBatch;
        tiling.dft_offset_arr[stageCount] = dftOff;
        tiling.tw_offset_arr[stageCount] = twOff;

        tempBatch *= radix;
        dftOff += 2 * radix * radix;
        twOff += 2 * radix * tiling.M_arr[stageCount];
        tempN = tiling.M_arr[stageCount];
        stageCount++;
    }

    tiling.radixListLen = stageCount;

    int64_t singleBufferSize = batches * fftN * 2 * sizeof(float);
    int64_t workspaceSize = 2 * singleBufferSize;

    tiling.workspaceOffsets[0] = 0;
    tiling.workspaceOffsets[1] = singleBufferSize;
    tiling.workspaceOffsets[2] = 0;
    tiling.workspaceOffsets[3] = 0;
    tiling.workspaceOffsets[4] = 0;

    return 0;
}

extern "C" aclError aclfftIrfft1DDft(float *x, float *y, uint32_t n, int32_t norm,
                                         uint32_t batches, int isForward, void *stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int32_t isInverse = 1 - isForward;
    int64_t fftN = static_cast<int64_t>(n);

    Irfft1DDftTilingData tilingData;
    if (SetTilingData(tilingData, fftN, isInverse, coreNum, batches) != 0) {
        return 1;
    }

    std::vector<float> allDftMatrices;
    std::vector<float> allTwiddleFactors;
    int64_t tempN = fftN;

    for (int32_t s = 0; s < tilingData.radixListLen; s++) {
        int32_t radix = tilingData.radix_arr[s];
        int64_t M = tilingData.M_arr[s];

        auto dftMat = GenerateDftMatrixHost(radix, isInverse);
        auto twMat = GenerateTwiddleFactorsHost(radix, M, isInverse);

        allDftMatrices.insert(allDftMatrices.end(), dftMat.begin(), dftMat.end());
        allTwiddleFactors.insert(allTwiddleFactors.end(), twMat.begin(), twMat.end());

        tempN = M;
    }

    const uint32_t inputSize = batches * (n / 2 + 1) * sizeof(float) * 2;
    const uint32_t outputSize = batches * n * sizeof(float);
    const uint32_t dftMatrixSize = allDftMatrices.size() * sizeof(float);
    const uint32_t twSize = allTwiddleFactors.size() * sizeof(float);
    const uint32_t radixListSize = tilingData.radixListLen * sizeof(float);
    const uint32_t workspaceSize = 2 * batches * n * sizeof(float) * 2;
    const uint32_t tilingSize = sizeof(Irfft1DDftTilingData);

    void *dev_input = nullptr;
    void *dev_output = nullptr;
    void *dev_dft_matrix = nullptr;
    void *dev_tw_matrix = nullptr;
    void *dev_radix_list = nullptr;
    void *dev_workspace = nullptr;
    void *dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_dft_matrix, dftMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_tw_matrix, twSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_radix_list, radixListSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft_matrix);
    std::unique_ptr<void, AclrtFreeDeleter> d_tw_guard(dev_tw_matrix);
    std::unique_ptr<void, AclrtFreeDeleter> d_radix_guard(dev_radix_list);
    std::unique_ptr<void, AclrtFreeDeleter> d_ws_guard(dev_workspace);
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft_matrix, dftMatrixSize, allDftMatrices.data(), dftMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tw_matrix, twSize, allTwiddleFactors.data(), twSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<float> radixListHost(tilingData.radixListLen);
    for (int32_t s = 0; s < tilingData.radixListLen; s++) {
        radixListHost[s] = static_cast<float>(tilingData.radix_arr[s]);
    }
    CHECK_ACL(aclrtMemcpy(dev_radix_list, radixListSize, radixListHost.data(), radixListSize, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    fft_c2r_multi_core<<<coreNum, nullptr, stream>>>(
        (__gm__ float *)dev_input,
        (__gm__ float *)dev_dft_matrix,
        (__gm__ float *)dev_tw_matrix,
        (__gm__ float *)dev_radix_list,
        (__gm__ float *)dev_output,
        (__gm__ float *)dev_workspace,
        (__gm__ uint8_t *)dev_tiling
    );

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
