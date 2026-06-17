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
#include "fft1_d_n_tilingdata.h"
#include "fft1_d_n_kernel.h"

/* ========================================================================== */
/* fft_n: radix-2 large FFT (n >= 32768) implementation                       */
/* ========================================================================== */
#define FFT1_D_N_MAX_ITERATION 5
constexpr int RADIXVEC_SIZE_THREE = 3;
constexpr int N_DOING_24 = 16777216;
constexpr int N_DOING_27 = 134217728;
constexpr int N_DOING_15 = 32768;
constexpr int N_DOING_19 = 524288;
constexpr int N_DOING_25 = 33554432;
constexpr int LOGN_13 = 13;
constexpr int LOGN_14 = 14;
constexpr int LOGN_15 = 15;
constexpr int LOGN_16 = 16;
constexpr int LOGN_17 = 17;
constexpr int LOGN_19 = 19;
constexpr int LOGN_20 = 20;
constexpr int LOGN_21 = 21;
constexpr int LOGN_22 = 22;
constexpr int LOGN_23 = 23;
constexpr int LOGN_25 = 25;
constexpr int LOGN_26 = 26;
constexpr int LOGN_27 = 27;

static void InitRadix(uint32_t n, std::vector<uint32_t> &radixVec)
{
    int64_t minRadix = 8;
    int64_t maxRadix = 64;
    int64_t logN = static_cast<int64_t>(log((float)n) / log(2.0f));
    if (n >= pow(minRadix, 2) && n <= pow(maxRadix, 2)) {
        radixVec = {8, 8};
        for (int64_t idx = 0; idx < logN - 6; idx++) {
            radixVec[1 - idx % 2] *= 2;
        }
    } else if (n >= pow(minRadix, 3) && n <= pow(maxRadix, 3)) {
        radixVec = {8, 8, 8};
        for (int64_t idx = 0; idx < logN - 9; idx++) {
            radixVec[2 - idx % 3] *= 2;
        }
    } else if (n >= pow(minRadix, 4) && n <= pow(maxRadix, 4)) {
        radixVec = {8, 8, 8, 8};
        for (int64_t idx = 0; idx < logN - 12; idx++) {
            radixVec[3 - idx % 4] *= 2;
        }
    } else {
        radixVec = {8, 8, 8, 8, 8};
        for (int64_t idx = 0; idx < logN - 15; idx++) {
            radixVec[4 - idx % 5] *= 2;
        }
    }
    if (logN == LOGN_13) {
        radixVec = {8, 16, 64};
    } else if (logN == LOGN_14) {
        radixVec = {16, 16, 64};
    } else if (logN == LOGN_15) {
        radixVec = {32, 16, 64};
    } else if (logN == LOGN_16) {
        radixVec = {32, 32, 64};
    } else if (logN == LOGN_17) {
        radixVec = {64, 32, 64};
    } else if (logN == LOGN_19) {
        radixVec = {2, 64, 64, 64};
    } else if (logN == LOGN_20) {
        radixVec = {32, 16, 32, 64};
    } else if (logN == LOGN_21) {
        radixVec = {16, 32, 64, 64};
    } else if (logN == LOGN_22) {
        radixVec = {16, 64, 64, 64};
    } else if (logN == LOGN_23) {
        radixVec = {64, 32, 64, 64};
    } else if (logN == LOGN_25) {
        radixVec = {32, 32, 8, 64, 64};
    } else if (logN == LOGN_26) {
        radixVec = {64, 32, 16, 32, 64};
    } else if (logN == LOGN_27) {
        radixVec = {64, 64, 16, 32, 64};
    }
}
static void InitTilingArgs(uint32_t n, uint32_t iterCount, std::vector<uint32_t> &aicInputAddr,
                           std::vector<uint32_t> &aivOutputAddr, std::vector<uint32_t> &lessTCopy,
                           std::vector<uint32_t> &syncTilingNum)
{
    if (iterCount == RADIXVEC_SIZE_THREE) {
        aicInputAddr = {1, 1, 0};
        aivOutputAddr = {1, 0, 1};
    } else if (n <= N_DOING_24) {
        aicInputAddr = {1, 0, 1, 0};
        aivOutputAddr = {0, 1, 0, 1};
    } else if (n <= N_DOING_27) {
        aicInputAddr = {1, 0, 0, 1, 0};
        aivOutputAddr = {0, 0, 1, 0, 1};
    } else {
        aicInputAddr = {1, 1, 0, 1, 0};
        aivOutputAddr = {0, 0, 1, 0, 1};
    }
    lessTCopy = {1, 1, 1, 1, 1};
    syncTilingNum = {4, 4, 4, 4, 4};
    if (n <= N_DOING_15) {
        syncTilingNum = {4, 6, 6, 4, 4};
    }
    if (n == N_DOING_19) {
        aicInputAddr = {0, 1, 1, 0};
        aivOutputAddr = {1, 1, 0, 1};
        syncTilingNum = {2, 2, 2, 2, 2};
    }
    if (n == N_DOING_25) {
        lessTCopy = {0, 1, 1, 1, 1};
    }
}
static void ComputeRepeatBatchSize(uint32_t n, uint32_t batch, int32_t &repeatBatchSize)
{
    float batchDataSize = float(n) * 2 * 4 / 1024 / 1024;
    float l2CacheSize = 92.0f;
    repeatBatchSize = static_cast<int32_t>(floor((l2CacheSize - 10 - 2) / 2 / batchDataSize));
    repeatBatchSize = repeatBatchSize < 1 ? static_cast<int32_t>(batch) : repeatBatchSize;
}
static void SetFft1DNTilingData(Fft1DNTilingData &tiling, uint32_t fftN, uint32_t batchSize,
                                int32_t repeatBatchSize, const std::vector<uint32_t> &radixVec,
                                int isForward, uint32_t coreNum)
{
    (void)coreNum;
    tiling.nFFT = fftN;
    tiling.batchSize = batchSize;
    tiling.repeatBatchSize = repeatBatchSize;
    tiling.iterCount = static_cast<uint32_t>(radixVec.size());
    for (uint32_t i = 0; i < radixVec.size() && i < FFT1_D_N_MAX_ITERATION; i++) {
        tiling.radixVec[i] = radixVec[i];
    }
    std::vector<uint32_t> aicInputAddr, aivOutputAddr, lessTCopy, syncTilingNum;
    InitTilingArgs(fftN, tiling.iterCount, aicInputAddr, aivOutputAddr, lessTCopy, syncTilingNum);
    for (uint32_t i = 0; i < tiling.iterCount && i < FFT1_D_N_MAX_ITERATION; i++) {
        tiling.aicInputAddr[i] = aicInputAddr[i];
        tiling.aivOutputAddr[i] = aivOutputAddr[i];
        tiling.lessTCopy[i] = lessTCopy[i];
        tiling.syncTilingNum[i] = syncTilingNum[i];
    }
    tiling.fftDirection = (isForward != 0) ? -1 : 1;
}
static std::vector<float> InitWMatrixRadix2(uint32_t radix, bool forward, bool isLastIter)
{
    // Match SIP InitWMatrixCommon logic exactly
    // SIP has different layouts for last vs non-last iteration
    int32_t size = 2 * radix * 2 * radix;
    std::vector<float> matrix(size, 0.f);
    double K_2PI = 2.0 * 3.14159265358979323846;
    
    if (isLastIter) {
        // Last iteration (SIP: radixVec.size() > 1 && it == radixVec.size() - 1)
        // Layout: row i and row i+radix
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[i * 2 * radix + radix + j] = sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + j] = sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    } else {
        // Non-last iteration (SIP: else case)
        // Layout: row 2*i and row 2*i+1
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[2 * i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[2 * i * 2 * radix + radix + j] = sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + j] = sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    }
    return matrix;
}
static std::vector<float> InitTMatrix(uint32_t fftN, const std::vector<uint32_t>& radixVec, uint32_t iterIndex, bool forward)
{
    // Match SIP InitTMatrixCommon logic
    // T matrix only for iterations 0 to size-1 (exclude last iteration)
    double K_2PI = 2.0 * 3.14159265358979323846;
    
    uint32_t tempRow = radixVec[iterIndex] * 2;
    uint32_t tempCol = 1;
    for (uint32_t j = iterIndex + 1; j < radixVec.size(); j++) {
        tempCol *= radixVec[j];
    }
    
    uint32_t size = tempRow * tempCol;  // Note: SIP uses tempRow*tempCol not tempRow*tempCol*2
    std::vector<float> matrix(size, 0.f);
    
    // SIP formula: angle = -2*pi/(tempRow/2 * tempCol) * i * j
    // Note: tempRow/2 = radixVec[iterIndex]
    double angle_base = -1.0 * K_2PI / (tempRow / 2 * tempCol);
    
    for (uint32_t i = 0; i < tempRow / 2; ++i) {
        for (uint32_t j = 0; j < tempCol; ++j) {
            // SIP layout: [2*i*tempCol + j] for real, [(2*i+1)*tempCol + j] for imag
            double angle = angle_base * i * j;
            matrix[2 * i * tempCol + j] = static_cast<float>(cos(angle));
            matrix[(2 * i + 1) * tempCol + j] = static_cast<float>(sin(angle));
        }
    }
    return matrix;
}
static std::vector<int32_t> InitIndexTable(uint32_t fftN, const std::vector<uint32_t>& radixVec)
{
    // Match SIP index generation in FFTCoreN::InitInDevice()
    uint32_t iterCount = radixVec.size();
    
    // SIP tN calculation
    int64_t tN = 1;
    constexpr int64_t CALCUL_TWO = 2;
    if (iterCount > CALCUL_TWO) {
        for (uint32_t it = 0; it < (iterCount - CALCUL_TWO); ++it) {
            tN *= radixVec[it];
        }
    } else {
        tN = radixVec[0];
    }
    
    int64_t tM = 2 * radixVec.back();
    int64_t N0 = (tM < 128) ? 256 : 128;
    tN = (tN < N0) ? tN : N0;
    
    int64_t tilingNum = (tM / 2) * tN;
    std::vector<int32_t> index(tilingNum);
    
    // SIP index pattern
    for (int64_t i = 0; i < tilingNum / 2; i++) {
        index[2 * i] = i * 4;
        index[2 * i + 1] = (i + tilingNum / 2) * 4;
    }
    return index;
}

constexpr int32_t SCRATCH_SIZES = sizeof(float) * 128 * 512 * 20;
extern "C" aclError aclfftFft1DN(float *x, float *y, uint32_t n, int32_t norm,
                                uint32_t batches, int isForward, void *stream) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }
    std::vector<uint32_t> radixVec;
    InitRadix(n, radixVec);
    if (radixVec.empty()) {
        std::cerr << "fft_n: n must be power of 2, got " << n << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    int32_t repeatBatchSize = 0;
    ComputeRepeatBatchSize(n, batches, repeatBatchSize);
    const uint32_t inputSize = n * batches * sizeof(float) * 2;
    const uint32_t outputSize = inputSize;
    std::vector<float> wMatrixHost;
    for (size_t i = 0; i < radixVec.size(); i++) {
        uint32_t radix = radixVec[i];
        uint32_t stageSize = 2 * radix * 2 * radix;  // Match SIP size
        size_t wBias = wMatrixHost.size();
        wMatrixHost.resize(wBias + stageSize);
        bool isLastIter = (i == radixVec.size() - 1);
        std::vector<float> stageMatrix = InitWMatrixRadix2(radix, isForward != 0, isLastIter);
        for (uint32_t j = 0; j < stageSize; j++) {
            wMatrixHost[wBias + j] = stageMatrix[j];
        }
    }
    const uint32_t wMatrixSize = wMatrixHost.size() * sizeof(float);
    std::vector<float> tMatrixHost;
    // T matrix only for iterations 0 to size-1 (exclude last iteration)
    for (size_t i = 0; i < radixVec.size() - 1; i++) {
        std::vector<float> stageT = InitTMatrix(n, radixVec, static_cast<uint32_t>(i), isForward != 0);
        tMatrixHost.insert(tMatrixHost.end(), stageT.begin(), stageT.end());
    }
    const uint32_t tMatrixSize = tMatrixHost.size() * sizeof(float);
    std::vector<int32_t> indexHost = InitIndexTable(n, radixVec);
    const uint32_t indexSize = indexHost.size() * sizeof(int32_t);
    uint32_t totalWorkspaceSize = SCRATCH_SIZES * coreNum;

    void *dev_input = nullptr;
    void *dev_output = nullptr;
    void *dev_wMatrix = nullptr;
    void *dev_tMatrix = nullptr;
    void *dev_index = nullptr;
    void *dev_workspace = nullptr;
    void *dev_tiling = nullptr;
    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_wMatrix, wMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_tMatrix, tMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_index, indexSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_workspace, totalWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_tiling, sizeof(Fft1DNTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    std::unique_ptr<void, AclrtFreeDeleter> d_wMatrix_guard(dev_wMatrix);
    std::unique_ptr<void, AclrtFreeDeleter> d_tMatrix_guard(dev_tMatrix);
    std::unique_ptr<void, AclrtFreeDeleter> d_index_guard(dev_index);
    std::unique_ptr<void, AclrtFreeDeleter> d_workspace_guard(dev_workspace);
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);
    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_wMatrix, wMatrixSize, wMatrixHost.data(), wMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tMatrix, tMatrixSize, tMatrixHost.data(), tMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_index, indexSize, indexHost.data(), indexSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Fft1DNTilingData tilingData;
    SetFft1DNTilingData(tilingData, n, batches, repeatBatchSize, radixVec, isForward, coreNum);
    CHECK_ACL(aclrtMemcpy(dev_tiling, sizeof(Fft1DNTilingData), &tilingData, sizeof(Fft1DNTilingData), ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void **)&sync));
    FFT1DNKernel::fft_n<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)sync,
        (__gm__ float *)dev_input,
        (__gm__ float *)dev_wMatrix,
        (__gm__ float *)dev_tMatrix,
        (__gm__ int32_t *)dev_index,
        (__gm__ float *)dev_output,
        (__gm__ float *)dev_workspace,
        (__gm__ uint8_t *)dev_tiling
    );
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
