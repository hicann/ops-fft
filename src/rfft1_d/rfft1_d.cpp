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
 * \file rfft1_d.cpp
 * \brief
 */

#include <cmath>
#include <numeric>
#include <vector>

#include "rfft1_d.h"
#include "arch35/rfft1_d_tilingdata.h"
#include "arch35/rfft1_d_fast.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"

static const uint32_t LAST_FACTOR = 64;
static const uint32_t COMPLEX_PART = 2;
static const uint32_t INIT_VALUE = 1;
static const uint32_t MATMUL_SIZE_MULTIPLIER = 24;
static const uint32_t SIZE_PER_BATCH_MULTIPLIER = 4;
static const uint32_t BYTES_ALIGN = 8;
static const uint32_t ROW_PAD = 16;
static const uint32_t COL_PAD = 8;
static const uint32_t TWIDDLE_MATRICES_AMOUNT = 2;
static const uint32_t DFT_BORDER_VALUE = 4096;
static const uint32_t DFT_OFFSETS_COUNT = 3;
static const uint32_t RFFT_SYMMETRY_DIVISOR = 2;
static const uint32_t SYS_WORKSPACE_SIZE_16MB = 16 * 1024 * 1024;
// NORM_VALUES
static const uint32_t BACKWARD = 1;
static const uint32_t FORWARD = 2;

// ACL 错误检查宏
#define CHECK_ACL(call)                                              \
    do {                                                             \
        aclError err = (call);                                       \
        if (err != ACL_SUCCESS) {                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ \
                    << ":" << __LINE__ << std::endl;              \
            return 1;                                                \
        }                                                            \
    } while (0)

// 自定义删除器，安全处理空指针
struct AclrtFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

static void CalcColleyTukeyFactors(uint32_t factors[], std::vector<uint32_t> availableFactors, const uint32_t n)
{
    std::vector<uint32_t> factorsTmp;
    int curFactorIndex = availableFactors.size() - 1;
    uint32_t tmpN = n;

    if (tmpN == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else if (tmpN > DFT_BORDER_VALUE) {
        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];

            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmp.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }

        while (factorsTmp.size() < MAX_FACTORS_LEN) {
            factorsTmp.emplace_back(1);
        }
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = factorsTmp[i];
        }
    }
}

static void CalcBluesteinFactors(uint32_t factors[], std::vector<uint32_t> availableFactors, const uint32_t pow2)
{
    std::vector<uint32_t> factorsTmpBluestein;
    if (pow2 == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else {
        uint32_t tmpN = pow2;
        int curFactorIndex = availableFactors.size() - 1;

        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];
            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmpBluestein.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }

        while (factorsTmpBluestein.size() < MAX_FACTORS_LEN) {
            factorsTmpBluestein.emplace_back(1);
        }

        for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
            factors[i] = factorsTmpBluestein[i];
        }
    }
}

static void CalcDftSizes(Rfft1DTilingData &tiling, const uint32_t factors[], const bool isBluestein, const uint32_t n)
{
    uint32_t dftRealOverallSize = 0;
    uint32_t dftImagOverallSize = 0;
    uint32_t twiddleOverallSize = 0;

    uint32_t dftRealOffsets[DFT_OFFSETS_COUNT] = {0, 1, 1};
    uint32_t dftImagOffsets[DFT_OFFSETS_COUNT] = {0, 1, 1};
    uint32_t twiddleOffsets[DFT_OFFSETS_COUNT] = {0, 0, 1};

    size_t prevFactors = 1;

    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        size_t curFactor = factors[curIndex];
        size_t rowsNum = curFactor * (1 + static_cast<size_t>(curIndex == 0 && isBluestein));
        size_t colsNum = curFactor * (COMPLEX_PART - static_cast<size_t>(curIndex != 0));
        size_t dftCurSize = rowsNum * colsNum;
        size_t twiddleCurSize = rowsNum * prevFactors * COMPLEX_PART;

        if (curIndex != 0) {
            dftRealOffsets[curIndex] = dftRealOverallSize;
            dftImagOffsets[curIndex] = dftImagOverallSize;
            if (curIndex != 1) {
                twiddleOffsets[curIndex] = twiddleOverallSize;
            }
        }

        dftRealOverallSize += dftCurSize;
        if (curIndex != 0) {
            dftImagOverallSize += dftCurSize;
            twiddleOverallSize += twiddleCurSize;
        }

        prevFactors *= curFactor;
    }
    uint32_t fftPadRow = n + (COL_PAD - n % COL_PAD);
    uint32_t fftPadCol = ((n % ROW_PAD) != 0) ? n + (ROW_PAD - n % ROW_PAD) : n;
    dftRealOverallSize = n <= DFT_BORDER_VALUE ? fftPadRow * fftPadCol : dftRealOverallSize;

    tiling.dftRealOverallSize = dftRealOverallSize;
    tiling.dftImagOverallSize = dftImagOverallSize;
    tiling.twiddleOverallSize = twiddleOverallSize;
    tiling.fftMatrOverallSize = dftRealOverallSize + dftImagOverallSize + TWIDDLE_MATRICES_AMOUNT * twiddleOverallSize;

    for (size_t i = 0; i < DFT_OFFSETS_COUNT; i++) {
        tiling.dftRealOffsets[i] = dftRealOffsets[i];
        tiling.dftImagOffsets[i] = dftImagOffsets[i];
        tiling.twiddleOffsets[i] = twiddleOffsets[i];
    }
}

static int SetTilingData(Rfft1DTilingData &tiling, uint32_t n, int32_t norm, uint32_t coreNum, uint32_t batches)
{

    uint32_t factors[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t prevRadices[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t nextRadices[MAX_FACTORS_LEN] = {n / factors[0], 1, 1};
    uint8_t prevRadicesAlign[MAX_FACTORS_LEN] = {0, 1, 1};

    std::vector<uint32_t> availableFactors(LAST_FACTOR - 1);
    std::iota(availableFactors.begin(), availableFactors.end(), COMPLEX_PART);

    CalcColleyTukeyFactors(factors, availableFactors, n);

    const bool isBluestein = (n % LAST_FACTOR != 0) || (factors[0] * factors[1] * factors[2] != n);
    const uint32_t pow2 = COMPLEX_PART * uint32_t(std::pow(2, std::ceil(std::log2(double(n)))));
    const uint32_t lengthPad = isBluestein ? pow2 : n;

    if (isBluestein) {
        CalcBluesteinFactors(factors, availableFactors, pow2);
    }

    for (uint8_t i = 1; i < MAX_FACTORS_LEN; ++i) {
        prevRadices[i] = prevRadices[i - 1] * factors[i - 1];
        nextRadices[i] = nextRadices[i - 1] / factors[i];
        prevRadicesAlign[i] = (COMPLEX_PART * prevRadices[i] % BYTES_ALIGN) == 0;
    }

    auto roundUpBlock = [](const uint32_t& src, const uint32_t blockLen) {
        return src != 0 ? src + (blockLen - src % blockLen) % blockLen : blockLen;
    };

    const uint32_t tailSize = COMPLEX_PART * (((n / RFFT_SYMMETRY_DIVISOR) + 1) - (factors[2] / COMPLEX_PART) * (n / factors[2]));
    const uint32_t tmpLenPerBatch = 3 * roundUpBlock(
                                            COMPLEX_PART * (isBluestein ? lengthPad : n) + factors[2] * tailSize + 1,
                                            BYTES_ALIGN * SIZE_PER_BATCH_MULTIPLIER);

    tiling.length = n;
    tiling.isBluestein = isBluestein;
    tiling.lengthPad = lengthPad;
    tiling.outLength = (n / RFFT_SYMMETRY_DIVISOR) + 1;
    tiling.tailSize = tailSize;
    tiling.batchesPerCore = batches / coreNum;
    tiling.leftOverBatches = batches % coreNum;
    tiling.tmpLenPerBatch = tmpLenPerBatch;
    tiling.tmpSizePerBatch = tmpLenPerBatch * SIZE_PER_BATCH_MULTIPLIER;
    tiling.normal = norm;
    tiling.matmulTmpsLen = MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch;
    tiling.matmulTmpsSize = MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch * sizeof(float);

    for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
        tiling.factors[i] = factors[i];
        tiling.prevRadices[i] = prevRadices[i];
        tiling.nextRadices[i] = nextRadices[i];
        tiling.prevRadicesAlign[i] = prevRadicesAlign[i];
    }
    CalcDftSizes(tiling, factors, isBluestein, n);

    return ACL_SUCCESS;
}

static std::vector<float> Rfft1DDftGen(int64_t fftLength, int64_t norm)
{
    std::vector<float> dftNz;
    float normParam = 1.;

    if (norm != BACKWARD) {
        normParam = norm == FORWARD ? normParam / float(fftLength) : normParam / sqrt(float(fftLength));
    }

    size_t fftLenPad = fftLength + (NZ_BORDER - fftLength % NZ_BORDER);
    size_t fftLenPadRow = fftLength + (NZ_BLOCK - fftLength % NZ_BLOCK);

    size_t fftOut = fftLength / RFFT_SYMMETRY_DIVISOR + 1;
    dftNz.reserve(fftLenPadRow * fftLenPadRow);

    std::vector<float> dftNd(fftLenPad * fftLenPadRow, 0);
    size_t k = 0;

    if ((fftLength % NZ_BLOCK != 0) && (fftLength % NZ_BLOCK <= NZ_BORDER)) {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                float param = -2. * M_PI * i * j / float(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
        }
        return dftNd;
    } else {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                float param = -2. * M_PI * i * j / float(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
            for (size_t j = 0; j + COMPLEX_PART * fftOut < fftLenPad; ++j) {
                dftNd[k] = INIT_VALUE;
                ++k;
            }
        }
        for (size_t n = 0; n < fftLenPad / NZ_BORDER; ++n) {
            for (size_t i = 0; i < fftLenPadRow; ++i) {
                for (size_t blockIdx = 0; blockIdx < NZ_BORDER; ++blockIdx) {
                    dftNz.emplace_back(dftNd[i * fftLenPad + NZ_BORDER * n + blockIdx]);
                }
            }
        }
        return dftNz;
    }
}

extern "C" __global__ __aicore__ void rfft1_d(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, Rfft1DTilingData tilingData)
{
    if (tilingData.length <= DFT_BORDER_VALUE) // FastDFT for length <= 4096
    {
        uint32_t* factorsP = const_cast<uint32_t*>(tilingData.factors);

        KernelRfftFastDFT op(tilingData.length, tilingData.batchesPerCore, tilingData.leftOverBatches, 
                            tilingData.normal, tilingData.dftRealOverallSize, factorsP);

        uint32_t modeLength = COMPLEX_PART * (tilingData.length / RFFT_SYMMETRY_DIVISOR + 1);
        auto t1 = PrepareTiling((op.batches + op.advancedBatches) / GetBlockNum(), op.modeLength, tilingData.length);
        REGIST_MATMUL_OBJ(&op.pipe, workspace, op.matmulObj, (void*)&t1, op.matmulObjNZ, (void*)&t1);

        op.Init(x, dft, y);
        op.Process();
    }
    return;
}

extern "C" aclError aclfftRfft1D(float *x, float *y, uint32_t n, int32_t norm, uint32_t batches, void *stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();

    auto dft = Rfft1DDftGen(n, norm);

    uint32_t sysWorkspaceSize = SYS_WORKSPACE_SIZE_16MB;

    const uint32_t inputSize = n * sizeof(float);
    const uint32_t dftSize = dft.size() * sizeof(float);
    const uint32_t outputSize = ((n / RFFT_SYMMETRY_DIVISOR) + 1) * COMPLEX_PART * sizeof(float);

    void *dev_x = nullptr;
    void *dev_dft = nullptr;
    void *dev_y = nullptr;
    void *workspace_ptr = nullptr;

    // Allocate device memory for inputs and output
    CHECK_ACL(aclrtMalloc(&dev_x, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_dft, dftSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dev_y, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspace_ptr, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // 智能指针指定删除器，在unique_ptr析构时自动调用指定的函数释放资源
    std::unique_ptr<void, AclrtFreeDeleter> d_x_guard(dev_x);
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft);
    std::unique_ptr<void, AclrtFreeDeleter> d_y_guard(dev_y);

    // Copy inputs to device
    CHECK_ACL(aclrtMemcpy(dev_x, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft, dftSize, dft.data(), dftSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Rfft1DTilingData tilingData;

    // set tiling data
    aclError ret = SetTilingData(tilingData, n, norm, coreNum, batches);

    // Call host-side launcher (must be implemented to actually launch kernel)
    rfft1_d<<<coreNum, nullptr, stream>>>((__gm__ uint8_t *)dev_x, (__gm__ uint8_t *)dev_dft, (__gm__ uint8_t *)dev_y, (__gm__ uint8_t *)workspace_ptr, tilingData);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy device output back to host
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_y, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}

