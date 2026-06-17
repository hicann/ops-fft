/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fft1_d_n_tilingdata.h
 * \brief fft1_d_n (radix-2 large FFT) tiling data structure definition
 */

#ifndef FFT1_D_N_TILINGDATA_H
#define FFT1_D_N_TILINGDATA_H

#include <cstdint>

constexpr uint32_t FFT1_D_N_MAX_ITERATION = 5;

struct Fft1DNTilingData {
    uint32_t nFFT{0};                       // FFT size
    uint32_t batchSize{0};                  // Batch size
    int32_t repeatBatchSize{0};             // Repeat batch size for chunking
    uint32_t iterCount{0};                  // Number of radix iterations
    uint32_t radixVec[FFT1_D_N_MAX_ITERATION]; // Radix for each iteration
    uint32_t aicInputAddr[FFT1_D_N_MAX_ITERATION];   // AIC workspace input addresses
    uint32_t aivOutputAddr[FFT1_D_N_MAX_ITERATION];  // AIV workspace output addresses
    uint32_t lessTCopy[FFT1_D_N_MAX_ITERATION];      // LessTCopy flag per iteration
    uint32_t syncTilingNum[FFT1_D_N_MAX_ITERATION];  // Sync tiling number per iteration
    int32_t fftDirection{-1};               // -1=forward, 1=inverse
};

#endif // FFT1_D_N_TILINGDATA_H
