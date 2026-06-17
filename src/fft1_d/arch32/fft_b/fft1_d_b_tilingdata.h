/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_B_TILINGDATA_H
#define FFT1_D_B_TILINGDATA_H

#include <cstdint>

constexpr uint32_t FFT_B_MAX_ITERATION = 10;

struct Fft1DBTilingData {
    uint32_t nFFT{0};
    uint32_t batchSize{0};
    int32_t fftDirection{-1};
    uint32_t iterCount{0};
    uint32_t radixVec[FFT_B_MAX_ITERATION];
};

#endif // FFT1_D_B_TILINGDATA_H
