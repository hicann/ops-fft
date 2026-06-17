/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_STRIDE_TILINGDATA_H
#define FFT1_D_STRIDE_TILINGDATA_H

#include <cstdint>

struct Fft1DStrideTilingData {
    uint32_t batchSize{0};
    uint32_t fftN{0};
    uint32_t fftStride{0};
    int32_t fftDirection{-1};
    uint32_t iterCount{0};
    uint32_t radixVec[5];
    uint32_t s0{0};
};

#endif