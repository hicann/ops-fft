/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT2_D_DD_TILINGDATA_H
#define FFT2_D_DD_TILINGDATA_H

#include <cstdint>

struct Fft2DTilingData {
    int32_t batchSize{0};
    int32_t fftX{0};
    int32_t fftY{0};
    int32_t batchNumsPerLoop{0};
};

#endif // FFT2_D_DD_TILINGDATA_H
