/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RFFT1_D_SIZE_UTILS_H
#define RFFT1_D_SIZE_UTILS_H

#include <cstddef>
#include <cstdint>

namespace rfft1_d {

constexpr size_t InputSize(uint32_t n, uint32_t batches)
{
    return static_cast<size_t>(n) * static_cast<size_t>(batches) * sizeof(float);
}

constexpr size_t OutputSize(uint32_t n, uint32_t batches)
{
    const size_t outputElements = static_cast<size_t>(n) / 2 + 1;
    return outputElements * 2 * static_cast<size_t>(batches) * sizeof(float);
}

constexpr size_t FftWorkspaceSize(uint32_t n, uint32_t batches)
{
    const size_t fftPointCount = (n % 2 != 0) ? static_cast<size_t>(n) : static_cast<size_t>(n) / 2;
    return 2 * fftPointCount * static_cast<size_t>(batches) * sizeof(float) * 2;
}

} // namespace rfft1_d

#endif // RFFT1_D_SIZE_UTILS_H
