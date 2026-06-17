/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT2_D_H
#define FFT2_D_H

#include <cstdint>
#include "acl/acl.h"

#if defined(_WIN32) || defined(__CYGWIN__)
    #define ACLFFT_API __declspec(dllexport)
#else
    #define ACLFFT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLFFT_API aclError aclfftFft2DDd(float *x, float *y, uint32_t fftX, uint32_t fftY,
                                   uint32_t batches, int isForward, void *stream);

#ifdef __cplusplus
}
#endif

#endif // FFT2_D_H
