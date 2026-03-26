/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"

static const uint32_t COMPLEX_PART = 2;

/**
 * @brief 计算数据大小和元素大小
 */
static void calculate_data_sizes(aclfftHandle_t* plan) {
    // 计算元素大小
    switch (plan->type) {
        case ACLFFT_C2C:
        case ACLFFT_R2C:
        case ACLFFT_C2R:
            plan->element_size = sizeof(float);  // 单精度
            break;
        case ACLFFT_Z2Z:
        case ACLFFT_D2Z:
        case ACLFFT_Z2D:
            plan->element_size = sizeof(double);  // 双精度
            break;
        default:
            plan->element_size = sizeof(float);
            break;
    }

    // 计算输入输出大小
    size_t total_elements = 1;
    for (int i = 0; i < plan->rank; ++i) {
        total_elements *= plan->lengths[i];
    }
    total_elements *= plan->batch;

    // 根据类型调整大小
    if (plan->type == ACLFFT_R2C || plan->type == ACLFFT_D2Z) {
        // 实数到复数：输入是实数，输出是复数
        plan->input_size = total_elements * plan->element_size;
        plan->output_size = total_elements * COMPLEX_PART * plan->element_size;  // 复数需要 2 倍空间
    } else if (plan->type == ACLFFT_C2R || plan->type == ACLFFT_Z2D) {
        // 复数到实数：输入是复数，输出是实数
        plan->input_size = total_elements * COMPLEX_PART * plan->element_size;
        plan->output_size = total_elements * plan->element_size;
    } else {
        // C2C 或 Z2Z：输入输出都是复数
        plan->input_size = total_elements * COMPLEX_PART * plan->element_size;
        plan->output_size = total_elements * COMPLEX_PART * plan->element_size;
    }
}

/**
 * @brief 初始化 1D FFT Plan
 *
 * @param plan Plan 句柄
 * @param nx X 维度大小
 * @param type 变换类型
 * @param batch 批量大小
 * @param workSize 输出参数：返回所需工作空间大小（bytes），可为 nullptr
 * @return aclfftResult 错误码
 */
aclfftResult aclfftMakePlan1d(aclfftHandle plan, int nx, aclfftType type, int batch,
                              size_t* workSize) {
    aclfftHandle_t* impl = plan;

    // 参数验证
    ACLFFT_CHECK_PLAN_NOT_INITIALIZED(impl);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(type >= ACLFFT_C2C && type <= ACLFFT_Z2D, ACLFFT_INVALID_TYPE);

    // 设置基本参数
    impl->rank = 1;
    impl->lengths[0] = nx;
    impl->batch = batch;
    impl->type = type;

    // 计算数据大小
    calculate_data_sizes(impl);

    // 返回工作空间大小（暂时设为 0）
    if (workSize != nullptr) {
        *workSize = 0;
    }

    // 标记为已初始化
    impl->is_initialized = true;

    return ACLFFT_SUCCESS;
}

/**
 * @brief 初始化 2D FFT Plan
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftMakePlan2d(aclfftHandle plan, int nx, int ny, aclfftType type,
                              size_t* workSize) {
    // 暂不实现
    return ACLFFT_NOT_IMPLEMENTED;
}

/**
 * @brief 初始化 3D FFT Plan
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftMakePlan3d(aclfftHandle plan, int nx, int ny, int nz, aclfftType type,
                              size_t* workSize) {
    // 暂不实现
    return ACLFFT_NOT_IMPLEMENTED;
}
