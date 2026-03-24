/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_fft.h"

/**
 * @brief 获取错误描述字符串
 *
 * @param result 错误码
 * @return 错误描述字符串
 */
const char* aclfftGetErrorString(aclfftResult result) {
    switch (result) {
        case ACLFFT_SUCCESS:
            return "Success";
        case ACLFFT_INVALID_PLAN:
            return "Invalid plan handle";
        case ACLFFT_ALLOC_FAILED:
            return "Memory allocation failed";
        case ACLFFT_INVALID_TYPE:
            return "Invalid data type";
        case ACLFFT_INVALID_VALUE:
            return "Invalid parameter value";
        case ACLFFT_INTERNAL_ERROR:
            return "Internal error";
        case ACLFFT_EXEC_FAILED:
            return "Execution failed";
        case ACLFFT_SETUP_FAILED:
            return "Setup failed";
        case ACLFFT_INVALID_SIZE:
            return "Invalid size parameter";
        case ACLFFT_UNALIGNED_DATA:
            return "Data not aligned";
        case ACLFFT_INCOMPLETE_PARAMETER_LIST:
            return "Incomplete parameter list";
        case ACLFFT_INVALID_DEVICE:
            return "Invalid device";
        case ACLFFT_PARSE_ERROR:
            return "Parse error";
        case ACLFFT_NO_WORKSPACE:
            return "No workspace available";
        case ACLFFT_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case ACLFFT_NOT_SUPPORTED:
            return "Feature not supported";
        default:
            return "Unknown error";
    }
}
