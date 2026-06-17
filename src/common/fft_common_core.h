/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_COMMON_CORE_H
#define FFT_COMMON_CORE_H

#include <cmath>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"

constexpr uint32_t DFT_SIZE_MULTIPLIER = 2;
static const double INIT_VALUE = 0.;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

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

#endif