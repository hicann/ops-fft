/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file rfft1_d_test.h
 * @brief Rfft1D算子测试函数声明
 */

#pragma once

#include "test_common.h"

namespace DftR2CTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
