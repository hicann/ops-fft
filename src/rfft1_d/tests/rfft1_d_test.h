/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file rfft1_d_api_integration_test.h
 * @brief R2C FFT API 集成测试函数声明
 */

#pragma once

#include "test_common.h"

namespace Rfft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
