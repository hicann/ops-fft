#-----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#-----------------------------------------------------------------------------

##############################################################################
# tests/CMakeLists.txt
#
# 自动扫描所有算子测试并生成 all_tests.cpp
##############################################################################

# 只在 BUILD_TESTING 开启时执行
if(BUILD_TESTING)

    # 扫描所有测试头文件和源文件（使用通配符）
    set(TEST_HEADERS "")
    set(TEST_SOURCES "")

    foreach(OP ${BUILD_OPERATORS})
        set(OP_TEST_DIR "${CMAKE_SOURCE_DIR}/src/${OP}/tests")

        # 检查测试目录是否存在
        if(EXISTS ${OP_TEST_DIR})
            # 扫描单元测试头文件 (*_test.h)
            file(GLOB_RECURSE OP_TEST_HEADERS
                "${OP_TEST_DIR}/*_test.h"
            )
            list(APPEND TEST_HEADERS ${OP_TEST_HEADERS})

            # 扫描单元测试源文件 (*_test.cpp)
            file(GLOB_RECURSE OP_TEST_SOURCES
                "${OP_TEST_DIR}/*_test.cpp"
            )
            list(APPEND TEST_SOURCES ${OP_TEST_SOURCES})
        endif()
    endforeach()

    # 去重并排序
    list(REMOVE_DUPLICATES TEST_HEADERS)
    list(REMOVE_DUPLICATES TEST_SOURCES)
    list(SORT TEST_HEADERS)
    list(SORT TEST_SOURCES)

    message(STATUS "找到测试头文件: ${TEST_HEADERS}")
    message(STATUS "找到测试源文件: ${TEST_SOURCES}")

    # 生成 include 语句列表（转换为相对于 src 目录的路径）
    set(TEST_INCLUDES "")
    foreach(test_header ${TEST_HEADERS})
        # 转换为相对路径
        file(RELATIVE_PATH rel_header ${CMAKE_SOURCE_DIR}/src ${test_header})
        set(TEST_INCLUDES "${TEST_INCLUDES}#include \"${rel_header}\"\n")
    endforeach()

    # 配置生成 all_tests.cpp
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/all_tests.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        @ONLY
    )

    # 创建测试可执行文件
    add_executable(all_ops_test
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
        ${TEST_SOURCES}
    )

    # 设置包含目录
    target_include_directories(all_ops_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include/
        ${ASCEND_HOME_PATH}/pkg_inc/op_common/
        ${ASCEND_HOME_PATH}/pkg_inc/base/
        ${ASCEND_HOME_PATH}/pkg_inc/
    )

    # 链接库
    target_link_libraries(all_ops_test PRIVATE
        ${OPS_FFT}  # 链接到 CMake 目标
        ${ASCEND_HOME_PATH}/lib64/libascendcl.so
        ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    )

    message(STATUS "已配置测试目标: all_ops_test")

endif()
