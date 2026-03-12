##############################################################################
# src/CMakeLists.txt
#
# 功能:
#   直接创建 ops_fft 动态库
#   各算子通过 register_operator() 函数注册源文件
##############################################################################

# 引入算子注册函数
include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)

##############################################################################
# 创建 ops_fft 动态库
##############################################################################

# 创建一个占位源文件（用于满足 CMake 的最低要求）
file(WRITE ${CMAKE_BINARY_DIR}/ops_fft_stub.cpp "// Stub file for ops_fft library")

# 创建动态库目标（使用占位源文件，后续通过 register_operator 添加实际源文件）
add_library(${OPS_FFT} SHARED ${CMAKE_BINARY_DIR}/ops_fft_stub.cpp)

# 设置库的版本属性
set_target_properties(${OPS_FFT} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "cann_ops_fft"  # 生成文件名：libcann_ops_fft.so
)

# 设置公共包含目录
target_include_directories(${OPS_FFT} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ASCEND_HOME_PATH}/include               # acl.h 等核心头文件
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/    # ASCEND 公共操作（kernel_operator.h 等）
    ${ASCEND_HOME_PATH}/pkg_inc/base/         # ASCEND 基础类型
    ${ASCEND_HOME_PATH}/pkg_inc/              # ASCEND 头文件
)

# 设置编译选项
target_compile_options(${OPS_FFT} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>
)

# 链接 CANN tiling 平台库 (提供 GetCoreNumAiv 等函数)
target_link_libraries(${OPS_FFT} PRIVATE
    tiling_api
)

##############################################################################
# 添加各算子子目录
##############################################################################

message(STATUS "==========================================")
message(STATUS "Registering operators...")
message(STATUS "==========================================")

foreach(OP ${BUILD_OPERATORS})
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/${OP}/CMakeLists.txt)
        add_subdirectory(${OP})
        message(STATUS "Added operator directory: ${OP}")
    else()
        message(WARNING "Operator ${OP} missing CMakeLists.txt, skipping")
    endif()
endforeach()

get_target_property(LIB_TARGET_SOURCE ${OPS_FFT} SOURCES)
if (NOT LIB_TARGET_SOURCE OR LIB_TARGET_SOURCE STREQUAL "")
    message(FATAL_ERROR "No sources added to cann_ops_fft target")
endif()

set_source_files_properties(
    ${LIB_TARGET_SOURCE}
    PROPERTIES LANGUAGE ASC
)

message(STATUS "==========================================")
message(STATUS "Operator registration completed")
message(STATUS "==========================================")
message(STATUS "")

##############################################################################
# 安装规则
##############################################################################

# 安装库文件到 ops_fft/lib64/
install(TARGETS ${OPS_FFT}
    LIBRARY DESTINATION ${OPS_FFT_LIB_INSTALL_DIR}
    RUNTIME DESTINATION bin
)

# 安装公共头文件到 ops_fft/include/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${OPS_FFT_INC_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

message(STATUS "Operator library 'ops_fft' configured successfully")
