#-----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it
# under the terms of the CANN Open Software License Agreement.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES
# OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text
# of the License.
#-----------------------------------------------------------------------------

##############################################################################
# src/fft1_d/CMakeLists.txt
#
# fft1D算子编译配置 - 根据SoC自动选择arch目录
##############################################################################

# 注册 fft1_d 算子（ARCH_DIR由register_operator自动根据NPU架构选择）
register_operator(NAME fft1_d)