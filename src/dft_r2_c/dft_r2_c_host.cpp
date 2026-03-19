#include "dft_r2_c_host.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>

#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"

// 常量定义
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;
constexpr int32_t CUBE_DATA_COUNT_SMALL = 128;
constexpr int32_t CUBE_DATA_COUNT_LARGE = 64;
constexpr int32_t THRESHOLD_K = 32768;
constexpr size_t WORKSPACE_SIZE = 32; // 对应 DFT_R2C_WORKSPACE_SIZE

#define GM_ADDR uint8_t*

// 声明外部函数 (实现在 dft_r2c.cce 中)
extern void dft_r2c_kernel_do(
    uint32_t blockDim, 
    void* l2ctrl, 
    void* stream, 
    GM_ADDR gmA, 
    GM_ADDR gmB, 
    GM_ADDR gmC, 
    GM_ADDR workSpace, 
    GM_ADDR tilingParaGm);

// 辅助函数：计算对齐后的尺寸
static size_t GetAlignedSize(size_t size) {
    return (size + 31) / 32 * 32; // 假设32字节对齐
}

// 核心实现函数
aclError aclFftR2C(aclrtStream stream, float* input_addr, float* output_addr, 
                  int64_t fft_n, int64_t batch_size, bool forward) 
{
    // 1. 直接使用传入的地址
    if (!input_addr || !output_addr) {
        std::cerr << "Input/Output address is nullptr" << std::endl;
        return -1;
    }

    for (int i = 0; i < 16; i ++ ) {
        std::cout << input_addr[i] << ' ';
    }
    std::cout << std::endl;

    // 2. 构造旋转矩阵 - 对应 dft_r2c_core.cpp 中的 InitRotationMatrix
    int64_t in_size = fft_n;
    int64_t out_size = 2 * (fft_n / 2 + 1); // 复数存储大小
    size_t matrix_size = in_size * out_size * sizeof(float);
    
    std::vector<float> rotation_matrix_host(in_size * out_size, 0);
    
    // 预计算 cos/sin 表
    std::vector<float> cos_table(fft_n);
    std::vector<float> sin_table(fft_n);
    for (int64_t i = 0; i < fft_n; i++) {
        cos_table[i] = std::cos(K_2PI * i / fft_n);
        sin_table[i] = std::sin(K_2PI * i / fft_n);
    }

    // 填充旋转矩阵
    // rotationMatrix[i * outSize + 2 * j] = cos
    // rotationMatrix[i * outSize + 2 * j + 1] = +/- sin
    float sign = forward ? -1.0f : 1.0f;
    for (int64_t i = 0; i < in_size; i++) {
        for (int64_t j = 0; j < (fft_n / 2 + 1); j++) {
            int64_t idx = (i * j) % fft_n;
            rotation_matrix_host[i * out_size + 2 * j] = cos_table[idx];
            rotation_matrix_host[i * out_size + 2 * j + 1] = sign * sin_table[idx];
        }
    }

    // 3. 分配设备内存并拷贝旋转矩阵
    void* rotation_matrix_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&rotation_matrix_dev, matrix_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(rotation_matrix_dev, matrix_size, rotation_matrix_host.data(), matrix_size, ACL_MEMCPY_HOST_TO_DEVICE));
    std::unique_ptr<void, AclrtFreeDeleter> d_matrix_guard(rotation_matrix_dev);

    // 4. 准备 Tiling 数据 - 对应 dft_r2c_tiling.cpp
    FftR2CTilingData tiling_data;
    tiling_data.batchSize = 1; // Kernel 内部循环处理 batch
    tiling_data.m = static_cast<int32_t>(batch_size);
    tiling_data.n = static_cast<int32_t>((fft_n / 2 + 1) * 2);
    tiling_data.k = static_cast<int32_t>(fft_n);
    tiling_data.transA = 0;
    tiling_data.transB = 0;

    // 设置分块大小
    if (tiling_data.k > THRESHOLD_K) {
        tiling_data.m0 = CUBE_DATA_COUNT_LARGE;
        tiling_data.n0 = CUBE_DATA_COUNT_LARGE;
        tiling_data.k0 = CUBE_DATA_COUNT_LARGE;
    } else {
        tiling_data.m0 = CUBE_DATA_COUNT_SMALL;
        tiling_data.n0 = CUBE_DATA_COUNT_SMALL;
        tiling_data.k0 = CUBE_DATA_COUNT_SMALL;
    }

    // 分配 Kernel 设备内存
    const int64_t inSignal = fft_n;
    const int64_t outSignal = fft_n / 2 + 1;
    const int64_t tensorInSize = batch_size * inSignal;
    const int64_t tensorOutSize = batch_size * outSignal;
    
    std::vector<int64_t> selfShape = {batch_size, inSignal};
    std::vector<int64_t> outShape = {batch_size, outSignal};

    size_t inputDataSize = GetShapeSize(selfShape) * sizeof(int64_t);
    size_t outputDataSize = GetShapeSize(outShape) * sizeof(int64_t);

    void* input_dev = nullptr;
    void* output_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&input_dev, inputDataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&output_dev, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(input_dev, inputDataSize, input_addr, inputDataSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(output_dev, outputDataSize, output_addr, outputDataSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 分配 Tiling 设备内存
    void* tiling_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&tiling_dev, sizeof(FftR2CTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tiling_dev, sizeof(FftR2CTilingData), &tiling_data, sizeof(FftR2CTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    // 智能指针指定删除器，在unique_ptr析构时自动调用指定的函数释放资源
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(input_dev);
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(tiling_dev);
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(output_dev);

    // 5. 分配 Workspace
    void* workspace_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&workspace_dev, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));

    // 6. 启动 Kernel
    // uint32_t numblocks = 8; 
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numblocks = ascendcPlatform->GetCoreNumAic();

    dft_r2c_kernel_do(
        numblocks, 
        nullptr, 
        stream, 
        (uint8_t*)input_dev,         // 直接转换
        (uint8_t*)rotation_matrix_dev, 
        (uint8_t*)output_dev,        // 直接转换
        (uint8_t*)workspace_dev, 
        (uint8_t*)tiling_dev
    );

    // 等待计算完成 (同步)
    aclrtSynchronizeStream(stream);

    // 5. 拷回结果并打印
    auto size = GetShapeSize(outShape) * 2; // complex64 size
    std::vector<float> outData(size, 0);
    auto ret = aclrtMemcpy(outData.data(),
        outData.size() * sizeof(float),
        output_dev,
        size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    // 打印前16个复数结果
    for (int64_t i = 0; i < 16; i++) {
        std::cout << "(" << outData[2*i] << ", " << outData[2*i+1] << ")\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;


    // 7. 资源清理
    aclrtFree(input_dev);
    aclrtFree(output_dev);
    aclrtFree(rotation_matrix_dev);
    aclrtFree(tiling_dev);
    aclrtFree(workspace_dev);

    return ACL_SUCCESS;
}