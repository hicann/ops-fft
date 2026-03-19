// src/fft_r2c/fft_r2c_tiling.h
#ifndef FFT_R2C_TILING_H
#define FFT_R2C_TILING_H

#include <cstdint>
#include <vector>
#include "acl/acl.h" // 仅需标准 ACL 头文件

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

struct FftR2CTilingData {
    int32_t batchSize{0};
    int32_t m{0};       // batch
    int32_t n{0};       // (N/2 + 1) * 2
    int32_t k{0};       // fftN
    int32_t transA{0};
    int32_t transB{0};
    int32_t m0{0};      // split m
    int32_t n0{0};      // split n
    int32_t k0{0};      // split k
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 修改后的接口：直接接受设备地址
extern "C" aclError aclFftR2C(
    aclrtStream stream, 
    float* input_addr,        // 输入数据设备地址
    float* output_addr,       // 输出数据设备地址
    int64_t fft_n, 
    int64_t batch_size, 
    bool forward
);

#endif