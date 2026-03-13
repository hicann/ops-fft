# FFT Handle 结构设计

## 1. 概述

本文档描述 ops-fft 库的 FFT 计划句柄（Handle）结构设计，参考 hipFFT 的设计经验，结合 CANN 算子的特点，设计一个精简高效的 Handle 结构。

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| 显式API | 创建计划时确定所有属性（transformType、direction、placement） |
| 精简结构 | 只需1个FFT计划，不像hipFFT需要4个 |
| 内部管理 | 工作缓冲区、TilingData由算子内部计算和管理 |
| 轻量高效 | 仅存储描述"做什么FFT"的信息 |

## 3. Handle 结构定义

```cpp
/**
 * ops-fft FFT 计划句柄结构
 * 仅存储 FFT 计划的基本参数和属性
 * TilingData、工作缓冲区等由算子内部计算和管理
 */
struct FftPlanHandle {
    // ==================== 变换参数 ====================
    int32_t         rank;               // FFT 维度 (1, 2, 3)
    int32_t         lengths[3];         // 各维度长度
    int32_t         batch;              // 批次数量
    DataType        dtype;              // 数据类型

    // ==================== 变换属性 ====================
    TransformType   transformType;      // R2C/C2R/C2C/Z2D/D2Z/Z2Z
    Direction       direction;          // FORWARD/BACKWARD
    Placement       placement;          // INPLACE/OUTOFPLACE
    NormMode        normMode;           // BACKWARD/FORWARD/ORTHO
};
```

## 4. 成员说明

| 成员 | 类型 | 说明 |
|------|------|------|
| `rank` | `int32_t` | FFT 维度，取值 1、2、3 |
| `lengths[3]` | `int32_t[3]` | 各维度的 FFT 长度 |
| `batch` | `int32_t` | 批量变换数量 |
| `dtype` | `DataType` | 数据类型（float/complex/double 等） |
| `transformType` | `TransformType` | 变换类型（创建时确定） |
| `direction` | `Direction` | 变换方向（创建时确定） |
| `placement` | `Placement` | 原位/非原位（创建时确定） |
| `normMode` | `NormMode` | 归一化模式 |

## 5. 枚举类型定义

```cpp
// 变换类型
enum class TransformType {
    R2C,     // 实数到复数
    C2R,     // 复数到实数
    C2C,     // 复数到复数
    D2Z,     // 双精度实数到复数
    Z2D,     // 双精度复数到实数
    Z2Z      // 双精度复数到复数
};

// 变换方向
enum class Direction {
    FORWARD  = -1,  // 前向 FFT
    BACKWARD = 1   // 逆 FFT
};

// 数据放置
enum class Placement {
    INPLACE,     // 原位变换
    OUTOFPLACE   // 非原位变换
};

// 归一化模式
enum class NormMode {
    BACKWARD,  // 仅逆变换归一化
    FORWARD,   // 仅正变换归一化
    ORTHO      // 正逆变换都归一化（正交FFT）
};
```

## 6. API 设计

```cpp
// 创建计划（显式指定所有属性）
FftResult FftPlanCreate(FftPlanHandle** plan,
                        int32_t rank,
                        int32_t* lengths,
                        TransformType type,
                        Direction direction,
                        Placement placement,
                        int32_t batch,
                        NormMode normMode);

// 设置数据类型（可选，默认 float）
FftResult FftSetDataType(FftPlanHandle* plan, DataType dtype);

// 执行变换
FftResult FftExecute(FftPlanHandle* plan, void* in, void* out, aclrtStream stream);

// 销毁计划
FftResult FftPlanDestroy(FftPlanHandle* plan);
```

## 7. 与 hipFFT 对比

| 方面 | hipFFT | ops-fft |
|------|--------|---------|
| **设计目标** | 兼容 cuFFT API | 精简高效 |
| **FFT 计划数** | 4 个（需覆盖所有 placement × direction 组合） | 1 个（创建时确定属性） |
| **属性确定时机** | 执行时确定（通过参数和指针比较） | 创建时确定 |
| **工作缓冲区** | Handle 管理 | 算子内部管理 |
| **TilingData** | 无 | 算子内部计算 |
| **回调支持** | 有 | 无（可扩展） |
| **多 GPU 支持** | 有 | 无（可扩展） |
| **内存开销** | ~4x | ~1x |

## 8. 设计优势

1. **内存效率**：只需 1 个 FFT 计划，内存占用约为 hipFFT 的 1/4

2. **创建效率**：计划创建时间约为 hipFFT 的 1/4

3. **执行高效**：执行时无需动态选择计划，直接执行

4. **简洁明了**：Handle 只包含描述"做什么 FFT" 的信息，所有执行细节由算子内部处理

5. **易于扩展**：结构清晰，后续可根据需要添加新功能

## 9. 使用示例

```cpp
// 创建 FFT 计划
FftPlanHandle* plan;
int32_t lengths[1] = {1024};

FftPlanCreate(&plan,
             1,              // rank
             lengths,        // lengths
             R2C,            // type
             FORWARD,        // direction
             OUTOFPLACE,     // placement
             1,              // batch
             BACKWARD);      // normMode

// 设置数据类型
FftSetDataType(plan, DT_FLOAT);

// 执行 FFT
FftExecute(plan, input, output, stream);

// 销毁计划
FftPlanDestroy(plan);
```

## 10. 参考资料

- [hipFFT API 参考](../hipfft_api_reference.md)
- [算子开发指南](../develop/operator_development_guide.md)
- [rfft1_d 算子实现](../../../src/rfft1_d/)
