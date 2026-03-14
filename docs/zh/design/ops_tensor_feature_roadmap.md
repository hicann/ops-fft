# ops-tensor 功能路线图

> 对标 cutensor/hiptensor 能力，按实现难度和优先级分阶段规划

---

## 一、功能全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ops-tensor 功能全景                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│  │   Phase 1   │   │   Phase 2   │   │   Phase 3   │   │   Phase 4   │   │
│  │   基础框架   │   │   算子扩展   │   │   高级操作   │   │   生态完善   │   │
│  │   ★☆☆☆☆    │   │   ★★☆☆☆    │   │   ★★★☆☆    │   │   ★★★★☆    │   │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   │
│         │                 │                 │                 │            │
│         ▼                 ▼                 ▼                 ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        功能依赖关系                                   │   │
│  │                                                                      │   │
│  │  Handle ──► TensorDesc ──► OpDesc ──► Plan ──► Execute              │   │
│  │     │           │            │          │          │                 │   │
│  │     └───────────┴────────────┴──────────┴──────────┘                 │   │
│  │                         核心框架 (Phase 1)                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Phase 1：基础框架（难度 ★☆☆☆☆）

**目标**：跑通端到端流程，实现最小可行版本

### 2.1 句柄管理 (Handle Management)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建句柄 | `acltensorCreate` | ★ | 0.1K | 初始化设备信息 |
| 销毁句柄 | `acltensorDestroy` | ★ | 0.05K | 释放资源 |
| ~~缓存调整~~ | `acltensorHandleResizePlanCache` | - | - | Phase 2 |
| ~~缓存文件~~ | `acltensorHandle*PlanCacheToFile` | - | - | Phase 3 |

### 2.2 张量描述符 (Tensor Descriptor)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建描述符 | `acltensorCreateTensorDescriptor` | ★ | 0.15K | 仅支持 FP32 |
| 销毁描述符 | `acltensorDestroyTensorDescriptor` | ★ | 0.05K | - |

### 2.3 操作描述符 (Operation Descriptor)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建 Binary | `acltensorCreateElementwiseBinary` | ★★ | 0.2K | 仅支持 ADD |
| 销毁描述符 | `acltensorDestroyOperationDescriptor` | ★ | 0.05K | - |

### 2.4 Plan 管理

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建偏好 | `acltensorCreatePlanPreference` | ★ | 0.1K | 简化版 |
| 销毁偏好 | `acltensorDestroyPlanPreference` | ★ | 0.02K | - |
| 创建 Plan | `acltensorCreatePlan` | ★★ | 0.15K | 内核选择逻辑 |
| 销毁 Plan | `acltensorDestroyPlan` | ★ | 0.05K | - |

### 2.5 执行函数

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| Binary 执行 | `acltensorElementwiseBinaryExecute` | ★★ | 0.2K | 仅 ADD + FP32 |

### 2.6 算子实现

| 算子 | 数据类型 | 难度 | 代码量 | 说明 |
|------|---------|------|--------|------|
| ADD | FP32 | ★★ | 0.4K | Host + Kernel |

### 2.7 工具函数

| 功能 | API | 难度 | 代码量 |
|------|-----|------|--------|
| 错误字符串 | `acltensorGetErrorString` | ★ | 0.05K |
| 版本号 | `acltensorGetVersion` | ★ | 0.02K |

### Phase 1 汇总

| 指标 | 数值 |
|------|------|
| API 数量 | 12 个 |
| 算子数量 | 1 个 (ADD) |
| 数据类型 | 1 种 (FP32) |
| 预估代码量 | **~1.8K** |
| 预估工时 | 2-3 周 |

---

## 三、Phase 2：算子扩展（难度 ★★☆☆☆）

**目标**：扩展算子和数据类型，添加缓存机制

### 3.1 扩展算子

| 算子 | 难度 | 代码量 | 优先级 | 说明 |
|------|------|--------|--------|------|
| MUL | ★★ | 0.3K | P0 | 乘法 |
| SUB | ★★ | 0.3K | P0 | 减法 |
| DIV | ★★ | 0.3K | P1 | 除法 |
| MAX | ★★ | 0.3K | P1 | 最大值 |
| MIN | ★★ | 0.3K | P1 | 最小值 |

### 3.2 扩展数据类型

| 类型 | 难度 | 代码量 | 优先级 | 说明 |
|------|------|--------|--------|------|
| FP16 | ★★ | 0.4K | P0 | 扩展现有算子 |
| BF16 | ★★ | 0.4K | P1 | 扩展现有算子 |

### 3.3 Trinary 操作

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建描述符 | `acltensorCreateElementwiseTrinary` | ★★ | 0.25K | - |
| 执行函数 | `acltensorElementwiseTrinaryExecute` | ★★ | 0.2K | - |

### 3.4 PlanCache 机制

| 功能 | 难度 | 代码量 | 说明 |
|------|------|--------|------|
| 缓存结构 | ★★ | 0.2K | LRU 缓存 |
| 缓存查询 | ★★ | 0.1K | hash → solution |
| 缓存持久化 | ★★ | 0.15K | 文件读写 |

### 3.5 日志系统

| API | 难度 | 代码量 |
|-----|------|--------|
| `acltensorLoggerSetCallback` | ★ | 0.05K |
| `acltensorLoggerSetFile` | ★ | 0.03K |
| `acltensorLoggerOpenFile` | ★ | 0.05K |
| `acltensorLoggerSetLevel` | ★ | 0.03K |
| `acltensorLoggerSetMask` | ★ | 0.03K |
| `acltensorLoggerForceDisable` | ★ | 0.02K |

### Phase 2 汇总

| 指标 | 数值 |
|------|------|
| 新增 API | 10 个 |
| 新增算子 | 5 个 |
| 新增数据类型 | 2 种 |
| 预估代码量 | **~3.4K** |
| 预估工时 | 3-4 周 |

---

## 四、Phase 3：高级操作（难度 ★★★☆☆）

**目标**：实现 Contraction 和 Reduction 操作

### 4.1 Contraction (张量收缩)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建描述符 | `acltensorCreateContraction` | ★★★ | 0.3K | D=α*A*B+β*C |
| 执行函数 | `acltensorContract` | ★★★ | 0.3K | - |
| MatMul 算子 | FP32/FP16 | ★★★ | 1.5K | 多种分块策略 |

### 4.2 Reduction (张量归约)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建描述符 | `acltensorCreateReduction` | ★★★ | 0.25K | D=α*reduce(A)+β*C |
| 执行函数 | `acltensorReduce` | ★★★ | 0.25K | - |
| ReduceSum | FP32/FP16 | ★★ | 0.4K | - |
| ReduceMax | FP32/FP16 | ★★ | 0.4K | - |
| ReduceMin | FP32/FP16 | ★★ | 0.4K | - |

### 4.3 一元操作符扩展

| 操作符 | 难度 | 代码量 | 说明 |
|--------|------|--------|------|
| SQRT | ★★ | 0.2K | 平方根 |
| RELU | ★★ | 0.2K | ReLU |
| EXP | ★★ | 0.2K | 指数 |
| LOG | ★★ | 0.2K | 对数 |
| ABS | ★ | 0.15K | 绝对值 |
| NEG | ★ | 0.15K | 取负 |
| SIGMOID | ★★ | 0.2K | Sigmoid |
| TANH | ★★ | 0.2K | Tanh |

### 4.4 属性接口

| API | 难度 | 代码量 |
|-----|------|--------|
| `acltensorOperationDescriptorSetAttribute` | ★★ | 0.1K |
| `acltensorOperationDescriptorGetAttribute` | ★★ | 0.1K |
| `acltensorPlanPreferenceSetAttribute` | ★★ | 0.1K |
| `acltensorPlanGetAttribute` | ★★ | 0.1K |
| `acltensorEstimateWorkspaceSize` | ★★ | 0.1K |

### Phase 3 汇总

| 指标 | 数值 |
|------|------|
| 新增 API | 10 个 |
| 新增算子 | 4 个 |
| 预估代码量 | **~4.8K** |
| 预估工时 | 4-6 周 |

---

## 五、Phase 4：生态完善（难度 ★★★★☆）

**目标**：实现 Permutation，扩展数据类型，性能优化

### 5.1 Permutation (张量置换)

| 功能 | API | 难度 | 代码量 | 说明 |
|------|-----|------|--------|------|
| 创建描述符 | `acltensorCreatePermutation` | ★★ | 0.2K | B=α*A |
| 执行函数 | `acltensorPermute` | ★★ | 0.2K | - |
| Transpose | FP32/FP16 | ★★ | 0.4K | - |

### 5.2 数据类型扩展

| 类型 | 难度 | 代码量 | 说明 |
|------|------|--------|------|
| INT8 | ★★ | 0.5K | 量化支持 |
| INT32 | ★★ | 0.3K | - |
| FP64 | ★★ | 0.4K | 双精度 |

### 5.3 激活函数

| 操作符 | 难度 | 代码量 |
|--------|------|--------|
| GELU | ★★★ | 0.3K |
| SILU | ★★★ | 0.3K |

### 5.4 性能优化

| 优化项 | 难度 | 代码量 |
|--------|------|--------|
| 内核自动调优 | ★★★★ | 0.5K |
| 多流并行 | ★★★ | 0.3K |
| 内存池 | ★★★ | 0.3K |

### Phase 4 汇总

| 指标 | 数值 |
|------|------|
| 新增 API | 4 个 |
| 预估代码量 | **~3.5K** |
| 预估工时 | 3-4 周 |

---

## 六、Phase 5：高级特性（难度 ★★★★★）

**目标**：Complex 支持、JIT 编译、自定义算子

### 6.1 Complex 支持

| 类型 | 难度 | 代码量 |
|------|------|--------|
| Complex FP32 | ★★★★ | 0.8K |
| Complex FP64 | ★★★★ | 0.6K |

### 6.2 JIT 编译

| 功能 | 难度 | 代码量 |
|------|------|--------|
| 运行时编译 | ★★★★★ | 1.0K |
| 内核生成 | ★★★★★ | 0.8K |

### 6.3 分布式支持

| 功能 | 难度 | 代码量 |
|------|------|--------|
| 多卡并行 | ★★★★★ | 1.0K |
| 通信优化 | ★★★★★ | 0.8K |

---

## 七、总体统计

### 7.1 API 总览

| 阶段 | 句柄 | 描述符 | Plan | 执行 | 工具 | 日志 | 总计 |
|------|------|--------|------|------|------|------|------|
| Phase 1 | 2 | 3 | 4 | 1 | 2 | 0 | **12** |
| Phase 2 | 3 | 1 | 0 | 1 | 0 | 6 | **11** |
| Phase 3 | 0 | 2 | 5 | 2 | 0 | 0 | **9** |
| Phase 4 | 0 | 1 | 0 | 1 | 0 | 0 | **2** |
| **总计** | 5 | 7 | 9 | 5 | 2 | 6 | **34** |

### 7.2 算子总览

| 类别 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | 总计 |
|------|---------|---------|---------|---------|------|
| Elementwise Binary | ADD | MUL/SUB/DIV/MAX/MIN | - | - | 6 |
| Elementwise Trinary | - | 融合运算 | - | - | 1 |
| Contraction | - | - | MatMul | - | 1 |
| Reduction | - | - | Sum/Max/Min | - | 3 |
| Permutation | - | - | - | Transpose | 1 |
| **总计** | 1 | 6 | 4 | 1 | **12** |

### 7.3 数据类型总览

| 阶段 | 支持的数据类型 |
|------|---------------|
| Phase 1 | FP32 |
| Phase 2 | + FP16, BF16 |
| Phase 3 | - |
| Phase 4 | + INT8, INT32, FP64 |
| Phase 5 | + Complex FP32, Complex FP64 |

### 7.4 代码量汇总

| 阶段 | 预估代码量 | 累计 |
|------|-----------|------|
| Phase 1 | ~1.8K | 1.8K |
| Phase 2 | ~3.4K | 5.2K |
| Phase 3 | ~4.8K | 10.0K |
| Phase 4 | ~3.5K | 13.5K |
| Phase 5 | ~4.2K | 17.7K |

---

## 八、优先级决策矩阵

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          优先级决策矩阵                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        实现难度                                             │
│                    ★☆☆☆☆    ★★☆☆☆    ★★★☆☆    ★★★★☆    ★★★★★       │
│                 ┌─────────┬─────────┬─────────┬─────────┬─────────┐       │
│           高    │ Phase1  │ Phase2  │ Phase3  │ Phase4  │ Phase5  │       │
│                 │ 框架    │ 算子扩展│ Contraction│ 优化   │ JIT    │       │
│    使用频率     ├─────────┼─────────┼─────────┼─────────┼─────────┤       │
│                 │         │ PlanCache│         │         │         │       │
│           中    │         │ Trinary │         │ Permute │ Complex│       │
│                 │         │ 日志    │         │ INT8   │         │       │
│                 ├─────────┼─────────┼─────────┼─────────┼─────────┤       │
│                 │         │         │         │ GELU   │ 分布式  │       │
│           低    │         │         │         │         │         │       │
│                 └─────────┴─────────┴─────────┴─────────┴─────────┘       │
│                                                                             │
│  推荐顺序: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 九、风险与依赖

### 9.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| AscendC API 变化 | 高 | 固定 CANN 版本 |
| 性能不达标 | 中 | 多种内核策略 |
| 内存对齐问题 | 中 | 严格对齐检查 |

### 9.2 外部依赖

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| CANN | >= 8.0 | AscendC 支持 |
| ACL | >= 8.0 | 运行时 API |
| CMake | >= 3.16 | 构建系统 |

---

## 十、参考对照表

### 10.1 与 hiptensor API 对应关系

| hiptensor API | ops-tensor API | Phase |
|---------------|----------------|-------|
| `hiptensorCreate` | `acltensorCreate` | 1 |
| `hiptensorDestroy` | `acltensorDestroy` | 1 |
| `hiptensorCreateTensorDescriptor` | `acltensorCreateTensorDescriptor` | 1 |
| `hiptensorDestroyTensorDescriptor` | `acltensorDestroyTensorDescriptor` | 1 |
| `hiptensorCreateElementwiseBinary` | `acltensorCreateElementwiseBinary` | 1 |
| `hiptensorElementwiseBinaryExecute` | `acltensorElementwiseBinaryExecute` | 1 |
| `hiptensorCreateElementwiseTrinary` | `acltensorCreateElementwiseTrinary` | 2 |
| `hiptensorElementwiseTrinaryExecute` | `acltensorElementwiseTrinaryExecute` | 2 |
| `hiptensorCreateContraction` | `acltensorCreateContraction` | 3 |
| `hiptensorContract` | `acltensorContract` | 3 |
| `hiptensorCreateReduction` | `acltensorCreateReduction` | 3 |
| `hiptensorReduce` | `acltensorReduce` | 3 |
| `hiptensorCreatePermutation` | `acltensorCreatePermutation` | 4 |
| `hiptensorPermute` | `acltensorPermute` | 4 |
| `hiptensorLogger*` (6个) | `acltensorLogger*` (6个) | 2 |

---

**文档版本**：v1.0
**最后更新**：2024年
