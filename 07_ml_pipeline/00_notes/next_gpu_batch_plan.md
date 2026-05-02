# 下一轮 GPU 批量打标签计划

更新日期：2026-04-13

## 目标

当前 formal `20000` 步 production 只有三个核心结构：

1. `bulk_ordered`
2. `gb_Sigma3_t3`
3. `gb_Sigma3_t3_Li_vac_c1_s1`

这足够支撑趋势分析，但不足以支撑真正的结构到输运机器学习预测。

下一轮 GPU 批量打标签的目标是：

- 扩大结构覆盖
- 保持标签定义一致
- 优先补 formal `20000` 步 MD

## 当前建议的下一批结构

1. `bulk_Li_vac_c1_s1`
2. `bulk_Li_vac_c2_s1`
3. `gb_Sigma3_t1`
4. `gb_Sigma3_t2`
5. `gb_Sigma3_t1_Li_vac_c1_s1`
6. `gb_Sigma3_t2_Li_vac_c1_s2`
7. `gb_Sigma3_t3_Li_vac_c1_s2`
8. `gb_Sigma3_t3_Li_vac_c2_s2`

对应 manifest 文件：

- [next_md_labeling_manifest.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch/next_md_labeling_manifest.csv)

## 标签策略

每个结构建议：

- `600 K`
- `700 K`
- `800 K`
- `20000` 步
- `save_interval = 100`

如果预算允许，后续增强优先顺序是：

1. 对三条核心 formal 结构补第二随机种子
2. 对新结构补 formal 第一条轨迹
3. 最后再延长到 `50000` 步

## 为什么不是直接重训势函数

因为当前瓶颈不是势函数本身，而是：

- 带标签的动力学样本太少

所以最优先的投入产出比仍然是：

- 继续生产一致的 MD 标签
- 而不是重训 UMA
