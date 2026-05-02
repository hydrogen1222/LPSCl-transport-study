# 云 GPU 运行计划

更新日期：2026-04-13

## 1. 当前项目状态

这套云端计算已经完成两个阶段：

### 阶段一：screening

- `19` 条 MD
- `8` 个结构
- `600 K / 700 K`
- 最长 `1.0 ps`

它的作用是：

- 判断结构是否稳定
- 判断 Li 是否有明显运动
- 从多个候选结构中筛选出最值得进入正式 conductivity 生产的对象

### 阶段二：formal conductivity production

已经完成：

- `bulk_ordered`
- `gb_Sigma3_t3`
- `gb_Sigma3_t3_Li_vac_c1_s1`

在以下温度上的正式轨迹：

- `600 K`
- `700 K`
- `800 K`

每条轨迹：

- `20000` 步
- `20 ps`

总计：

- `9` 条正式 production 轨迹

## 2. 当前正式 production 的意义

这批 `20 ps` 轨迹和前面的 `1 ps` screening 不一样。

它们现在已经可以用来：

- 提取 `Li MSD`
- 估算 `D(T)`
- 做 Arrhenius 拟合
- 外推到 `300 K`
- 在明确说明假设的前提下估算室温离子电导率

## 3. 当前结果应该怎样表述

当前阶段最合理的表述是：

- 已经得到一版基于 UMA MD 的 `D(T)` 与室温电导率估计
- 但这些结果仍然属于**第一版上界估计**

原因是：

- 每个结构每个温度目前只有一条轨迹
- 当前电导率换算基于 Nernst-Einstein 关系
- Haven ratio 暂未单独校正
- Li 载流子浓度目前按总 Li 数密度估算

所以现在不应把这些数值写成“最终精确值”，而应写成：

- first-pass estimate
- upper-bound estimate
- preliminary conductivity estimate

## 4. 当前最核心的比较对象

现在最重要的比较已经非常清楚：

1. `bulk_ordered`
   体相基线
2. `gb_Sigma3_t3`
   中性晶界参考
3. `gb_Sigma3_t3_Li_vac_c1_s1`
   晶界 + vacancy 耦合体系

这一组三元对比已经足够支撑你的毕业设计主线：

- 晶界是否改变 Li 输运趋势
- 晶界上的 vacancy 是否进一步增强或削弱输运

## 5. 当前不建议继续做的事

现在不建议立刻做：

- 所有 GB 家族全部扩展成 formal production
- 重新训练或微调模型
- 再回头大规模重复 screening

这些都不是当前最优先的工作。

## 6. 当前建议的后续顺序

1. 用 [conductivity_production_summary.csv](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/conductivity_production_summary.csv) 和 [conductivity_production_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/conductivity_production_summary.md) 整理第一版结论
2. 开始搭论文里的图表：
   - `MSD(t)` 曲线
   - `D(T)` 对比
   - `ln(D)` vs `1/T`
   - `sigma(300 K)` 对比
3. 如果时间允许，再做一轮增强统计：
   - 三结构补独立随机种子
   - 或三结构补到 `50000` 步

## 7. 当前路线边界

当前路线仍然是：

- 预训练 UMA 推理
- 结构优化
- 分子动力学
- 后处理分析

不是：

- 重新训练一个势函数

训练不是当前毕设主线。
