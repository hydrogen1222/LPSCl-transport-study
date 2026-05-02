# 正式 Conductivity Production 摘要

- 当前正式 production 轨迹长度：`20000` 步
- 纳入本次分析的结构数：`3`
- 扩散系数来自每条轨迹后半段 `MSD` 的线性拟合。
- Arrhenius 外推使用 `600 K`、`700 K`、`800 K` 三个温度点。
- 下表中的室温电导率不是最终值，而是 `tracer diffusion + Nernst-Einstein` 的第一版上界估计。

## 汇总表

| 结构 | Ea (eV) | D_tracer(300 K) (cm^2/s) | sigma_NE,upper(300 K) (mS/cm) |
| --- | ---: | ---: | ---: |
| gb_Sigma3_t3_Li_vac_c1_s1 | 0.1348 | 8.506e-07 | 111.050 |
| gb_Sigma3_t3 | 0.1661 | 4.564e-07 | 59.790 |
| bulk_ordered | 0.1974 | 1.139e-07 | 15.593 |

## 当前解释

- `bulk_ordered` 是当前体相基线。
- `gb_Sigma3_t3` 是当前最稳定的中性晶界代表。
- `gb_Sigma3_t3_Li_vac_c1_s1` 是当前晶界 + vacancy 的代表结构。
- 这些结果目前更适合用来讨论相对趋势，而不是直接当作实验可比的绝对室温电导率。
- 当前电导率偏高是预期现象，因为这里使用的是 tracer diffusion、总 Li 数密度以及 Haven ratio = 1 的 Nernst-Einstein 上界。
- 如果后续要让 conductivity 结论更稳，最划算的增强方式是：
  1. 对这三个结构每个温度再补一条独立随机种子；或
  2. 把现有三结构轨迹延长到 `50000` 步以上；或
  3. 在后处理中加入相关扩散修正，而不只使用 tracer diffusion。

## 当前可直接使用的结论

- 相对趋势目前支持：`bulk_ordered < gb_Sigma3_t3 < gb_Sigma3_t3_Li_vac_c1_s1`。
- 当前数据适合支撑“晶界及 vacancy 倾向于提高 Li 迁移能力”的趋势判断。
- 当前数据不适合支撑“绝对室温离子电导率已经准确预测”的结论。
