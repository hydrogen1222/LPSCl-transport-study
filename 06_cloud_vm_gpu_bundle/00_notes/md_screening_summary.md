# 当前 MD Screening 总结

- 已完成 MD 运行数：`28`
- 已覆盖结构数：`8`
- 当前已覆盖温度：`600 K, 700 K, 800 K`
- 当前最长轨迹长度：`20.00 ps`

## 当前结果如何理解

- 当前 `1000` 步轨迹属于 screening，不是最终 conductivity 生产数据。
- 在 `1.0 fs` 步长下，`1000` 步大约只有 `1.0 ps`；这足够做稳定性和短时间 Li 运动趋势判断，但不足以稳健给出室温电导率。
- `600 K` 和 `700 K` 属于高温采样点，用来加速 Li 跳跃并为后续 Arrhenius 外推做准备；它们不是室温电导率本身。
- 当前 workflow 仍然是预训练 UMA 推理 + MD，不是模型训练阶段。

## 初步扩散排序

| 结构 | 运行名 | 目标温度 (K) | 末帧 Li MSD (A^2) | 粗略 D (cm^2/s) | 最短原子对 (A) |
| --- | --- | ---: | ---: | ---: | ---: |
| gb_Sigma3_t1 | md_700K_1000steps | 700 | 3.5996 | 5.678e-05 | 1.882 (P-S) |
| gb_Sigma3_t1 | md_600K_1000steps | 600 | 3.5143 | 5.451e-05 | 1.865 (P-S) |
| gb_Sigma3_t3_Li_vac_c2_s2 | md_600K_1000steps | 600 | 2.1189 | 3.825e-05 | 1.891 (P-S) |
| gb_Sigma3_t2_Li_vac_c1_s2 | md_700K_1000steps | 700 | 2.4913 | 3.586e-05 | 1.871 (P-S) |
| bulk_ordered | md_700K_1000steps | 700 | 2.2158 | 3.357e-05 | 1.941 (P-S) |
| gb_Sigma3_t3 | md_700K_1000steps | 700 | 2.0689 | 3.282e-05 | 1.934 (P-S) |
| gb_Sigma3_t2_Li_vac_c1_s2 | md_600K_1000steps | 600 | 2.1116 | 3.260e-05 | 1.901 (P-S) |
| gb_Sigma3_t1_Li_vac_c1_s1 | md_700K_1000steps | 700 | 1.9741 | 2.739e-05 | 1.872 (P-S) |
| gb_Sigma3_t3_Li_vac_c2_s2 | md_700K_1000steps | 700 | 1.9255 | 2.697e-05 | 1.899 (P-S) |
| gb_Sigma3_t1_Li_vac_c1_s1 | md_600K_1000steps | 600 | 1.8829 | 2.651e-05 | 1.876 (S-S) |

- 当前 `1 ps` screening 中，`t1` 的短时间 Li 运动看起来比 `t3` 更强，但这仍然只是 screening 级别的观察。
- 正式 production 仍然优先保留 `t3`，因为在已有静态验证里，`t3` 仍然是结构和能量上更稳妥的 GB 家族。

## 下一步正式 production 建议

正式 conductivity 生产，建议先保留这三个结构：

- `bulk_ordered`
- `gb_Sigma3_t3`
- `gb_Sigma3_t3_Li_vac_c1_s1`

推荐的第一轮 production 矩阵：

- 温度：`600 K`, `700 K`, `800 K`
- 单条轨迹长度：至少 `20000` 步，也就是 `20 ps`
- 保存间隔：`100` 步
- 晶胞保持固定，继续复用当前已准备好的起始结构

之后再对 `ln(D)` 与 `1/T` 做 Arrhenius 拟合，外推到 `300 K`，并在明确说明 Nernst-Einstein 假设的前提下换算为电导率。
