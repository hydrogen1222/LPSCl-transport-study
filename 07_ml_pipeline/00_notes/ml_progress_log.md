# ML Pipeline Progress Log

更新日期：2026-04-13

## 已完成

1. 重新定义 conductivity 结果口径  
   当前统一使用：
   - `D_tracer(T)`
   - `Ea`
   - `sigma_NE,upper(300 K)`

2. 新建独立 ML 流水线目录  
   - [07_ml_pipeline](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline)

3. 规划本地数据流水线  
   当前拆成三层：
   - 结构特征表
   - MD 动力学标签表
   - 可训练样本表

4. 规划 baseline 验证方式  
   - 优先验证 `log10(D_tracer)` 预测
   - 使用 leave-one-structure-out
   - formal 数据不足时只做 fit-preview，不做强评估

5. 规划下一轮 GPU 标签扩展  
   - 准备 formal MD 结构清单
   - 先扩展结构覆盖，再谈更强模型

6. 训练数据表已经生成  
   当前落盘结果：
   - `structure_features.csv`：20 个结构
   - `transport_run_labels.csv`：28 条 MD 运行标签
   - `structure_label_summary.csv`：17 个结构具备至少一部分标签
   - `ml_training_table.csv`：28 条可训练样本

7. baseline 已经跑完  
   当前主要结果：
   - screening `1000` 步、`600/700 K`：16 行、8 个结构，可做 leave-one-structure-out
   - formal `20000` 步、`600/700/800 K`：9 行、3 个结构，不足以做严肃验证
   - 当前 `mean` baseline 反而优于 `ridge` / `knn`
   - 这说明现阶段最大的瓶颈仍然是标签量太少，而不是模型不够复杂

8. 下一轮 GPU manifest 已生成  
   - `24` 条任务
   - `8` 个待补 formal 标签的结构

9. 下一轮 GPU 批跑入口已准备  
   - [run_manifest_batch.sh](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch/run_manifest_batch.sh)
   - 可直接驱动 `06_cloud_vm_gpu_bundle` 里的 `run_md_single.sh`

## 当前阶段判断

- 当前最重要的不是继续解释旧的绝对电导率数值。
- 当前最重要的是把“结构 -> 输运标签”的数据链固定下来。
- 一旦数据链稳定，后续每补一轮 GPU 结果，都可以直接接到训练表里。

## 下一步

1. 把下一轮 `24` 条 formal MD 提交到云 GPU
2. 回传结果后重跑 `build_ml_datasets.py`
3. 再次运行 baseline，观察 leave-one-structure-out 误差是否下降
4. 只有当 formal 结构数明显增加后，才开始做更强的回归模型
