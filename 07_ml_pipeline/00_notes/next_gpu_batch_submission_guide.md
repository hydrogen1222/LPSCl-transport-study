# 下一轮云 GPU 提交说明

更新日期：2026-04-13

## 1. 当前准备好的输入

下一轮待提交清单位于：

- [next_md_labeling_manifest.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch/next_md_labeling_manifest.csv)

当前规模：

- `8` 个结构
- `24` 条 MD 任务
- 每个结构对应 `600 / 700 / 800 K`
- 每条任务 `20000` 步，`save_interval = 100`

## 2. 这一轮的目标

这轮不是为了直接再给论文加一张图，而是为了：

- 把 formal MD 标签从 `3` 个结构扩到 `11` 个左右
- 让“结构 -> 输运标签”的机器学习任务第一次具备像样的样本覆盖

## 3. 建议的提交顺序

优先级已经写在 manifest 里，建议按这个顺序提交：

1. `bulk_Li_vac_c1_s1`
2. `bulk_Li_vac_c2_s1`
3. `gb_Sigma3_t1`
4. `gb_Sigma3_t2`
5. `gb_Sigma3_t1_Li_vac_c1_s1`
6. `gb_Sigma3_t2_Li_vac_c1_s2`
7. `gb_Sigma3_t3_Li_vac_c1_s2`
8. `gb_Sigma3_t3_Li_vac_c2_s2`

## 4. 提交原则

这轮仍然保持和当前 formal production 完全一致的设置：

- 同一套模型：`uma-s-1p2.pt`
- 同一套 thermostat 与 runtime
- 同一套 `600 / 700 / 800 K`
- 同一套 `20000` 步
- 同一套 `save_interval = 100`

不要在这一轮中途改模型、改步长、改统计方式，否则标签定义会被污染。

## 5. 当前不建议做的事

这轮先不要：

- 补第二随机种子
- 直接延长到 `50000` 步
- 临时换成别的 UMA 模型

原因很简单：先扩结构覆盖，比先堆单结构统计更重要。

## 6. 回传之后本地要做什么

等你把云端结果拉回本地后，下一步直接按顺序执行：

1. 更新 `06_cloud_vm_gpu_bundle/04_runs/md`
2. 重跑 [postprocess_md_runs.py](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/03_scripts/postprocess_md_runs.py)
3. 重跑 [analyze_conductivity.py](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/03_scripts/analyze_conductivity.py)
4. 重跑 [build_ml_datasets.py](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/03_scripts/build_ml_datasets.py)
5. 重跑 [run_baseline_models.py](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/03_scripts/run_baseline_models.py)

## 7. 云主机上如何直接批跑

如果云主机上已经同时放好了：

- `06_cloud_vm_gpu_bundle`
- `07_ml_pipeline`

那么可以直接在云主机执行：

```bash
cd ~/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch
bash run_manifest_batch.sh
```

这个脚本会逐条读取：

- [next_md_labeling_manifest.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch/next_md_labeling_manifest.csv)

并自动调用：

- [run_md_single.sh](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/03_scripts/run_md_single.sh)

## 8. 这一轮完成后的判断标准

如果这轮 24 条任务顺利完成，下一阶段最重要的判断标准是：

- formal `20000` 步结构数是否达到 `>= 8`
- baseline 在 leave-one-structure-out 下是否开始优于简单 mean baseline

如果这两个条件都满足，下一步就可以认真进入“第一版结构到输运 ML 回归”阶段了。
