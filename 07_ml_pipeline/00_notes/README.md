# 07 ML Pipeline

本目录用于承接“结构 -> 输运标签 -> 机器学习预测”的本地数据流水线。

当前定位：

- 不负责生产 MD
- 不直接连接云 GPU 运行
- 负责把现有结构、CP2K 校验和 UMA-MD 标签整理成可训练表
- 负责 baseline 回归和下一轮 GPU 批量打标签 manifest

核心脚本：

- [build_ml_datasets.py](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/03_scripts/build_ml_datasets.py)
- [run_baseline_models.py](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/03_scripts/run_baseline_models.py)
- [prepare_next_gpu_batch.py](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/03_scripts/prepare_next_gpu_batch.py)

核心输出：

- [structure_features.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/01_datasets/structure_features.csv)
- [transport_run_labels.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/01_datasets/transport_run_labels.csv)
- [structure_label_summary.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/01_datasets/structure_label_summary.csv)
- [ml_training_table.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/01_datasets/ml_training_table.csv)
- [baseline_metrics.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/01_datasets/baseline_metrics.csv)
- [next_md_labeling_manifest.csv](D:/毕业设计/LPSCl_UMA_transport_project/07_ml_pipeline/02_gpu_batch/next_md_labeling_manifest.csv)

当前原则：

- 直接训练目标优先使用 `log10(D_tracer)`。
- `sigma_NE,upper(300 K)` 只保留为展示型派生标签。
- 在 formal 结构数还很少时，不对“绝对室温离子电导率预测”做强声明。
