# LPSCl UMA GPU Cloud Bundle

这是当前用于云 GPU 虚拟机的正式计算目录。当前这套目录已经不只是“待运行包”，而是包含了：

- 结构输入
- UMA 运行时
- 云端执行脚本
- screening 结果
- formal conductivity production 结果
- 后处理与摘要文档

## 当前状态

### screening

- 已完成 `19` 条 screening MD
- 覆盖 `8` 个结构
- 温度覆盖 `600 K`、`700 K`
- 最长 screening 轨迹为 `1.0 ps`

### formal conductivity production

已完成一轮正式生产：

- 结构：
  - `bulk_ordered`
  - `gb_Sigma3_t3`
  - `gb_Sigma3_t3_Li_vac_c1_s1`
- 温度：
  - `600 K`
  - `700 K`
  - `800 K`
- 每条轨迹：
  - `20000` 步
  - `20 ps`
- 共完成：
  - `9` 条正式 production 轨迹

## 当前最重要的结果文件

- [md_run_summary.csv](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/md_run_summary.csv)
- [md_screening_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/md_screening_summary.md)
- [conductivity_production_summary.csv](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/conductivity_production_summary.csv)
- [conductivity_production_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/conductivity_production_summary.md)

## 当前方法学状态

- `XDATCAR` 写出格式已修复，能正常用于后处理和可视化
- NVT thermostat 已按 ASE 3.28 推荐方式修正：
  - `fixcm=False`
  - `FixCom`
  - `Stationary`
- 当前 conductivity 结果已经可以做第一版分析
- 但室温电导率仍应理解为：
  - **基于单条轨迹/单个随机种子的第一版上界估计**
  - 而不是最终定值

## 目录说明

- [00_notes](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes)
  说明文档、结果摘要、分析结论
- [01_inputs](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/01_inputs)
  结构输入
- [02_runtime](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/02_runtime)
  patched `umakit`
- [03_scripts](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/03_scripts)
  运行、修复、后处理脚本
- [04_runs](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/04_runs)
  已完成的优化和 MD 输出

## 推荐阅读顺序

1. [cloud_vm_operation_guide.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/cloud_vm_operation_guide.md)
2. [cloud_vm_run_plan.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/cloud_vm_run_plan.md)
3. [md_screening_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/md_screening_summary.md)
4. [conductivity_production_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/conductivity_production_summary.md)
5. [transport_analysis_plan.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/transport_analysis_plan.md)

## 当前建议的下一步

现在不建议再盲目扩大计算规模。更合理的是：

1. 先基于已经完成的 `9` 条正式 production 轨迹做第一版论文图表
2. 把室温扩散系数和电导率写成“第一版上界估计”
3. 如果时间允许，再做一项增强：
   - 同样三结构再补一条独立随机种子
   - 或把现有三结构加长到 `50000` 步

当前这一步已经从“算不动”切换到了“可以开始系统分析”。 
