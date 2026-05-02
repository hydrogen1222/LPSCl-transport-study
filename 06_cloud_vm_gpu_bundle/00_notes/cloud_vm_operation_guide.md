# 云 GPU 操作说明

更新日期：2026-04-11

这份文档对应当前的正式方案：

- GPU：`3080 Ti 12GB`
- 环境管理：`uv`
- Python：`3.11`
- 无 `slurm`
- 模型：`uma-s-1p2.pt`

## 1. 上传内容

需要上传到云主机的内容：

- 整个 [06_cloud_vm_gpu_bundle](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle)
- 模型文件 `uma-s-1p2.pt`

推荐目录：

```bash
~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle
~/models/uma-s-1p2.pt
```

兼容目录：

```bash
~/models/uma/uma-s-1p2.pt
```

脚本会优先找：

1. `~/models/uma-s-1p2.pt`
2. `~/models/uma/uma-s-1p2.pt`

## 2. 检查 GPU

```bash
nvidia-smi
```

如果这一步不正常，不要继续做环境安装。

## 3. 安装 uv

如果还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

## 4. 安装环境

进入 bundle 根目录：

```bash
cd ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle
```

执行：

```bash
bash 03_scripts/setup_gpu_env.sh
```

这个脚本会：

1. 安装 `Python 3.11`
2. 用 `uv` 创建 `.venv`
3. 安装 GPU 版 `torch`
4. 安装较新的 `fairchem-core`
5. 安装本地运行层 `02_runtime`

## 5. 检查环境

```bash
bash 03_scripts/check_gpu_env.sh
```

关键输出应包括：

```text
torch.cuda.is_available() = True
GPU 0 name = ...
```

## 6. 建议用 tmux 跑长任务

```bash
sudo apt update
sudo apt install -y tmux
tmux new -s lpscl_gpu
```

分离：

```bash
Ctrl+B 然后 D
```

恢复：

```bash
tmux attach -t lpscl_gpu
```

## 7. 先做 screening

### 7.1 smoke

```bash
MD_TEMP=600 MD_STEPS=20 MD_SAVE_INTERVAL=1 bash 03_scripts/run_md_single.sh bulk_ordered
MD_TEMP=600 MD_STEPS=20 MD_SAVE_INTERVAL=1 bash 03_scripts/run_md_single.sh gb_Sigma3_t3
```

### 7.2 benchmark

```bash
MD_TEMP=600 MD_STEPS=200 MD_SAVE_INTERVAL=20 bash 03_scripts/run_md_single.sh gb_Sigma3_t3
```

### 7.3 batch screening

```bash
bash 03_scripts/run_md_batch.sh smoke
bash 03_scripts/run_md_batch.sh core
bash 03_scripts/run_md_batch.sh contrast
```

## 8. 结果下载回本地后的检查

先修轨迹格式：

```bash
python 03_scripts/repair_xdatcar.py --root 04_runs
```

再做汇总：

```bash
python 03_scripts/postprocess_md_runs.py \
  --root 04_runs/md \
  --csv-out 00_notes/md_run_summary.csv \
  --md-out 00_notes/md_screening_summary.md
```

当前 bundle 本地已经做过这一轮，结果文件是：

- [md_run_summary.csv](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/md_run_summary.csv)
- [md_screening_summary.md](D:/毕业设计/LPSCl_UMA_transport_project/06_cloud_vm_gpu_bundle/00_notes/md_screening_summary.md)

## 9. 当前 screening 应该怎么理解

当前 `600 K / 700 K`、`1000` 步的结果，只能视为：

- 结构稳定性检查
- 短时间 Li 运动趋势检查
- 正式 production 的结构筛选

不能直接当成：

- 室温电导率
- 最终扩散系数

因为：

- `1000` 步在 `1 fs` 下只有 `1 ps`
- 室温下需要更长轨迹才有足够统计

## 10. 正式 conductivity 生产

当前推荐结构：

- `bulk_ordered`
- `gb_Sigma3_t3`
- `gb_Sigma3_t3_Li_vac_c1_s1`

当前推荐正式矩阵：

- 温度：`600 K`, `700 K`, `800 K`
- 每条轨迹：`20000` 步
- 保存间隔：`100`

直接运行：

```bash
TEMPERATURES="600 700 800" \
STEPS=20000 \
SAVE_INTERVAL=100 \
bash 03_scripts/run_conductivity_batch.sh
```

### 10.1 关于 ASE Langevin 的 `fixcm` 提醒

如果日志里看到：

```text
FutureWarning: The implementation of `fixcm=True` in `Langevin` ...
```

这不是作业崩溃，而是 ASE 3.28 的接口弃用提醒。

当前 bundle 已修正为：

- `Langevin(..., fixcm=False)`
- 同时给体系加 `FixCom`
- 并在初始化速度后调用 `Stationary(...)` 清零总动量

这是 ASE 当前推荐的写法。

处理原则：

- 之前的 `600 K / 700 K` screening 数据可以保留，因为它们主要用于筛选，而且体系规模较大，这个偏差通常不构成主导误差。
- 但正式 conductivity production 最好统一方法学，因此如果当前长轨迹批跑是用旧版 thermostat 启动的，建议停掉后从零重跑。

## 11. 时间预估

当前 screening 的实测速度大约是：

- `1000` 步约 `5.16 min`

因此：

- `20000` 步约 `1.72 h / run`
- `3` 个结构 × `3` 个温度 ≈ `15.5 h`

所以正式 production 可以安排在一个完整夜间窗口内完成。

## 12. 正式 production 完成后的下一步

后处理路线是：

1. 提取 Li 的 `MSD(t)`
2. 由 Einstein 关系得到 `D(T)`
3. 做 `ln(D)` 对 `1/T` 的 Arrhenius 拟合
4. 外推到 `300 K`
5. 在明确说明假设的前提下换算为离子电导率

## 13. 这一步不是训练

当前主线仍然是：

- 预训练 UMA 模型推理
- 结构优化
- MD
- 轨迹后处理

不是：

- 训练新模型
- 微调 UMA

训练需要单独的高质量标注数据和新一轮算力预算，不在当前毕设主线中。
