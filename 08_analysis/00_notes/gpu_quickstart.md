# 云 GPU 快速上手 — 15 天冲刺版

> 更新日期：2026-04-14

## 0. 连接

```bash
ssh -p 39704 root@120.221.212.211
# 密码: hi0lfY6A7bjy
```

## 1. 检查现有环境

先确认机器上是否已有之前的环境和数据：

```bash
# 检查 GPU
nvidia-smi

# 检查 uv
uv --version

# 检查之前的 bundle
ls ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle/.venv/bin/python 2>/dev/null && echo "ENV EXISTS" || echo "NEED SETUP"

# 检查模型
ls ~/models/uma-s-1p2.pt 2>/dev/null && echo "MODEL EXISTS" || echo "NEED UPLOAD"
```

## 2. 如果需要重新部署（全新机器）

### 2.1 上传 bundle（本地 PowerShell 执行）

```powershell
# 在本地 Windows 上
scp -P 39704 -r "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle" root@120.221.212.211:~/LPSCl_UMA_gpu/
scp -P 39704 "path/to/uma-s-1p2.pt" root@120.221.212.211:~/models/
```

### 2.2 安装环境（云端执行）

```bash
cd ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle
bash 03_scripts/setup_gpu_env.sh
bash 03_scripts/check_gpu_env.sh
```

## 3. 如果环境已存在（同一台机器）

只需要上传新脚本：

```powershell
# 本地 PowerShell
scp -P 39704 "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle\03_scripts\run_formal_batch.sh" root@120.221.212.211:~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle/03_scripts/
```

## 4. 开始批量计算

### 4.1 创建 screen 会话

```bash
screen -S lpscl
```

### 4.2 启动批量 MD

```bash
cd ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle
bash 03_scripts/run_formal_batch.sh 2>&1 | tee formal_batch.log
```

### 4.3 分离 screen（不会中断计算）

按 `Ctrl+A` 然后按 `D`

### 4.4 查看进度

```bash
# 重新连接 screen
screen -r lpscl

# 或直接看日志尾部
tail -20 ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle/formal_batch_*.log
```

## 5. 时间估计

| 结构类型 | 原子数 | 预计每条 (default) | 预计每条 (turbo) |
|---------|--------|-------------------|-----------------|
| bulk_* | 414-416 | ~70 min | ~40 min |
| gb_* | 622-624 | ~105 min | ~60 min |

总计 24 条任务：
- default 模式：~36 小时
- turbo 模式：~20 小时

## 6. 完成后下载结果

```powershell
# 本地 PowerShell
scp -P 39704 -r root@120.221.212.211:~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle/04_runs/md/ "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle\04_runs\md\"
scp -P 39704 root@120.221.212.211:~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle/formal_batch_*.log "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle\"
```

## 7. 本地分析（下载后执行）

```powershell
cd "D:\毕业设计\LPSCl_UMA_transport_project"
python 08_analysis/01_scripts/run_all_analysis.py
```

这会自动：
1. 计算所有 MSD + D_tracer
2. Arrhenius 拟合
3. RDF 分析
4. ML 预测器训练
5. 生成所有论文图

结果在 `08_analysis/02_results/` 和 `08_analysis/03_figures/`。
