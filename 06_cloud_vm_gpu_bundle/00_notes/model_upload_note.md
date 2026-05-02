# 模型上传与兼容性说明

## 1. 当前默认模型

当前云端正式方案默认使用：

```bash
uma-s-1p2.pt
```

原因很直接：

- `3080 Ti 12GB` 对 `uma-m-1p1.pt` 风险太高
- `uma-s-1p2.pt` 更适合当前显存条件下的结构优化和 MD

## 2. 推荐放置位置

优先放在：

```bash
~/models/uma-s-1p2.pt
```

兼容放在：

```bash
~/models/uma/uma-s-1p2.pt
```

当前脚本支持两种路径。

## 3. 为什么不能再配旧版 fairchem

你之前遇到的报错是：

```text
HydraModel.__init__() got an unexpected keyword argument 'model_id'
```

这说明：

- `uma-s-1p2.pt` 是较新的 checkpoint
- PyPI 上较旧的 `fairchem-core==2.14.0` 不兼容

因此当前环境方案是：

- `Python 3.11`
- 较新的 `fairchem-core`

## 4. 如果你想手动指定模型

可以显式设置：

```bash
UMA_MODEL_PATH=/absolute/path/to/your_model.pt
```

例如：

```bash
UMA_MODEL_PATH=~/models/uma-s-1p2.pt bash 03_scripts/run_md_single.sh gb_Sigma3_t3
```

## 5. 当前不建议的做法

在 `3080 Ti 12GB` 上，不建议把默认模型切回：

```bash
uma-m-1p1.pt
```

除非你已经单独验证过显存占用和 walltime。
