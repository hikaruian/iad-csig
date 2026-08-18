# CSIG Real-IAD Variety — INP-Former SOTA Pipeline

第七届 CSIG 图技大赛 / 复杂工业场景异常检测。

本仓库实现 **INP-Former（CVPR 2025）** 的完整多类无监督训练与提交流水线，并按赛题的 **5 视角 Real-IAD Variety 子集** 与官方 zip 格式做了适配。

论文：*Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection*  
官方代码：[luow23/INP-Former](https://github.com/luow23/INP-Former)  
骨干权重：Facebook DINOv2-with-registers（公开通用预训练，赛题允许）。

在 MVTec-AD / VisA / Real-IAD 的 **unified multi-class** 设定下，INP-Former / INP-Former++ 目前是公开榜上的第一档方法（Real-IAD：约 90.5–90.7 I-AUROC，99.0–99.2 P-AUROC），也是本题题面明确推荐的冲分基准。本实现保持与论文一致的：

- 冻结 DINOv2-Reg 中层特征（ViT-B/14 默认，可切 ViT-L）
- INP Extractor + INP Coherence Loss
- 8 层 INP-Guided Decoder（无 attention residual，强制经原型重建）
- Soft Mining Loss（`global_cosine_hm_adaptive`, y=3）
- 分组松重建（Dinomaly 同款 layer groups）
- StableAdamW + warmup cosine，200 epoch

并针对本题加了三件必要的工程：

1. **5 视角样本级打分**：每视角取 top-1% 像素均值，再对 5 个视角取 max（任一视角出现缺陷即判异常）。
2. **448×448 单通道 mask**，使用**全局**线性缩放写入 uint8（禁止逐图 min-max，否则会破坏 P-AUROC / P-AUPR / P-F1max 的跨图排序）。
3. INP 从测试图自身提取，对 B 榜未见类有零样本泛化能力。

---

## 环境

目标机器：**单机 2×T4 16GB**（也兼容单卡）。T4 是 Turing，**只有 fp16 AMP，没有 bf16 / TF32**。

```bash
conda activate py311          # 赛题指定
pip install -r requirements.txt
```

第一次运行会通过 `torch.hub` 下载 DINOv2-Reg 权重。双卡时由 **rank0 先下、再 barrier**，避免两个进程把缓存写坏。无外网时：`--encoder-source timm`。

---

## 数据

```
CSIG/
├── Train/      # 50 类 × 20 样本 × 5 视角，全部为正常
│   └── <class>/Sxxxx/{0,1,2,3,4}.png
└── Test_A/     # 50 类 × 15 样本 × 5 视角，正常 + 异常
    └── (同上)
```

只允许使用本题 Train / Test。DINOv2 / ImageNet 等**通用**预训练权重可以使用。

---

## 训练（2×T4 16GB）

`torchrun` 启动 DDP。`--batch-size` 是 **每张卡** 的 micro-batch。

默认：`2 GPU × batch 4 × grad-accum 2 = global 16`（与论文一致），fp16 AMP 默认打开。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1   # 无 NVLink 的双 T4 建议开，防 NCCL 卡住
export NCCL_IB_DISABLE=1

torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --train-root /path/to/CSIG/Train \
  --save-dir runs/inpformer_b14 \
  --encoder dinov2reg_vit_base_14 \
  --image-size 448 \
  --inp-num 6 \
  --epochs 200 \
  --batch-size 4 \
  --grad-accum 2 \
  --num-workers 2 \
  --amp
```

单卡（不走 DDP）：

```bash
python train.py --train-root /path/to/CSIG/Train --save-dir runs/inpformer_b14 \
  --batch-size 4 --grad-accum 4 --amp
```

若 16GB 仍 OOM：`--batch-size 2 --grad-accum 4`，或加 `--grad-checkpoint`。

ViT-L 在 T4 16GB 上很紧，需要 `--batch-size 1 --grad-checkpoint --grad-accum 8`；更稳妥仍用 ViT-B。

`--residual` 打开 INP-Former++ 残差重建。产出：`model.pth` / `last.pth` / `best.pth`（已去掉 DDP 的 `module.` 前缀，可直接给 `infer.py`）。

也可直接：`bash run_example.sh`（先改脚本里的数据路径）。

---

## 推理与打包

单卡：

```bash
python infer.py \
  --test-root /path/to/CSIG/Test_A \
  --ckpt runs/inpformer_b14/model.pth \
  --train-root /path/to/CSIG/Train \
  --out-dir outputs/submission \
  --zip outputs/my_submission.zip \
  --samples-per-batch 2 \
  --sigma 4 --max-ratio 0.01 --reduce max --tta-flip --amp
```

双卡分片推理（各写一部分 mask，rank0 汇总 csv / zip）：

```bash
torchrun --standalone --nproc_per_node=2 infer.py \
  --test-root /path/to/CSIG/Test_A \
  --ckpt runs/inpformer_b14/model.pth \
  --train-root /path/to/CSIG/Train \
  --zip outputs/my_submission.zip \
  --samples-per-batch 2 --tta-flip --amp
```

`--train-root` 用于两件事：全局 mask 缩放，以及（默认开启的）**按类/视角正常统计**，用来抬 P-AP / P-F1max。

拉高像素指标请看下面「如何抬 P-AP / P-F1max」。

zip 解压后结构（评测要求）：

```
submission.csv
predicted_masks/<class>/<Sxxxx>/{0,1,2,3,4}_mask.png
```

`submission.csv`：

```
group_folder,anomaly_score
3_adapter/S0001,0.03210000
battery/S0002,0.87650000
```

---

## 本地验证

若你自己划了一份带 `ground_truth.csv` + `masks/` 的验证集：

```bash
python eval_local.py \
  --standard-dir path/to/val_gt \
  --submission-dir outputs/submission \
  --out metrics.json
```

会同时打印题面公式 `0.4·S_cls + 0.6·S_seg` 与 INP-Former 选手包公式 `0.3·S_cls + 0.7·S_seg`。

---

## 为什么这是本题该用的 SOTA

| 方法 | 设定 | Real-IAD I-AUROC | 像素定位 | 零样本 |
|---|---|---|---|---|
| PatchCore / SimpleNet | 多类 | 明显掉点 | 中 | 弱 |
| UniAD / DiAD / MambaAD | 多类 | 中上 | 中上 | 弱 |
| Dinomaly (CVPR 2025) | 多类 | ~89.3 / 90.1 (ViT-L) | 强 | 弱 |
| **INP-Former / ++** | 多类 + few-shot + 部分 ZS | **90.5 / 90.7** | **最强** | **有** |

本题 60% 分数在像素级（P-AUROC / P-AUPR / P-F1max 宏平均），INP 从**当前图**提取正常原型，对划痕、凹陷这类局部缺陷的对齐远好于“训练集记忆库”。B 榜冷启动同类也吃这套机制。

不在这里实现非 SOTA 的 one-class 小模型或纯分类头。

---

## 复现建议（按性价比）

1. 先用 2×T4 + ViT-B/14、200 epoch、global batch 16 出 A 榜第一版。
2. 用一小份自划验证集看 I-AUROC 与 P-F1max；若定位虚高/过碎，调 `--sigma`（4→6）或 `--mask-scale`。
3. 有 24 GB 再训 ViT-L + `--residual` + `--tta-flip`。
4. 样本分默认 `max`（漏检少）。若正常样本被顶得太高，可试 `--reduce lse`。
5. **不要**对每张图做 min-max 归一化再存 mask。

---

## 如何抬 P-AP / P-F1max

这两项看的是「高分像素里有多少是真缺陷」。P-AUROC 很容易到 0.93+，假阳性一多 AP/F1 照样只有 0.3。

推理（**不用重训**，默认已开 `--refine`）：

```bash
python infer.py ... --train-root /path/to/CSIG/Train \
  --refine --sigma 2.5 --gamma 1.4 --fg-gate --tta-flip
```

做了什么：

1. 用 Train 正常图估计每个 `(类别, 视角)` 的误差 μ/σ，测试图做 **z-score**。金属拉丝、螺丝牙这些「永远难重建」的纹理不再占高分。
2. 负 z 截断 + `gamma=1.4`，把真缺陷峰拉尖。
3. 画布边缘衰减（ViT 插值伪影）。
4. 便宜的 RGB 前景门控，压掉背景误报。
5. 模糊从 4 降到 **2.5**，小划痕不会被糊没。

训练侧（再涨一截）：

- 加 `--residual`（INP-Former++，论文里主要涨的就是 P-AP）。
- 不要用 ViT-L 硬上 16GB 导致训不稳。
- 若显存够，`--image-size 518`（DINOv2 原生分辨率，须能被 14 整除）。

**禁止**：逐图 min-max、把 `--sigma` 加到 6+、对 mask 做连通域滤除（Variety 缺陷经常只有几像素）。

---

## 目录

```
csig_sota/
  train.py              DDP 多类无监督训练（torchrun）
  infer.py              五视角推理 + 打包 zip（可双卡分片）
  eval_local.py         本地宏平均指标
  run_example.sh        2×T4 一键训练/推理
  requirements.txt
  src/
    dist_utils.py       torchrun / DDP / AMP(fp16)
    checkpoint.py       去 module. 前缀、校验可训练权重
    encoder.py          DINOv2-Reg（rank0 下载）
    blocks.py           Aggregation / Prototype attention
    model.py            INP-Former / ++
    losses.py           AMP 安全的 Soft Mining + INP Coherence
    optim.py            StableAdamW + WarmCosine
    data.py             CSIG 多视角 DataLoader
    postprocess.py      高斯平滑 / top-1% / 全局缩放
    submission.py       官方 zip
    metrics.py          与天池口径一致的宏平均
  tools/sanity_check.py 无 GPU 的格式自检
```

训练 / 调参 / 提交只使用本题 Train 与 Test。未引入任何外部异常检测数据集，也未调用在线大模型。

---

## 已修的关键 bug

| 问题 | 后果 | 修复 |
|---|---|---|
| Soft-mining hook 在 fp16 下 factor 爆炸 | T4 AMP 梯度 Inf/NaN | factor 改 fp32 并 clamp 到 32 |
| `StableAdamW` 里 `max(1.0, Tensor)` | 新版 PyTorch 直接报错 | `.item()` 后再比较 |
| `inference_mode` 特征进训练图 | 反传报错 / 图断裂 | 改为 `no_grad` + `detach().clone()` |
| 双进程同时 `torch.hub.load` | 权重缓存损坏 | rank0 下载 → 文件戳 → 其余 rank 读缓存 |
| `state_dict` 带 `module.` | 推理 decoder 变随机初始化 | 保存前 unwrap，加载时剥前缀并检查可训练 key |
| 某一 rank 遇到 NaN 就 `continue` | **DDP 死锁** | 全体 all_reduce MIN(finite) 后一起跳过 |
| 未 `DistributedSampler.set_epoch` | 每个 epoch 同一顺序 | 每个 epoch `set_epoch` |
| 首次下载 DINOv2 超过 NCCL 默认 10min | torchrun 首跑超时退出 | **prefetch 放在 init_process_group 之前**，NCCL timeout=2h |
| `opt_steps = steps // accum` | leftover micro-step 让 lr 对不齐 | 改为向上取整 |
| rank0 `auto_scale` 抛错 | rank1 卡在 broadcast | 先 all_reduce 状态再 broadcast |
| WORLD_SIZE>1 但没有 CUDA | 两个进程当单卡互踩 | 直接报错 |
| 推理忽略 ckpt 的 image_size | 分辨率错位 | 默认跟 checkpoint |
| 无 NVLink 双 T4 NCCL P2P | 启动后卡住 | 默认 `NCCL_P2P_DISABLE=1` |
