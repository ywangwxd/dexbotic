---
name: dm0-finetune-robotwin2
description: DM0 模型在 robotwin2_adjust_bottle 上微调的完整适配记录
metadata: 
  node_type: memory
  type: reference
  originSessionId: a7b47d96-5f6f-4a46-89f4-435f1c0c6026
---

## DM0 微调 RobotWin2：完整适配记录

### 最终配置

| 参数 | 值 |
|------|------|
| 任务 | `robotwin2_adjust_bottle` (50 demos, 7220 帧, 3 视角) |
| 基座模型 | `./checkpoints/DM0-base` (2.9B) |
| 可训练参数 | ~0.79B / 2.90B |
| 冻结 | LLM (1.5B) + PE 视觉塔 (0.3B) |
| 训练 | Action Expert (0.5B) + Projector + lm_head |
| 优化器 | AdamW, lr=5e-5, warmup=200, cosine_with_min_lr |
| Batch | 4/GPU × 8 grad_accum = 有效 32 |
| 精度 | bf16 |
| 总步数 | 10000 |
| 保存 | 每 1000 步 |
| GPU | 单卡 24GB (4090D) |
| DeepSpeed | 无（单卡不需要） |

---

### 关键修复清单

#### 1. SDPA 替换 Eager Attention
**文件**: `dexbotic/model/dm0/dm0_arch.py:_compute_merged_layer`

DM0 合并注意力直接用 `eager_attention_forward`，每层物化 `[B, 16, S, S]` 注意力矩阵，28 层合计 ~12GB。替换为 `F.scaled_dot_product_attention`，Flash Attention 不物化矩阵。

#### 2. PE 视觉塔 RoPE freqs_cache NaN
**文件**: `dexbotic/model/modules/mm_vision/pe/pe_model.py:RoPE2D.forward`

**根因**：`to_bfloat16_for_selected_params` 调用 `Module.to(dtype=bf16)` 递归转换模型树时，PE 视觉塔内预计算的三角函数 buffer（`freqs_cache`，非持久 buffer）的 fp32→bf16 转换会损坏数值，产生 NaN。NaN 经 RoPE 传播至整个前向，导致 loss=NaN。

**修复**：RoPE forward 入口处自检 `freqs_cache` 是否含 NaN，若含 NaN 则用 `_compute_2d_freqs()` 重新计算并放到正确设备和 dtype。

```python
# pe_model.py RoPE2D.forward():
if torch.isnan(self.freqs_cache).any():
    self.freqs_cache = self._compute_2d_freqs().to(
        device=q.device, dtype=self.freqs_cache.dtype)
```

#### 3. freeze 未生效
**文件**: `playground/benchmarks/robotwin2/robotwin2_dm0.py:DM0ModelConfig.build_model`

`build_model` 覆盖父类方法但没调用 `self._freeze_model(model)`，导致 `freeze_llm=True, freeze_mm_vision=True` 从未生效。全部 2.9B 参数设为可训练，Adam 优化器状态 ~20GB，直接 OOM。

**修复**：在 `build_model` 末尾加 `self._freeze_model(model)`。

#### 4. Loss / Grad Norm 显示异常
**文件**: `dexbotic/exp/trainer.py:compute_loss`

DM0 forward 用 `**kwargs`，HF Trainer 检测到后设 `model_accepts_loss_kwargs=True`，跳过了 `loss = loss / gradient_accumulation_steps`。导致日志 loss 和 grad_norm 都是 8 个 micro-batch 的累加值而非平均。

**修复**：`compute_loss` 中手动 `loss = loss / self.args.gradient_accumulation_steps`。

#### 5. transformers 5.x 兼容性
| 问题 | 文件 | 修复 |
|------|------|------|
| `tokenizer` → `processing_class` | `trainer.py` | 重映射参数名 |
| Mistral regex 警告 | `base_exp.py` | `fix_mistral_regex=True` |
| `llm_config` 无默认值 | `dexbotic_arch.py` | 加 `None` 默认 |
| `low_cpu_mem_usage` vs vision tower | `cogact_exp.py` 等 | 移除 |
| `numpydantic` 不兼容 pydantic 2.13 | — | 升级 1.1.0→1.8.1 |

#### 6. suffix_out dtype 不匹配
**文件**: `dexbotic/model/dm0/dm0_arch.py:forward`

Action Expert 的 `model.norm` 保留 fp32，suffix_out 为 fp32，但 `action_out_proj` 是 bf16。修复为显式 cast。

#### 7. wandb 配置
**文件**: `trainer.py:_link_exp_config`, `robotwin2_dm0.py:DM0TrainerConfig`

添加 `report_to="wandb"` 并传入 TrainingArguments。

---

### 图像分辨率

DM0-base 使用 `pe_lang_l14_728`——PE 感知编码器，不是普通 CLIP。两步 stride-2 卷积做 4×下采样，728×728 → 182×182 → CLIP 输出仅 ~170 token/图。**无需 resize**。

---

### 诊断中的错误方向

1. **图片 resize**：误以为 728×728 进入 CLIP 会产生 2705 token/图。DM0 有 PE 编码器已做下采样。
2. **反复切换 ZeRO 版本**：多次在 ZeRO-2/3/纯 DDP 间切换，浪费在参数收集机制而非根本问题上。
3. **`torch.no_grad()` 导致 NaN**：试图用 `no_grad()` 阻止 ZeRO-3 参数收集，但 ZeRO-3 不收集时参数为零，导致 NaN。
4. **`GatheredParameters` 尝试**：试图用 DeepSpeed API 手动收集参数，但 exit 时参数冲突崩溃。
5. **耗时最长的坑**：`freqs_cache` NaN 的根因定位——从 OOM 排查到 attention 排查到 RoPE，层层深入才找到非持久 buffer 的 bf16 转换问题。

---

### 最终结论

单卡 24GB 跑冻结 LLM+PE 的 DM0 微调完全够用，不需要 DeepSpeed 不需要多卡。关键就是三个点：SDPA、freqs_cache 自愈、freeze 真正生效。
