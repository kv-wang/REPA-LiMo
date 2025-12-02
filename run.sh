#!/bin/bash

export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"

# 训练脚本：使用FSDP、最小模型(SiT-B/2)和LiMO优化器
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8-fsdp" \
  --data-dir=[YOUR_DATA_PATH] \
  --use-fsdp \
  --fsdp-sharding-strategy="FULL_SHARD" \
  --fsdp-backward-prefetch="BACKWARD_PRE" \
  --optimizer="limo" \
  --momentum=0.95 \
  --limo-momentum-2=0.98 \
  --rms-scale \
  --nesterov \
  --ns-steps=5 \
  --limo-eps=1e-8 \
  --limo-use-scale


