#!/bin/bash

# 设置环境变量，例如 MASTER_ADDR 和 MASTER_PORT（如果适用）
# export MASTER_ADDR=localhost
# export MASTER_PORT=12345

# 定义要传递给 deeptrain.py 的参数
#ckpt_path=<your_ckpt_path>
#fg_path=<your_fg_path>
#matte_path=<your_matte_path>
#batch_size=32
#epochs=<number_of_epochs>

# 如果您希望从命令行传递参数，请取消以下注释并根据需要提供默认值
ckpt_path=${ckpt_path:-None}
fg_path=${fg_path:-/mnt/data/Train/FG}
matte_path=${matte_path:-/mnt/data/Train/Alpha}
batch_size=${batch_size:-0}
epochs=${epochs:-50}
config_file=${config_file:-deepspeed.config}

# 使用 DeepSpeed 启动分布式训练，传递参数
deepspeed deeptrain.py \
    --ckptpath="$ckpt_path" \
    --fg_path="$fg_path" \
    --matte_path="$matte_path" \
    --batch_size="$batch_size" \
    --epoch="$epochs" \
    --deepspeed_config="$config_file"

# 检查 RANK 环境变量，只在 rank 0 上执行模型保存和上传
if [ "$RANK" -eq 0 ]; then
    echo "Rank 0 is saving and uploading the model..."
    python3 preserve.py push
else
    echo "Rank $RANK finished training but will not save or upload the model."
fi