#!/bin/bash


# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --ckpt_path)
            ckpt_path="$2"
            shift # past argument
            ;;
        --fg_path)
            fg_path="$2"
            shift # past argument
            ;;
        --matte_path)
            matte_path="$2"
            shift # past argument
            ;;
        --batch_size)
            batch_size="$2"
            shift # past argument
            ;;
        --epochs)
            epochs="$2"
            shift # past argument
            ;;
        --config_file)
            config_file="$2"
            shift # past argument
            ;;
        --access_id)
            access_id="$2"
            shift # past argument
            ;;
        --secret_key)
            secret_key="$2"
            shift # past argument
            ;;
        --area)
            area="$2"
            shift # past argument
            ;;
        --endpoint)
            endpoint="$2"
            shift # past argument
            ;;
        --save_path)
            save_path="$2"
            shift # past argument
            ;;
        *)    # unknown option
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
    shift # past argument or value
done



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
access_id=${access_id:-<your_access_id>}
secret_key=${secret_key:-<your_secret_key>}
area=${area:-tdcasual}
endpoint=${endpoint:-beijing}
save_path=${save_path:-model.pth}

echo "$batch_size"
echo "$fg_path"
echo "$matte_path"
echo "$batch_size"
echo "$epochs"
echo "$config_file"
echo "$access_id"
echo "$secret_key"
echo "$area"
echo "$endpoint"
echo "$save_path"
# 使用 DeepSpeed 启动分布式训练，传递参数
deepspeed deepown.py \
    --ckptpath="$ckpt_path" \
    --fg_path="$fg_path" \
    --matte_path="$matte_path" \
    --batch_size="$batch_size" \
    --epoch="$epochs" \
    --deepspeed_config="$config_file" \
    --epoch="$epochs" \
    --save_path="$save_path" 




# 检查 RANK 环境变量，只在 rank 0 上执行模型保存和上传
if [ "$RANK" -eq 0 ]; then
    echo "Rank 0 is saving and uploading the model..."
    
    # 直接执行 preserve.py 并传递参数
    python3 preserve.py push \
        --access-id="$access_id" \
        --secret-key="$secret_key" \
        --area="$area" \
        --endpoint="$endpoint"
else
    echo "Rank $RANK finished training but will not save or upload the model."
fi