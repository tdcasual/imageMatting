import torch
import deepspeed
import wraptrain
from wraptrain import MODNet, OriginModNetDataLoader, init_model, train_modnet
from torch.utils.data import DataLoader
from src.trainer import supervised_training_iter


# 配置DeepSpeed
deepspeed_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 100
        }
    }
}

def deepspeed_train_modnet(all_data, model, epochs=100, ckpt_path=None, deepspeed_config=deepspeed_config):
    # 初始化模型
    model = init_model(model, ckpt_path=ckpt_path)
    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )

    # 创建DataLoader
    dataloader = DataLoader(all_data, batch_size=1, shuffle=True)  # batch_size设为1，DeepSpeed会自动调整

    # 开始训练
    for epoch in range(epochs):
        model.train()
        for batch_idx, (image, trimap, gt_matte) in enumerate(dataloader):
            image, trimap, gt_matte = image.cuda(), trimap.cuda(), gt_matte.cuda()
        # 使用supervised_training_iter来替代之前的calculate_loss
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                model, optimizer, image, trimap, gt_matte,
                semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0)
        # 可以根据需要调整semantic_scale, detail_scale和matte_scale的值
        # 这里不需要显式地调用model.backward(loss)和model.step()，
        # 因为supervised_training_iter函数内部已经处理了这些步骤

if __name__ =="__main__":
    # 使用wraptrain中的函数加载数据
    files = wraptrain.create_dataframe("/path/to/fg", "/path/to/alpha")
    all_data = OriginModNetDataLoader(files, resize_dim=[512, 512], transform=wraptrain.transformer)

    # 初始化模型
    model = MODNet()

    # 调用deepspeed进行训练
    deepspeed_train_modnet(all_data, model, epochs=100, ckpt_path="pretrained/modnet_photographic_portrait_matting.ckpt")
