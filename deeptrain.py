import torch
import deepspeed
import wraptrain
from wraptrain import MODNet, OriginModNetDataLoader, init_model, train_modnet
from torch.utils.data import DataLoader
from src.trainer import supervised_training_iter
import torchvision.transforms as transforms
import torch.distributed as dist

# 引入必要的模块
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



# ... 其他代码不变 ...

# 在deepspeed_train_modnet函数中，删除对"Dataloader.shuffle=True"的设定
# 因为现在由DistributedSampler负责数据的随机打乱

# ... 其他代码不变 ...



# ... 其他代码 ...
# 配置DeepSpeed
deepspeed_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": False
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


transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
def deepspeed_train_modnet(all_data, model, epochs=100, ckpt_path=None, deepspeed_config=deepspeed_config):
    # 确保在主进程中设置分布式环境

    
    # 获取全局世界大小和当前rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 创建DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(all_data, num_replicas=world_size, rank=rank)

    # 更新DataLoader
    dataloader = DataLoader(all_data, 
                            batch_size=deepspeed_config["train_batch_size"] // world_size, 
                            shuffle=False, # 由于DistributedSampler会处理，这里不需要shuffle
                            sampler=sampler,
                            num_workers=num_workers, 
                            pin_memory=True)

    # 初始化模型
    model = init_model(model, ckpt_path=ckpt_path)
    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )

    # 开始训练
    for epoch in range(epochs):
        model.train()
        for batch_idx, (image, trimap, gt_matte) in enumerate(dataloader):
            # 如果需要，这里可以添加 .cuda() 和 .half() 的操作
            # ...

            # 训练迭代
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                model, optimizer, image, trimap, gt_matte,
                semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0)
            
    # ... 其他代码不变 ...

if __name__ =="__main__":
    # 确保已经设置了分布式环境（在主进程中）
    import os
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0:
        dist.init_process_group(backend='nccl')

    base_path = "/mnt/data/Train/"
    fg = base_path+"FG"
    matte= base_path+"Alpha"
    files = wraptrain.create_dataframe(fg, matte)
    all_data = OriginModNetDataLoader(files, resize_dim=[512, 512], transform=transformer)
    

    # 初始化模型
    model = MODNet()

    # 调用deepspeed进行训练
    deepspeed_train_modnet(all_data, model, epochs=100, ckpt_path="pretrained/modnet_photographic_portrait_matting.ckpt")
