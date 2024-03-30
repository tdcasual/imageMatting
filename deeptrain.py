import torch
import deepspeed
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from wraptrain import ReadImage,OriginModNetDataLoader,ImageMatteLoader,ModNetImageGenerator,NetTrainer
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

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
    "train_batch_size": 8,
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
    # 确保在主进程中设置分布式环境

    
    # 获取全局世界大小和当前rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 创建DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(all_data, num_replicas=world_size, rank=rank)
    num_workers = min(mp.cpu_count(), 1024)  # 假设我们使用不超过4个workers
    #num_workers=4
    # 更新DataLoader
    dataloader = DataLoader(all_data, 
                            batch_size=deepspeed_config["train_batch_size"] // world_size, 
                            shuffle=False, # 由于DistributedSampler会处理，这里不需要shuffle
                            sampler=sampler,
                            num_workers=num_workers, 
                            pin_memory=True)

    # 初始化模型
    model = NetTrainer(model, ckpt_path=ckpt_path).get_model()
    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )

    model.train()
    # 开始训练
    for epoch in range(epochs):
        

        # Initialize the gradient scaler for automatic mixed precision
        scaler = GradScaler()

        for batch_idx, (image, trimap, gt_matte) in enumerate(dataloader):
            # Resets gradients of the optimizer
            image,trimap,gt_matte = image.cuda().half(), trimap.cuda().half(), gt_matte.cuda().half()
            optimizer.zero_grad()

            # Automatic Mixed Precision block
            with autocast():
                # 训练迭代中的前向传播
                semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                    model, optimizer, image, trimap, gt_matte,
                    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0)

            # Scales the loss, calls backward to create scaled gradients, and step the optimizer
            # Unscales the gradients of optimizer's assigned params in-place before calling optimizer.step()
            scaler.scale(semantic_loss + detail_loss + matte_loss).backward()
            scaler.step(optimizer)
            scaler.update()


import argparse
import os

# 默认的fg和matte路径
default_fg_path = "/mnt/data/Train/FG"
default_matte_path = "/mnt/data/Train/Alpha"

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    # 解析命令行参数
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptpath', type=str, default=None, help='Path to the checkpoint file.')
    parser.add_argument('--fg_path', type=str, default=default_fg_path, help='Foreground data path (default: {})'.format(default_fg_path))
    parser.add_argument('--matte_path', type=str, default=default_matte_path, help='Matte data path (default: {})'.format(default_matte_path))
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training (default: 0).')  # 添加对 --local_rank 的支持
    parser.add_argument('--batch_size', type=int, default=0, help='Local batch_size for distributed training (default: 0).')  # 添加对 --local_rank 的支持
    parser.add_argument('--epoch', type=int, default=0, help='Local epoch for distributed training (default: 0).')  # 添加对 --local_rank 的支持
    args = parser.parse_args()

    default_epoch = 50 if args.epoch == 0 else args.epoch
    deepspeed_config["train_batch_size"]= 8 if args.batch_size == 0 else args.batch_size

    # 使用 args.local_rank 设置分布式环境
    #torch.cuda.set_device(args.local_rank)
    #torch.distributed.init_process_group(backend='nccl')

    #args = parser.parse_args()

    # 确保已经设置了分布式环境（在主进程中）
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0:
        dist.init_process_group(backend='nccl')

    # 使用命令行参数或默认值设置fg和matte路径
    fg = args.fg_path
    matte = args.matte_path
    #print(fg,matte)
    files = ReadImage(fg, matte).read_same_names()
    #print(files)
    all_data = OriginModNetDataLoader(files, resize_dim=[512, 512])

    # 初始化模型
    model = MODNet()

    # 根据命令行参数设置ckpt_path
    ckpt_path = args.ckptpath

    # 调用deepspeed进行训练
    deepspeed_train_modnet(all_data, model, epochs=default_epoch, ckpt_path=ckpt_path)

    # 获取模型权重
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()

    # 保存模型权重
    torch.save(model_state_dict, 'model.pth')