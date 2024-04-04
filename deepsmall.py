import torch
import deepspeed
from wraptrain import ReadImage,OriginModNetDataLoader,ImageMatteLoader,ModNetImageGenerator,NetTrainer
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

from src.models.finitenetsmall import FiniteNet
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



# ... 其他代码不变 ...

# 在deepspeed_train_modnet函数中，删除对"Dataloader.shuffle=True"的设定
# 因为现在由DistributedSampler负责数据的随机打乱

# ... 其他代码不变 ...

global_summary_writer = None

def setup_tensorboard(rank):
    # 只在 rank 为 0 的进程中创建 SummaryWriter
    global global_summary_writer
    # 获取当前日期和时间
    current_time = datetime.now()
    # 格式化日期和时间，例如："YYYY-MM-DD_HH-MM-SS"
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M')

    # 将格式化的时间字符串添加到 log_dir 路径中
    log_dir = f"/mnt/data/runs/deepspeed_train_{formatted_time}"

    
    # 引入必要的模块
    if rank == 0:
        global_summary_writer = SummaryWriter(log_dir=log_dir)

def write_to_tensorboard(epoch, batch_idx, losses, rank, datalength):
    # 将训练损失写入 TensorBoard
    if rank == 0:
        global_summary_writer.add_scalar('Training/Loss', losses, epoch * datalength + batch_idx)

# ... 其他代码 ...
# 配置DeepSpeed


def deepspeed_train_FiniteNet(all_data, model, deepspeed_config,epochs=100, ckpt_path=None, ):
    # 确保在主进程中设置分布式环境
    setup_tensorboard(dist.get_rank())

    
    # 获取全局世界大小和当前rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 创建DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(all_data, num_replicas=world_size, rank=rank)
    num_workers = min(mp.cpu_count(), 4)  # 假设我们使用不超过4个workers
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

    for epoch in range(epochs):
        print(f"now begin epoch at {datetime.now()}")
        model.train()
        

        # Initialize the gradient scaler for automatic mixed precision
        scaler = GradScaler()


        
        for idx, (image, keypoint, gt_matte) in enumerate(dataloader):
            with autocast():
                # 解析batch_data，通常它会包含一个字典或元组
                #image, trimap, gt_matte = batch_data['image'], batch_data['trimap'], batch_data['gt_matte']

                    # 将数据转移到指定设备
                image, keypoint, gt_matte = [item.cuda() for item in (image, keypoint, gt_matte)]

                    # 获取模型的预测matte
                _, key, pre_matte = model(image)

                    # 计算损失，这里以MSE为例
                matte_loss = torch.nn.functional.mse_loss(pre_matte, gt_matte)
                keypoint_loss = nn.functional.mse_loss(key, keypoint)
                    # 使用DeepSpeed进行反向传播和优化步骤
                total_loss = matte_loss * 2 + keypoint_loss
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()


                # 记录损失
                #global_summary_writer.add_scalar('matte_loss', matte_loss.item(), epoch * len(dataloader) + idx)
            write_to_tensorboard(epoch, idx, total_loss.item(), rank, len(dataloader))  # 修

            # 在每个epoch后更新学习率（如果lr_scheduler支持DeepSpeed，则直接调用step()；否则可能需要特殊处理）
            scheduler.step()

    if dist.get_rank() == 0:
        global_summary_writer.close()
def load_deepspeed_config(config_file_path):
    with open(config_file_path, 'r') as f:
        deepspeed_config = json.load(f)
    return deepspeed_config


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
    parser.add_argument('--deepspeed_config', default="deepspeed.config", type=str, help='Path to the DeepSpeed configuration file.')
    parser.add_argument('--save_path', default="model.pth", type=str, help='Path to the checkpoint file.')
    args = parser.parse_args()
    print(f"checkpoint will be saved on {args.save_path}")
    if args.ckptpath == "None":
        args.ckptpath = None
    deepspeed_config = load_deepspeed_config(args.deepspeed_config)
    default_epoch = 50 if args.epoch == 0 else args.epoch
    print(f"now, batch_size is {args.batch_size}")
    if args.batch_size != 0:
        deepspeed_config["train_batch_size"]= args.batch_size

    print(deepspeed_config)
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
    all_data = OriginModNetDataLoader(files, resize_dim=[512, 512],hot_path=base_path+"newHeatmaps.pt")

    # 初始化模型
    model = FiniteNet()

    # 根据命令行参数设置ckpt_path
    ckpt_path = args.ckptpath

    # 调用deepspeed进行训练
    deepspeed_train_FiniteNet(all_data, model,deepspeed_config, epochs=default_epoch, ckpt_path=ckpt_path)

    # 获取模型权重
    #model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()


    # 训练完成后，在 rank 0 的 worker 上保存模型
    if dist.get_rank() == 0:
        # 获取模型的状态字典
        model_state_dict = model.state_dict()

        # 检查键名是否已经包含`module.`前缀
        # 如果没有，则手动添加
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if not k.startswith('module.'):
                new_key = 'module.' + k  # 添加`module.`前缀
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 保存带有`module.`前缀的模型权重
        torch.save(new_state_dict, args.save_path)


    # 通知所有工作进程训练完成，以便优雅地退出
    dist.barrier()

