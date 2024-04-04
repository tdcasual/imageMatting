#载入所有需要使用的包
#from torch.cuda.amp import autocast, GradScaler
import torch
#from src.models.modnet import MODNet  # 假设这是自定义模型，需要保留
 # 如果这是自定义的训练迭代器，需要保留
from torch.utils.data import DataLoader
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections import OrderedDict

class OwnNetTrainer:
    def __init__(self, model, device='cuda', ckpt_path=None):
        self.model =  torch.nn.DataParallel(model).cuda()
        self.device = torch.device(device)
        # 仅当检查点路径存在时尝试加载权重
        #只加载了权重，没有加载别的可以继续训练的信息。
        if ckpt_path is not None :
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"The provided checkpoint path '{ckpt_path}' does not exist.")
            if torch.cuda.is_available():
                weights = torch.load(ckpt_path)
            else:
                # 如果 CUDA 不可用，将权重映射到 CPU
                weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'model_state_dict' in weights.keys():
                weights = weights['model_state_dict']
            self.model.load_state_dict(weights)
            
            print(f"Loaded checkpoint from {ckpt_path}.")
        self.model.train()  # 将训练模式设置移至此处，避免每次初始化模型后都需要调用initialize_model方法

    def train(self, dataset, model_name="OwnNet", model_save_path="./model.pth",
              batch_size=4, learning_rate=0.01, epochs=40, step_size=0.25, gamma=0.1,
              update_freq=100, checkpoint_freq=5, checkpoint_dir='checkpoints', load_or_not=True):
        #scaler = GradScaler()
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size * epochs), gamma=gamma)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(step_size * epochs)), gamma=gamma)

        
        model_config = "_lr_{}_bs_{}".format(learning_rate, batch_size) + "_" + datetime.now().strftime('%m_%d_%H_%M')
        self.writer = SummaryWriter('runs/' + model_name + model_config)

        start_epoch = 0
        
        if load_or_not and os.path.exists(checkpoint_dir):
            checkpoint_files = os.listdir(checkpoint_dir)
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda cp: os.path.getmtime(os.path.join(checkpoint_dir, cp)))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Loaded checkpoint from {checkpoint_path}. Starting from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(train_dataloader):

                # 将数据转移到指定设备
                image, trimap, gt_matte = [item.to(self.device) for item in (image, trimap, gt_matte)]

                # 获取模型的预测matte
                _,_, pre_matte  = self.model(image)

                # 计算损失，这里以MSE为例
                matte_loss = torch.nn.functional.mse_loss(pre_matte, gt_matte)
                # 清除之前的梯度
                optimizer.zero_grad()

                # 反向传播
                matte_loss.backward()

                # 更新模型参数
                optimizer.step()

                # 记录损失
                self.writer.add_scalar('matte_loss', matte_loss.item(), epoch * len(train_dataloader) + idx)

            # 每个epoch后更新学习率
            lr_scheduler.step()

            
            # 定期保存检查点
            if (epoch + 1) % checkpoint_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # 在所有训练完成后，保存最终的模型状态
        torch.save(self.model.state_dict(), model_save_path) 
        
    def train_with_key(
        self,
        dataset,
        model_name: str = "OwnKeyNet",
        model_save_path: str = "./keymodel.pth",
        batch_size: int = 4,
        learning_rate: float = 0.01,
        epochs: int = 40,
        step_size: float = 0.25,
        gamma: float = 0.1,
        update_freq: int = 100,
        checkpoint_freq: int = 5,
        checkpoint_dir: str = "keycheckpoints",
        load_or_not: bool = True,
    ) -> None:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(step_size * epochs)), gamma=gamma)

        model_config = "_lr_{}_bs_{}".format(learning_rate, batch_size).replace(":", "_") + "_" + datetime.now().strftime('%m%d%H%M')
        self.writer = SummaryWriter('runs/' + model_name + model_config)

        start_epoch = 0

        try:
            if load_or_not and os.path.exists(checkpoint_dir):
                checkpoint_files = os.listdir(checkpoint_dir)
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda cp: os.path.getmtime(os.path.join(checkpoint_dir, cp)))
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Loaded checkpoint from {checkpoint_path}. Starting from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

        for epoch in range(start_epoch, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(train_dataloader):
                image, keypoint, gt_matte = [item.to(self.device) for item in (image, trimap, gt_matte)]
                _, key, pre_matte = self.model(image)
                matte_loss = nn.functional.mse_loss(pre_matte, gt_matte)
                keypoint_loss = nn.functional.mse_loss(key, keypoint)
                total_loss = matte_loss * 2 + keypoint_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 添加各项损失到TensorBoard
                self.writer.add_scalar('matte_loss', matte_loss.item(), epoch * len(train_dataloader) + idx)
                self.writer.add_scalar('keypoint_loss', keypoint_loss.item(), epoch * len(train_dataloader) + idx)
                self.writer.add_scalar('total_loss', total_loss.item(), epoch * len(train_dataloader) + idx)

            lr_scheduler.step()

            if (epoch + 1) % checkpoint_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        torch.save(self.model.state_dict(), model_save_path)

    def get_model(self):
        return self.model
    

if __name__=="__main__":
    from wraptrain import ReadImage,OriginModNetDataLoader,ImageMatteLoader,ModNetImageGenerator,NetTrainer
    from src.models.finitenet import FiniteNet
    base_path = "/mnt/data/Train/"
    fg = base_path+"FG"
    matte= base_path+"Alpha"
    files = ReadImage(fg,matte).read_same_names()
    begin = FiniteNet()
    all_data = OriginModNetDataLoader(files,[512,512])
    trainer = OwnNetTrainer(begin)
    trainer.train(all_data,batch_size=3,epochs=1,checkpoint_dir="owncheckpoint")
    #five = FiniteNet()
    #from collections import OrderedDict
    # 加载模型权重
    #state_dict = torch.load("stupid.pth")

    # 修改键名以适应当前模型架构
    #new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
        #name = k[7:] if k.startswith('module.') else k  # 移除`module.`前缀
        #new_state_dict[name] = v

    # 加载调整后的状态字典
    #five.load_state_dict(new_state_dict)
