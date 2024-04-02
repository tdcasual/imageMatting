import pytest
from unittest.mock import Mock, patch
from src.models.finitenet import FiniteNet  # 假设路径和实际项目相匹配
from wrapown import OwnNetTrainer
from torch.utils.data import Dataset

# 创建一个模拟的数据集
class MockDataset(Dataset):
    def __len__(self):
        return 10  # 假设有10个样本

    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), torch.zeros(224, 224), torch.zeros(224, 224)  # 模拟图像数据和标签

@pytest.fixture
def mock_dataset():
    return MockDataset()

@patch('src.trainer.os.path.exists')
@patch('src.trainer.os.makedirs')
@patch('src.trainer.torch.save')
def test_train(mock_torch_save, mock_os_makedirs, mock_os_path_exists, mock_dataset):
    # 设定模拟返回值
    mock_os_path_exists.return_value = True

    model = MODNet()  # 假设模型初始化
    trainer = OwnNetTrainer(model)

    # 调用训练方法，这里可以根据实际情况减少epochs来加速测试
    trainer.train(dataset=mock_dataset, epochs=1, batch_size=2, learning_rate=0.01, model_save_path="./model_test.pth")

    # 验证是否创建了必要的目录
    mock_os_makedirs.assert_called()

    # 验证模型是否保存
    mock_torch_save.assert_called()

    # 这里可以添加更多的assert语句来验证学习率更新、检查点保存等逻辑
