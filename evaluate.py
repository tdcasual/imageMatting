import argparse
import datetime
import os
import wraptrain
from wraptrain import MODNet, ModNetImageGenerator, init_model  # 假设ModNetImageGenerator继承自ModNetImageGeneratorParentClass，并且evaluate方法在父类中定义

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptpath', type=str, default=None,required=True, help='Path to the checkpoint file.')
    parser.add_argument('--fg_path', type=str, required=True, help='Foreground data path.')
    parser.add_argument('--matte_path', type=str, required=True, help='Matte data path.')
    
    args = parser.parse_args()
    
    # 初始化模型并加载权重
    model = init_model(MODNet(),args.ckptpath)

    # 创建数据生成器
    files = wraptrain.create_dataframe(args.fg_path, args.matte_path)
    print(files)
    image_generator = ModNetImageGenerator(files, model)  # 如果ModNetImageGenerator是子类，这里可以直接使用子类名称

    # 运行评估并保存结果
    evaluation_result = image_generator.evaluate([1,2,3])  # 若evaluate无需额外参数，则直接调用

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = "result_{}.txt".format(timestamp)
    with open(result_file, 'w') as f:
        f.write(str(evaluation_result))

if __name__ == "__main__":
    main()