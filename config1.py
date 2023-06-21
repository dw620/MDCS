# 数据集的类别
NUM_CLASSES = 7

# 训练时batch的大小
BATCH_SIZE = 32

# 训练轮数
NUM_EPOCHS = 200

# 训练完成，精度和损失文件的保存路径,默认保存在trained_models下
TRAINED_MODEL = 'F:/Code/RANet-Submitted-to-IEEE-JSTARS-main/trained_models/data_record.pth'

# 数据集的存放位置  WHURSDataset  UC Merced Land Use Dataset  SIRI-WHU  NWPU-RESISC45  OPTIMAL-31  PatternNet  RSSCN7  AID
TRAIN_DATASET_DIR = 'F:/Datasets/dataset/RSSCN7/55/train'
VALID_DATASET_DIR = 'F:/Datasets/dataset/RSSCN7/55/val'

# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = 'F:/Datasets/dataset/RSSCN7/55/val/'  # 数据集的根目录
    model = 'RANet'  # ResNet18 ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    freeze = True  # 是否冻结卷基层
    seed = 1000  # 固定随机种子
    num_workers = 4  # DataLoader 中的多线程数量
    num_classes = 7  # 分类类别数
    # num_epochs = 30
    batch_size = 8
    lr = 0.01  # 初始lr
    width = 256  # 输入图像的宽
    height = 256  # 输入图像的高
    iter_smooth = 105  # 打印&记录log的频率

    resume = False

    # checkpoint = 'ResNet152.pth' # 训练完成的模型名
    checkpoint = 'new_model_7_98.91_55.pth'

config = DefaultConfigs()
