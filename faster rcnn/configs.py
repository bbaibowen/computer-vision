

# 预测图片存放位置
PREDICT_IMG_DATA_PATH = './experiments/experiment1/data/images'

CLASSES = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# 是否需要将结果写入txt
TXT_RESULT_WANTED = False

# txt结果保存路径
TXT_RESULT_PATH = './experiments/experiment1/result/txt_result'

# 是否可视化结果
IS_VISIBLE = True

# 结果可视化图保存路径
IMAGE_RESULT_PATH = './experiments/experiment1/result/image_result'

# 模型参数保存路径
CHECKPOINTS_PATH = './experiments/experiment1/checkpoints'

# 训练迭代轮数
MAX_EPOCH = 5

# 图片归一化后的最大高度与宽度

RESIZED_IMAGE_SIZE = [600,800]
# 训练集中较难识别的目标是否参与训练
USE_DIFFICULT = False

# anchor的放大倍数，原始尺寸为16*16
ANCHOR_SCALES = [8, 16, 32]

# anchor的长宽比
ANCHOR_RATIOS = [0.5, 1.0, 2.0]

# 分类阶段输入的rois数目
BATCH_SIZE = 128

# 非背景比例
FG_FRACTION = 0.25

# 训练时，若当前ROI对应的概率大于0.5，认为它是前景
FG_THRESH = 0.5

# 类别确认阈值
CONF_THRESH = 0.7

# 区域检测阈值
NMS_THRESH = 0.3
