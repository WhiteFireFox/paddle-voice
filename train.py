get_ipython().system('pip install paddlex -i https://mirror.baidu.com/pypi/simple')

#开始模型的训练

# 设置使用0号GPU卡
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

# 图像预处理+数据增强
from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=500, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=500, interp='CUBIC'),
    transforms.Normalize(),
])

# 数据迭代器的定义
train_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)

# 开始训练
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.0001,
    warmup_steps = 500,
    lr_decay_epochs=[50, 170],
    save_interval_epochs=10,
    save_dir='output/yolov3_darknet53')

# 开始预测
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('test.jpg')
b,g,r = cv2.split(img1)
img1 = cv2.merge([r,g,b])
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img1)

#加载模型
image_name = 'test.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='PrePicture')

img2 = cv2.imread('PrePicture/visualize_test.jpg')
b,g,r = cv2.split(img2)
img2 = cv2.merge([r,g,b])
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img2)
