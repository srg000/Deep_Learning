# 深度学习在图像处理中的应用

* 图像分类
  > 图片分类的数据集一般有两种实现方式：
  > 1. 用文件夹名来表示图片的类别，同一类图片放到同一个文件夹下。 最后使用datasets.ImageFolder来处理我们的数据集，得到的数据集就包括了标签信息，我们不需要自己重写 DataSet类
  > 2. 训练集和测试集文件夹内存放所有类别的图片，需要再编写 train.txt 和 val.txt文件来标注每张图片及其对应的类别标签。并且还需要我们自己重写 DataSet类来处理图片信息和标签信息。
  * LeNet（已完成）
 
  * AlexNet（已完成）

  * VggNet（已完成）

  * GoogLeNet（已完成）
 
  * ResNet（已完成）

  * Vision Transformer（已完成）

  * Swin Transformer

  * RepVGG

  * ConvNeXt

  * MobileViT

* 目标检测
  * Faster-RCNN/FPN（已完成）

  * SSD/RetinaNet

  * [YOLOv10（已完成）](https://github.com/srg000/yolov10)
    
  * YOLOv8

  * FCOS

* 语义分割 
  * FCN 

  * DeepLabV3 

  * LR-ASPP

  * U-Net

  * U2Net

* 实例分割
  * Mask R-CNN

* 关键点检测
  * HRNet
---

## 所需环境
* Anaconda3（建议使用）
* python 3.8/3.9/3.10
* pycharm (IDE)
* pytorch 1.13.0 (pip package)
* torchvision 0.14.0 (pip package)
