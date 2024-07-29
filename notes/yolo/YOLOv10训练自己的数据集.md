# YOLOv10训练自己的数据集
## 阿里云GPU算力平台
[PAI_DSW_人工智能_深度学习_算法开发_Notebook_可视建模_开发环境_阿里云](https://www.aliyun.com/activity/bigdata/pai/dsw?spm=5176.21213303.J_qCOwPWspKEuWcmp8qiZNQ.21.3fea2f3docwKw1&scm=20140722.S_card%40%40%E5%95%86%E5%93%81%40%401892983.S_card0.ID_card%40%40%E5%95%86%E5%93%81%40%401892983-RL_pai~DAS~dsw-LOC_search~UND~card~UND~item-OR_ser-V_3-RE_cardOld-P0_0)
实例搭建：
①、进入网站后，点击管理控制台。
![image.png](https://img-blog.csdnimg.cn/img_convert/47cc8659b19cb7b979a790b15d2dec4e.png)
②、选择交互式建模（DSW），并点击 新建实例
![image.png](https://img-blog.csdnimg.cn/img_convert/f23029875b436827d5730c5b4366cded.png)
③、填写信息 及 选择相关配置，选择后点击下方的确定。
![image.png](https://img-blog.csdnimg.cn/img_convert/77962b2f0ebbf4d2ee3bb4bc6fcdbdfe.png)
④、点击实例右侧的 打开
![image.png](https://img-blog.csdnimg.cn/img_convert/2337ecf34f79da1b56d49816002f4be3.png)
⑤、通过命令行 把yolov10代码拉取下来即可进行后续操作。
![image.png](https://img-blog.csdnimg.cn/img_convert/3380eb267331d1bd6fb0dc5a7799939a.png)
## 项目克隆和安装  
### 克隆YOLOv10并安装
克隆项目到本机
网址: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
```bash
cd /mnt/workspace
git clone https://github.com/THU-MIG/yolov10.git
```
把requirements.txt中的==改为>=
执行
```bash
cd yolov10
pip install -r requirements.txt
pip install -e .
```
### 下载预训练权重文件
下载yolov10s.pt权重文件，并上传放置在/mnt/workspace/yolov10/ultralytics/weights文件夹下
权重地址：[https://github.com/THU-MIG/yolov10/releases](https://github.com/THU-MIG/yolov10/releases)
```bash
mkdir weights
```
### 安装测试
下面这些操作均在 /mnt/workspace/yolov10路径下执行
测试图片：
```bash
yolo predict model=/mnt/workspace/yolov10/ultralytics/weights/yolov10s.pt source=/mnt/workspace/yolov10/ultralytics/assets/bus.jpg
```
批量预测图片：
```bash
yolo predict model=/mnt/workspace/yolov10/ultralytics/weights/yolov10s.pt source=/mnt/workspace/yolov10/ultralytics/assets
```
预测图片并存储推理结果
```bash
yolo predict model=/mnt/workspace/yolov10/ultralytics/weights/yolov10s.pt source=/mnt/workspace/yolov10/ultralytics/assets/bus.jpg save_txt
```
预测摄像头：（远程算力用不了，本地可以使用)
```bash
yolo predict model=/mnt/workspace/yolov10/ultralytics/weights/yolov10s.pt 
source=0 show
```

## 标注自己的数据集
###  1、安装图像标注工具labelImg  
网址：[https://github.com/HumanSignal/labelImg](https://github.com/HumanSignal/labelImg)
下载后得到文件labelImg-master.zip 
解压：D:\labelImg-master 
建议使用Anaconda安装
以管理员身份运行Anaconda Prompt并到labelImg-master目录下执行命令 
在conda虚拟环境下执行:  
```bash
# 安装依赖
conda install pyqt=5
conda install -c anaconda lxml
pip install pyqt5-tools
pyrcc5 -o libs/resources.py resources.qrc

# 启动labelImg
python labelImg.py
```
###  2、添加自定义类别 
 修改文件labelImg-master/data/predefined_classes.txt  
```latex
ball
messi
trophy
```
###  3、使用labelImg进行图像标注  
 用labelImg标注生成PASCAL VOC格式的xml标记文件。例如：  
![image.png](https://img-blog.csdnimg.cn/img_convert/2d3c153334d945210a577cc400319ae8.png)
 width =1000   height = 654  
 PASCAL VOC标记文件如下：
![image.png](https://img-blog.csdnimg.cn/img_convert/d8e7fe9f67337c31ed79aac454603c87.png)
 也可以直接生成YOLO格式的txt标记文件如下：
```latex
class_id       x         y           w      h  
2           0.295000   0.495413 0.216000 0.926606
```
 x = x_center/width = 295/1000 = 0.2950 
y = y_center/height = 324/654 = 0.4954 
w = (xmax - xmin)/width = 216/1000 = 0.2160 
h = (ymax - ymin)/height = 606/654 = 0.9266 
class_id: 类别的id编号 
x: 目标的中心点x坐标（横向）/图片总宽度 
y: 目标的中心的y坐标（纵向）/图片总高度 
w: 目标框的宽带/图片总宽度 
h: 目标框的高度/图片总高度 
可以用python代码实现两种标记格式的转换：  
```python
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
```
box[0]：xmin 
box[1]:  xmax 
box[2]:  ymin 
box[3]:  ymax  
## 准备自己的数据集
### 1、下载项目文件
百度网盘下载链接
链接：[VOCdevkit_bm.zip](https://pan.baidu.com/s/1s61gc6DGYNjzfxNqu6NLaA?pwd=c4db) 
链接：[testfiles.zip](https://pan.baidu.com/s/10QJvP-4DWvxroWkd9FH5yg?pwd=wjkb) 
链接：[prepare_data.py](https://pan.baidu.com/s/1UeYHTvEdVCOpATHmftALtg?pwd=b0zj) 

- VOCdevkit_bm.zip (上传到/mnt/workspace/yolov10/ultralytics目录下并解压)
- testfiles.zip (上传到/mnt/workspace/yolov10/ultralytics目录下并解压)
- prepare_data.py (上传到/mnt/workspace/yolov10/ultralytics目录下)

unzip VOCdevkit_bm.zip
unzip testfiles.zip
### 2、解压建立或自行建立数据集  
使用PASCAL VOC数据集的目录结构:
建立文件夹层次为 VOCdevkit / VOC2007
VOC2007下面建立两个文件夹：Annotations，JPEGImages
JPEGImages放所有的数据集图片；Annotations放所有的xml标记文件；
### 3、划分训练集和验证集
/mnt/workspace/yolov10/ultralytics路径下执行python脚本：
```python
python prepare_data.py
```

- 注意：prepare_data脚本中的 classes=["ball","messi"]要根据自己的数据集类别做相应的修改
   - 在VOCdevkit目录下生成了images和labels文件夹
   - images文件夹下有train和val文件夹，分别放置训练集和验证集图片；
   - labels文件夹有train和val文件夹，分别放置训练集和验证集标签（yolo格式）
   - /mnt/workspace/yolov10/ultralytics下生成了两个文件yolov10_train.txt和yolov10_val.txt。

yolov10_train.txt和yolov10_val.txt分别给出了训练图片文件和验证图片文件的列表，含有每个图片的路径和文件名
### 4、修改配置文件
修改文件cfg/datasets/VOC.yaml为VOC-bm.yaml
```python
path: /mnt/workspace/yolov10/ultralytics/VOCdevkit  
train: # train images (relative to 'path') 16551 images
 - images/train 
val: # val images (relative to 'path') 4952 images
 - images/val
test: # test images (optional)
# Classes
names:
 0: ball
 1: messi
```
## 训练自己的数据集
**训练命令**
/mnt/workspace/yolov10路径下执行
```bash
yolo detect train data=/mnt/workspace/yolov10/ultralytics/cfg/datasets/VOC-bm.yaml model=/mnt/workspace/yolov10/ultralytics/weights/yolov10s.pt epochs=500 
imgsz=640 batch=16 patience=500 workers=4
```
命令参数说明：[https://docs.ultralytics.com/modes/train/#arguments](https://docs.ultralytics.com/modes/train/#arguments)
注意：如果出现显存溢出，可减小batch size

**断点续训**
```bash
olo detect train data=/mnt/workspace/yolov10/ultralytics/cfg/datasets/VOC-bm.yaml model=/mnt/workspace/ultralytics/runs/detect/train/weights/best.pt 
epochs=500 imgsz=640 batch=16 patience=500 workers=4 resume
```
注意：应使用断点时保存的相应文件下的best.pt权重文件进行断点续训
注意：如果出现显存溢出，可减小batch size

**训练结果的查看**
查看/mnt/workspace/yolov10/runs/detect/train目录下的文件

### 测试训练出的网络模型
测试图片
```bash
yolo predict model=/mnt/workspace/yolov10/runs/detect/train/weights/best.pt source=/mnt/workspace/yolov10/ultralytics/testfiles/img1.jpg conf=0.5
```
批量测试图片：
```bash
yolo predict model=/mnt/workspace/yolov10/runs/detect/train/weights/best.pt source=/mnt/workspace/yolov10/ultralytics/testfiles conf=0.5
```
测试视频：
```bash
yolo predict model=/mnt/workspace/yolov10/runs/detect/train/weights/best.pt source=/mnt/workspace/yolov10/ultralytics/testfiles/messi.mp4 show
```
性能统计
```bash
yolo val model=/mnt/workspace/yolov10/runs/detect/train/weights/best.pt data=/mnt/workspace/yolov10/ultralytics/cfg/datasets/VOC-bm.yaml conf=0.25
```
注意：性能评估结果和conf阈值的设置有关
