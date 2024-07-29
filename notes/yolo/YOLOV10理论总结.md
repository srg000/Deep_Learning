论文题目：YOLOv10: Real-Time End-to-End Object Detection
研究单位：清华大学
论文链接：[http://arxiv.org/abs/2405.14458](http://arxiv.org/abs/2405.14458)
代码链接：[https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

## v10论文解读
<a name="SxTM7"></a>
### yolo模型的不足
v10论文中作者指出了 当前yolo模型的不足： 

1. 依赖于非极大值抑制（NMS）进行后处理阻碍了YOLO模型的端到端部署（增加了推理过程的延迟）
2. YOLO模型中各个组件的设计缺乏全面和彻底的检查，导致了显著的**计算冗余**，限制了模型的能力，从而导致**效率不佳**，并有很大的性能提升潜力。  

针对这两个问题，论文从**后处理**和**模型架构**两方面进一步提高YOLO模型的性能效率边界。在后处理方面，作者提出  **无NMS训练的一致性双重分配策略**，模型架构方面， 引入了全面的**效率-准确性驱动的 YOLO模型设计策略**。

<a name="YjA5y"></a>
### v10的性能提升
![image.png](https://img-blog.csdnimg.cn/img_convert/1042d64c78f87eb4e4fcb8b03d422afc.png)

<a name="QkfPQ"></a>
### 改进方法
**一、用于无NMS训练的一致性双重分配(Consistent Dual Assignments for NMS free Training)**<br />具体问题：YOLO在训练过程中通常采用 一对 多标签分配策略，即一个真实物体对应多个正样本。尽管这种方法能提供丰富的监督信号，促进优化并获得较好的性能 ，但在 推理 时需要依赖NMS来选择最佳的正预测。这会降低推理速度，并使性能对NMS的超参数非常敏感，从而阻碍了YOLO的端到端部署。  （在训练时，需要使用一对多标签分配广撒网的形式来提高性能，但是推理时需要用nms这个包袱来选择最佳的正预测，这是作者想甩掉的）我们希望在推理的时候，能够做到一对一匹配来抑制冗余的预测。所以作者就提出了一种无 NMS训练策略，通过双重标签分配和一致匹配度量实现了高效且具有竞争力的性能  <br />![image.png](https://img-blog.csdnimg.cn/img_convert/7efc9a617fd38e3b0428b024776b2662.png)<br />下面，我对这幅图进行一些解释：

- 从图中可以看到，输入经过Backbone，然后经过PAN，但是它的头部和以前的不同，引入了双头结构，它有两个头：one-to-many Head 和 one-to-one Head，每个头中都需要做 分类任务就是确定物体的类别 和 回归任务也就是确定物体所在边界框的坐标。
- 在训练的时候，两个头同时优化，我们有一对多和一对一， 让backbone和neck享受到one-to-many分配带来的丰富监督信号。但是在推理时, 舍弃 one-to-many head, 仅使用 one-to-one head进行预测。这样就不需要使用NMS后处理, 也不会带来额外的推理开销  
- 下面是一个 一致性匹配度量，用于计算预测与实例之间的一致性，度量的指标是这个公式。它是几个部分的相乘。这个度量即考虑的分类得分，也考虑了回归的IOU

![image.png](https://img-blog.csdnimg.cn/img_convert/7371a2f77333f2be4d3c6140b02369e0.png)

- 现在我们有了Metric的定义，但是这两个头的 Metric是否是一样的呢？是不是采用同样的Metric来度量的呢？确切的说，其中的超参 ɑ和ß对于这两个头，是否一致呢？这就牵扯到是否一致性。作者建议使用一致性的度量。看一下右边这幅图性能的对比。橙色的是Consistent，蓝色的是Inconsistent，对于Top-1，Top-5，Top-10，它会对比两者的一致的频率。也就是说我用一对多的头和用一对一的头最后预测的是目标是不是同一个目标，如果一样的话，就在Frequency中进行计算。它的作用就是缩小两个分支的监督差异。
- 看一下这幅图的下方，论文中说：默认对于一对多的头，采用ɑ=0.5 和 ß=6。如果使用一致性参数的话，一对一的头，采用的也是ɑ=0.5 和 ß=6。如果使用不一致参数的话，一对一的头，采用的是ɑ=0.5 和 ß=2。

上面这些步骤就是一致性双重分配，是YOLOv10中提出的一种NMS-free训练策略, 旨在兼顾one to-many分配的丰富监督信号和one-to-one分配的高效率。  

**二、YOLOV8 中采用的生成 anchors的方式**<br />在V10中，因为只在预测的时候，使用多个候选框来预测真实物体，所以这个操作只在一对多标签分配中。<br /> 一对多标签分配 (One-to-many Label Assignment)是目标检测中的一种**动态标签分配策略**，它允许一个ground truth(GT)匹配到多个预测框(anchors)。这种分配方式在YOLO系列模型 中被广泛使用。传统的目标检测算法通常使用IoU阈值来确定正负样本, 即IoU大于某一阈值的anchor被视为正样本,反之为负样本。这种方式会导致每个GT只能匹配到一个anchor, 即一对一(one-to-one)匹配。而一对多标签分配采用一种更加灵活的方式。以YOLO系列为例, 它们使用一个matching score来衡量anchor和GT之间的匹配程度, 得分越高说明两者越匹配。score的计算考虑了三个因素:

1. 空间位置关系: anchor的中心点是否落在GT内部。
2. 尺度关系: anchor和GT的尺度比例是否接近。
3. 语义相似性: 使用分类预测分数来表示anchor和GT在语义上的相似程度。

通过这种综合考虑空间、尺度和语义信息的matching score, 每个GT可以选择数个高分anchor作<br />为其正样本, 而不局限于IoU最高的那一个。这带来了以下好处:

1. 一个目标可以匹配到多个尺度和位置的anchors, 提高召回率。
2. 增强了样本多样性,一个GT指导多个anchors学习, 从而获得更加丰富的监督信号。
3. matching更加鲁棒, 即便某个anchor没有最高IoU, 但语义相似度高也可能成为正样本。因此, 一对多标签分配可以在训练时为检测器提供更加充足且多样的监督信息, 有利于提升检测性能。但在推理时, 为了避免一个目标产生多个检测框, 需要用到NMS等后处理操作去除冗余检测结果。

**三、 全面的效率-准确性驱动模型设计(Holistic Efficiency-Accuracy Driven Model Design)**  
> 内在秩 (Intrinsic Rank)是评估卷积层冗余度的一个指标, 它反映了卷积核张量的有效维度。（用到了内在秩的概念，这里不做过多讲解） 。

![image.png](https://img-blog.csdnimg.cn/img_convert/9539708f82f129087c82278ace79efbe.png)<br />作者分析了YOLOv8各个阶段(stage)的卷积层的内在秩，发现深层阶段和大尺度模型的内在秩较低，说明存在一定冗余。有冗余就可以对相应的模块进行简化。 针对内在秩较低的阶段, 将原有的基础模块替换为更加轻量化的CIB模块, 从而在保持性能的同时降低计算开销, 提高参数利用效率  <br />从图b中可以看到CIB模块，有三层深度可分离卷积进行下采样和 两层 逐点卷积进行通道数增加，这样我们的成本和参数量都会相应的降低。

效率驱动型模型设计:

- 使用内在秩分析来识别和减少模型阶段的冗余，用更有效的结构代替复杂的块。
- 通过使用CIB的简化架构来减少计算开销。
- 将空间下采样和通道增加解耦来 减少计算成本和参数量。在常见的卷积神经网络中, 下采样通常使用stride=2的卷积层同时实现空间尺度的缩减(H×W → H/2 × W/2)和通道数的增加(C → 2C)  

精度驱动的模型设计:

- 通过增加深度阶段的感受野来增强模型能力，有选择地使用大核深度卷积来避免浅阶段的开销。
- 通过PSA划分特征并将自注意力应用于部分特征，结合有效的自注意力，降低计算复杂性和内存使用，同时增强全局表示学习。

① 采用大核深度卷积是 扩大感受野和增强模型能力的有效方式。然而，简单地在所有阶段采用大核卷积可能会对用于检测小物体的浅层特征造成污染。所以作者建议在CIB的深层阶段中使用大核深度卷积。具体来说，我们将 CIB中第二个3×3深度可分离卷积的核尺寸增大到7×7  <br /> ② 如图c所示。具体来说，我们在1×1 卷积后将特征在channel维度上上均匀分成两部分。我们仅将其中一部分送入由多头自注意力模块 （MHSA）和前馈网络（FFN）组成的NPSA模块。然后，将两部分特征拼接并通过一个1×1 卷积融合  

<a name="pwjq2"></a>
## v10网络架构
![image.png](https://img-blog.csdnimg.cn/img_convert/3ae2f67f14bceea7655b8b40b8df67e4.png)<br />![image.png](https://img-blog.csdnimg.cn/img_convert/114f34ddcf40f7b78f7b532dfdfc31ea.png)

<a name="wnkjg"></a>
## v10关键代码解析
<a name="Lfl0A"></a>
### YOLOv10-S模型的配置文件
 下面是YOLOv10-S模型的配置文件, 使用YAML格式定义了模型的架构和超参数。  
```python
# Parameters
 nc: 80 # number of classes
 scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
 # [depth, width, max_channels]
  s: [0.33, 0.50, 1024]
 backbone:
 # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, SCDown, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, SCDown, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2fCIB, [1024, True, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 1, PSA, [1024]] # 10
 # YOLOv8.0n head
 head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 16 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 19 (P4/16-medium)
  - [-1, 1, SCDown, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2fCIB, [1024, True, True]] # 22 (P5/32-large)
  - [[16, 19, 22], 1, v10Detect, [nc]] # Detect(P3, P4, P5)
```

1. nc: 80 - 模型需要检测的目标类别数为80。
2. scales - 定义了模型缩放的比例因子, 包括深度(depth)、宽度(width)和最大通道数(max_channels)。<br />这里 's' 表示small模型, 深度缩放0.33倍, 宽度缩放0.5倍, 最大通道数为1024。
3. backbone - 定义了模型的主干网络结构, 每一行代表一个模块。
   1. [-1, 1, Conv, [64, 3, 2]] 表示一个64输出通道、3x3卷积核、步长为2的卷积层。-1表示输入来<br />自上一层。
   2. C2f是一种残差块结构, SCDown是一种空间下采样和通道变换解耦的下采样模块, SPPF是空间<br />金字塔池化模块。
   3. backbone的输出为P3, P4, P5三个尺度的特征图, 对应的下采样率分别为8, 16, 32。
   4. PSA模块, 用于捕获全局依赖。
   5. C2fCIB是一种改进的残差块, 引入了高效的Compact Inverted Bottleneck结构。
4. head - 定义了检测头的结构。<br />使用nn.Upsample进行上采样, 与backbone的特征图concat后再次进行融合。<br />v10Detect是YOLOv10特有的检测层, 在3个尺度上预测目标的类别和位置。
5. 整个网络结构以CSPDarknet为backbone提取特征, 再通过FPN结构融合多尺度特征用于预测。

<a name="zzfmf"></a>
### SCDown
> yolov10-main\ultralytics\nn\modules\block.py  
> SCDown 类实现了一个两层卷积的网络模块。 首先利用点卷积调节通道维度，然后利用深度可分离卷积进行空间下采样。
> 这种结构在神经网络中常用于降低维度和提取特征，特别是在计算资源有限的情况下。  

```python
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))
```

- SCDown 继承自 nn.Module 。 
- __init__ 方法接收四个参数： 
   - c1 ：输入通道数。 
   - c2 ：输出通道数。 
   - k ：卷积核大小。 
   - s ：卷积步幅。 
- super().__init__() 调用父类的初始化方法。 
- self.cv1 是第一个卷积层，使用 Conv(c1, c2, 1, 1) 表示输入通道数为 c1 ，输出通道数为 c2，卷积核大小为1，步幅为1。这一步就是逐点卷积，实现了通道变化。
- self.cv2 是第二个卷积层，使用 Conv(c2, c2, k=k, s=s, g=c2, act=False) ，表示输入和 输出通道数都为 c2，卷积核大小为 k，步幅为 s，分组数为 c2，且不使用激活函数 ( act=False )。  这一步就是深度可分离卷积，实现了空间下采样。

<a name="xHlSs"></a>
###  PSA (Partial Self-Attention)  
> Attention 类实现了一个多头自注意力机制，用于计算输入特征的注意力得分，并进行加权求和。
> PSA 类结合了卷积层、自注意力层和前馈神经网络，旨在增强输入特征的表示能力。
> 通过这种结构，PSA 可以在卷积神经网络中应用自注意力机制，提高特征提取的效果。

![image.png](https://img-blog.csdnimg.cn/img_convert/ffc4535ba54157f77c771145506d0962.png)<br />**Attention类**
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```
初始化方法：

- dim ：输入通道数。
- num_heads ：注意力头的数量。
- attn_ratio ：缩放比例。
- 计算每个头的维度 self.head_dim 和键的维度 self.key_dim 。
- self.scale 用于缩放注意力得分。
- self.qkv 是一个卷积层，用于生成查询（Q）、键（K）和值（V）。
- self.proj 是一个卷积层，用于输出投影。
- self.pe 是一个卷积层，用于位置编码。

前向传播方法：

- 计算输入张量的形状 B, C, H, W 。
- 使用 self.qkv 生成查询、键和值，并拆分为 q, k, v 。
- 计算注意力得分，并通过 softmax 进行归一化。
- 计算加权值，并添加位置编码 self.pe 。
- 最后通过 self.proj 进行输出投影。

**PSA类**
```python
class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
```
初始化方法：

- c1 和 c2：输入和输出的通道数（这里要求它们相等）。
- e ：缩放比例。
- self.cv1 和 self.cv2 是卷积层。
- self.attn 是一个 Attention 层。
- self.ffn 是一个前馈神经网络（使用两个卷积层实现）。

前向传播方法：

- 使用 self.cv1 将输入 x 分为 a 和 b 两部分。
- 对 b 进行注意力计算并加上自身。
- 对 b 进行前馈神经网络计算并加上自身。
- 最后将 a 和 b 拼接起来，通过 self.cv2 进行输出。

<a name="qswic"></a>
###  CIB (Compact Inverted Bottleneck)  
![image.png](https://img-blog.csdnimg.cn/img_convert/be6a83d288c35875283bbba9e223c0fc.png)
```python
class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)
```

- c1 ：输入通道数。
- c2 ：输出通道数。
- shortcut ：是否使用快捷连接（残差连接）。
- e ：扩展系数，用于计算隐藏层的通道数。
- lk ：是否使用 RepVGGDW 代替标准的卷积层。
<a name="Soldj"></a>
###  C2fCIB  <br />![image.png](https://img-blog.csdnimg.cn/img_convert/0cc0d565c345546410658f1d7d85c97c.png)
```python
class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
```

<a name="vWCfa"></a>
###  v10Detect  
>  yolov10-main\ultralytics\nn\modules\head.py  
> 这个函数实现的功能就是 网络架构中的head部分，v10中head部分使用了一致性双重分配的策略。有一对一和一对多的头。

```python
class v10Detect(Detect):

    max_det = 300

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), \
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), \
                                                nn.Conv2d(c3, self.nc, 1)) for i, x in enumerate(ch))

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
    
    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.export:
            one2many = super().forward(x)

        if not self.training:
            one2one = self.inference(one2one)
            if not self.export:
                return {"one2many": one2many, "one2one": one2one}
            else:
                assert(self.max_det != -1)
                boxes, scores, labels = ops.v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
        else:
            return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        super().bias_init()
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
```
 v10Detect类继承自 Detect 类，添加了多路特征提取和处理的功能。通过不同的卷积层序列和特征提取方法，该类可以在导出和训练模式下灵活地处理特征，并提供特定格式的输出。  

<a name="lzxGG"></a>
###  YOLOv10DetectionPredictor  
>  yolov10-main\ultralytics\models\yolov10\predict.py  
>  该类的目的是处理 YOLOv10 模型的预测结果，也就是进行一些后处理操作。

```python
class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # 处理预测结果的形状
        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1]-4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        # 应用置信度阈值和类别筛选
        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        # 处理输入图像格式（从 torch.Tensor 转换为 NumPy 数组）
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 调整边界框尺寸并生成结果
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
```

1. preds : 模型的预测结果。 img : 输入图像。 orig_imgs : 原始输入图像。  
2. 它处理预测结果的格式和形状，应用置信度阈值和类别筛选，并调整边界框的尺寸以匹配原始图像。  
3. 训练时，把特征送入v10Detect 之后，把通过一对一和一对多的头预测的结果，送入v10DetectLoss ，通过计算损失来反向传播更新参数，训练过程中不进行预测结果的处理，也就是不进行后处理，因为后处理是要把预测的结果进行处理，要展示出来的，而训练阶段只是学习的阶段更新参数的阶段。
4. 从 predict.py 文件中可以看出，在推理阶段代码的设计确实舍弃了 one-to-many head，仅使用 one-to-one head 进行预测。不需要NMS后处理，就没有额外的开销。
5. 在推理之后，预测到的图片之后，才进行YOLOv10DetectionPredictor ，才进行后处理
<a name="zFdy6"></a>
###  v10DetectLoss  
>  yolov10-main\ultralytics\utils\loss.py  

```python
class v10DetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)
    
    def __call__(self, preds, batch):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], torch.cat((loss_one2many[1], loss_one2one[1]))
```

1. 这个代码定义了一个名为 v10DetectLoss 的类，用于计算一种新的损失函数，结合了两种不同的检测损失。
2. self.one2many: 使用 v8DetectionLoss 类创建一个损失实例，tal_topk 参数设置为10。这意味着这个损失函数会考虑前 10 个预测结果。self.one2one : 使用 v8DetectionLoss 类创建另一个损失实例， tal_topk 参数设置为 1。这意味着这个损失函数只考虑最顶层的一个预测结果。  
3.  preds : 包含预测结果的字典。字典中有两个键，分别是 one2many 和 one2one ，对应两种不同的预测结果。 batch : 一个批次的数据，用于计算损失。  

在v8DetectionLoss中__call__方法中有下面这段代码：loss.sum() * batch_size计算了整个批次的总损失，并且这个总损失将用于反向传播。而loss.detach()提供了损失的一个副本，该副本不包含梯度信息，可以用于记录、评估或其他不需要梯度跟踪的操作。
```python
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
```

4. 上方在 v10DetectLoss类中的 loss_one2many 和 loss_one2one，都包括了 整个批次的总损失和损失副本。最后在v10DetectLoss的 __call__方法中返回的是 两个损失的标量和  两个损失的张量结果拼接在一 起。  
5. 在给定的代码中，我们可以看到 v10DetectLoss 类的实现，其中引入了双头结构，包括一个one-to-many head和一个one-to-one head。 两个head结构相同，使用相同的损失函数，但分别进行一对多和一对一标签分配。 在训练时，两个head同时优化，使backbone和neck能够享受到one-to-many分配带来的丰富监督信号。  

<a name="oAPDb"></a>
