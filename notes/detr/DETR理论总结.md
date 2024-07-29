# DETR
> 1. detr对大物体表现特别的好，归功于transformer，可以做全局的建模。同时detr也有缺陷，在小物体上效果就不怎么样。**Deformable DETR**不仅很好的通过多尺度的特征，解决了在小物体上的问题，也解决了DETR训练太慢的问题。
> 2. detr不像 transfomer使用掩码多头自注意机制，**自回归**的预测。而是使用不带掩码机制的Decoder，detr一股脑一口气把这个预测目标框全部给你输出，从而达到parallel decoding，时效性大大增强（RNN也是自回归的模型，是串行结构，时效性就比较差）
> 3. detr不像一阶段的anchor，也不像二阶段的proposal来进行人工先验知识的干预，而是使用集合检测的方法直接进行预测。使用了 Set-based loss目标函数
> 4. 使用object query去替代了原来生成anchor的机制。使用了二分图匹配替换原来NMS这一步。


## 基于集合的目标函数
> 作者如何通过一个二分图匹配把预测的框和 Ground Truth的框连接在一起，从而算得这个目标函数的，才能一对一的出框方式，才能不需要NMS

DETR这个模型，最后的输出是一个固定的集合。就是说不管你这个图片是什么，最后都会输出n个输出，论文中n=100。也就是说任何一张图片进来，最后都会给你扔出来n个框。（一般n这个个数会比你图片中物体的个数多很多的）。但是会有一个问题：DETR每次都会出100个输出，但是实际上，一个图片上gound truth的bounding box可能只有几个，如何做这种匹配呢？如何去算loss呢？怎么知道哪个预测框就对应那个gound truth框呢？所以作者就把这个问题转换成了 二分图匹配的问题。

二分图匹配的问题可以认为是如何分配工人和工作使最终的开销最小的问题。比如现在有三个工人：a，b，c。然后要去完成三个工作：x，y，z。这三个工人完成这三个工作所需的时间也不一样，所以最后会有下面这个矩阵，每个格子就有他完成这些任务所需要的时间，这个矩阵就叫 cost matrix，也叫作损失矩阵。最优二分图匹配的意思就是最后我能找到一个唯一解，能给**每个人**去分配他对应最擅长的那项工作，使得把这三个工作完成的时间最小。
![image.png](https://img-blog.csdnimg.cn/img_convert/03bb3fa4103eec61d3c6830f014352b0.png)
像匈牙利算法就是一个高效的解决算法，对应`scipy.optimize`这个库中的`linear_sum_assignment`函数实现。`linear_sum_assignment`这个函数的输入就是上面这个cost matrix，把cost matrix给它，它就能算出来一个最优的排列，告诉你哪个人应该干哪个活。

对应到目标检测，其实也是这种二分图匹配的问题，把a，b，c看成是100个预测的框，把x，y，z看作是ground truth的框，cost matrix可能是长方形，把预测框和真实框看成二分图的两个部分，使其准确度最高，得到最优匹配。对于目标检测而言。这个cost matrix里面的值应该放cost，也就是loss。用下面这个公式来算，包括分类的loss和 出框的loss。也就是说，遍历所有这些预测的框，拿这些预测的框去和ground truth的框去算这这个loss，然后把算出的loss放入到cost matrix中。最后通过匈牙利算法得到最优解，也就是最终匹配出来的结果损失值最小。
![image.png](https://img-blog.csdnimg.cn/img_convert/6ced8aea18b2635b677d5dbc70209e0f.png)
而这种方法和之前的anchor box或者proposal的loss是差不多的，只不过之前的约束更弱，为一对多的框，而这里二分图问题则是一对一的框，从而解决了NMS的问题。

经过上面的操作，完成了匹配的步骤。也就是说，知道了这100个预测框中有哪几个框和ground truth框是对应的。接下来就可以算真正的 目标函数，然后用这个loss去做梯度回传，去更新模型的参数了。最后的目标函数就是下面这个公式：
![image.png](https://img-blog.csdnimg.cn/img_convert/c8466b32c4387a976db0ba9922d9066c.png)
## DETR具体模型架构
> encoder 负责学习全局的特征，区分不同物体。decoder 负责学习边缘，更好的区分物体以及解决遮挡的问题。

![image.png](https://img-blog.csdnimg.cn/img_convert/9d3bcc4c8a8f6dce82ed1f541e8c0c34.png)
![image.png](https://img-blog.csdnimg.cn/img_convert/3b3850a839b5d202b774601f1dfe5540.png)

1. 首先通过卷积网络得到特征，由于要将特征扔给Transformer，所以在卷积最后加上一个1x1的降维操作。
2. 接下来因为要进入Transformer，而Transformer没有位置信息，所以加上位置编码，和特征的shape值一致。
3. 在输入到Transformer之前，将特征的 W和H维度拉直，变为两维，分别是序列长度和 head的维度。
4. 然后输入到encoder，detr中将encoder堆叠了6层来学习特征，经过encoder和decoder block之后，输入和输出的维度是不发生改变的。
5. 接着就是进入decoder，做框的输出。这里面有个新东西：object queries，它是可以学习的，替代了原来生成anchor的机制，也就是限制最后输出框的个数。在Decoder里面其实做的就是一个 cross attention，也就是一个输入是 object queries，另一个输入是是从图像端经过encoder拿到的全局特征。detr中decoder也是堆叠了6层。
6. 经过decoder之后，拿到最终的特征之后，最后就需要做一个预测了，通过两个检测头进行预测，一个是做物体类别的预测，一个是做出框的预测。

## detr相较于Transformer结构做的一些变化
**v不加位置编码**
在Transformer模型中，q和k必须要加上位置编码，但v对于位置编码不强求，但通常需要加上。位置编码是Transformer模型区分单词顺序的关键。由于Transformer的自注意力机制本身不包含序列的顺序信息，位置编码成为必需。通过将位置编码添加到输入特征中，模型能够有效地利用这些信息进行更准确的预测。
对于Q（查询）和K（键），它们在自注意力机制中通过**点乘**来计算不同词之间的相似度。这一过程生成了关注度得分，用于表示序列中各单词间的关注程度。因为这一计算过程密切依赖单词之间的相对位置，所以Q和K向量必须加入位置编码以保持正确的序列顺序。
虽然V（值）向量在技术上可以不添加位置编码，但为了保持所有输入向量的一致性和完整性，通常会对V应用相同的位置编码。这样做不仅统一了处理流程，而且增强了模型捕捉更细微关系的能力。从实用性角度来看，使用同样的位置编码方法为V提供了与Q和K一致的位置信息，这有助于下游任务中更加准确地重构输出序列。
**v代表每个元素的特征向量**，在detr中，作者对于v没有加上 位置编码。从而保留原始特征的完整性和独立性。这种设计可能是为了保持v向量的信息纯粹性，**避免引入可能影响特征表达的位置先验信息**。

**加位置编码的阶段不同**
![image.png](https://img-blog.csdnimg.cn/img_convert/fd4b0391a6ca5ffcf3e91c0453f8c153.png)     ![image.png](https://img-blog.csdnimg.cn/img_convert/5b634784c5b577436bea8ccef1169933.png)

- 在传统的Transformer中，在拿到输入向量后，直接加上位置编码，然后进行N次的encoder堆叠。位置编码只操作了一次。
- 而detr做了一些改变：位置编码是在每个堆叠的encoder和decoder中都要使用的。在encoder阶段，位置编码要操作N次，在decoder阶段，位置编码操作M次。

## Encoder 和 Decoder
![image.png](https://img-blog.csdnimg.cn/img_convert/832711a1c536d297e7e1ee0afd6f1dc4.png)
```python
def forward_post(self,
                 src,
                 src_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None):
    # 将图像特征src 和 位置信息pos进行嵌入，得到q和k
    q = k = self.with_pos_embed(src, pos)
    # 使用self_attn模块对q和k进行自注意力计算，得到src2
    src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    # 对src2进行dropout操作，使用残差结构
    src = src + self.dropout1(src2)
    # 对src进行归一化操作
    src = self.norm1(src)
    # FFN模块：对src进行线性变换、激活函数和dropout操作，得到src2
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    # 对src2进行dropout操作，使用残差结构
    src = src + self.dropout2(src2)
    # 对src进行归一化操作
    src = self.norm2(src)
    # 返回最终的src
    return src
```
```python
def forward_post(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
    # 对queries(tgt) 和 object queries(query_pos) 进行位置嵌入，得到q和k
    q = k = self.with_pos_embed(tgt, query_pos)
    # 第一个自注意力：对q、k进行自注意力计算，得到tgt2
    tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    # 对tgt2进行dropout操作，使用残差结构
    tgt = tgt + self.dropout1(tgt2)
    # 对tgt进行归一化操作
    tgt = self.norm1(tgt)
    # 第二个自注意力：q是queries 和 object queries进行位置嵌入；k是 encoder 输出和位置编码进行位置嵌入；v是encoder输出向量
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    # FFN模块：对tgt进行线性变换、激活函数和dropout操作，得到tgt2
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    # 对tgt2进行dropout操作，使用残差结构
    tgt = tgt + self.dropout3(tgt2)
    # 对tgt进行归一化操作
    tgt = self.norm3(tgt)
    return tgt
```
## 损失函数
detr的输出包括三部分：pred_logits (bs，100，92)、pred_boxes (bs，100，4) 和 aux_outputs (是一个list，里面包括了中间层的pred_logits和pred_boxes)。pred_logits、pred_boxes是最终的Decoder的拿到的输出经过两个检测头得到的类别检测结果和坐标预测结果。aux_outputs是Decoder的5个中间层的输出结果，论文中Decoder，要堆叠6次，所以有5个中间层结果，同样也要经过两个检测头得到中间层的类别检测结果和坐标预测结果。（所有阶段使用的检测头都是一致的）
![微信截图_20240729112627.png](https://img-blog.csdnimg.cn/img_convert/8a5a0e599f2e21f7fcc3885caff3aed1.png)
**detr损失计算**
![image.png](https://img-blog.csdnimg.cn/img_convert/3b3850a839b5d202b774601f1dfe5540.png)
detr需要计算两次损失。第一次是 最优匹配所需要的损失。第二次是真实预测所需要的损失。

1. 从输出中拿到了100个候选框，找到和真实标注框 所匹配的2个预测框（匈牙利算法）
   1. 使用匈牙利算法解决二分图匹配问题，从100个里面中筛选出2个
   2. 损失函数如下：表示真实标注框的类别集合

![image.png](https://img-blog.csdnimg.cn/img_convert/6ced8aea18b2635b677d5dbc70209e0f.png)

   - 匹配阶段 计算的是最终的预测信息和标注信息，不涉及到中间层的预测信息。
   - 前半段是分类损失。这里只计算标注类别的损失，并且类别不等于背景的。
   - 后半段是坐标的损失函数，根据论文中介绍，这一部分包括了 L1损失 - GLou损失，框的损失相对简单，直接拿100个框的坐标信息和 真实标注框的坐标信息进行L1损失 和 GLou损失，代码中有所体现。

![image.png](https://img-blog.csdnimg.cn/img_convert/220893e021e90a9c26593faca6ff76eb.png)
最后将三部分加权计算得到的损失值填充到 cost matrix，然后通过匈牙利算法二分图匹配算法得到最优解。
![image.png](https://img-blog.csdnimg.cn/img_convert/7cb4fb89d3134fe4e8da366bbb6b18a5.png)

2. 使用筛选出的预测框 和 真实标注框 计算损失，用于反向传播。

![image.png](https://img-blog.csdnimg.cn/img_convert/c8466b32c4387a976db0ba9922d9066c.png)
这里也是包括类别损失和坐标损失，但是和前面有所不同：

- 这里的类别损失是计算所有的，也就是100个预测框的损失，前面的匹配损失是只计算标注框并且是前景的损失。
- 类别损失使用的是带log，也就是交叉熵损失。
- 坐标损失：计算出来的每一部分的损失是要算平均值的。
- 从可以看出，计算坐标损失时，只计算标注框，并且是前景的损失。
- 中间层的输出，在这里也是参与计算了。





## 参考

[DETR 论文精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1GB4y1X72R/?spm_id_from=333.999.0.0&vd_source=03f639227780b0f99f96809cb7b135d3)
[DETR ｜ 1、算法概述_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Zo4y1A7FB/?spm_id_from=333.788)
