### 1.搭建环境

环境在实验进行时已经搭建完毕，具体步骤就不过多赘述（参考：[https://blog.csdn.net/weixin_39574469/article/details/117454061](https://blog.csdn.net/weixin_39574469/article/details/117454061)）

接下来只需导入所需的包即可

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers,activations
from tensorflow.keras.datasets import mnist,cifar10
```

### 2.获取CIFAR-10数据集

CIFAR-10数据集是大小为32*32的彩色图片集，数据集一共包括50000张训练图片和10000张测试图片，共有10个类别，分别是飞机（airplane）、汽车（automobile）、鸟（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog）、马（horse）、船（ship）、卡车（truck）。

```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 将像素的值标准化至0到1的区间内。

train_images, test_images = train_images / 255.0, test_images / 255.0
```

将测试集的前 25 张图片和类名打印出来，来确保数据集被正确加载。

```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # CIFAR 的标签是 array，需要额外的索引。
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021060422460596.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




---



### 3.建立图像分类模型

#### 3.1ResNet

网络越深，获取的信息就越多，特征也越丰富。但是在实践中，随着网络的加深，优化效果反而越差，测试数据和训练数据的准确率反而降低了。针对这一问题，何恺明等人提出了残差网络（ResNet）在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。

#### 3.2残差块

假设 F(x) 代表某个只包含有两层的映射函数， x 是输入， F(x)是输出。假设他们具有相同的维度。在训练的过程中我们希望能够通过修改网络中的 w和b去拟合一个理想的 H(x)(从输入到输出的一个理想的映射函数)。也就是我们的目标是修改F(x) 中的 w和b逼近 H(x) 。如果我们改变思路，用F(x) 来逼近 H(x)-x ，那么我们最终得到的输出就变为 F(x)+x（这里的加指的是对应位置上的元素相加，也就是element-wise addition），这里将直接从输入连接到输出的结构也称为shortcut，那整个结构就是残差块，ResNet的基础模块。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604224956544.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)


ResNet沿用了VGG全3×33×3卷积层的设计。残差块里首先有2个有相同输出通道数的3×33×3卷积层。每个卷积层后接BN层和ReLU激活函数，然后将输入直接加在最后的ReLU激活函数前，这种结构用于层数较少的神经网络中，比如ResNet34。若输入通道数比较多，就需要引入1×11×1卷积层来调整输入的通道数，这种结构也叫作瓶颈模块，通常用于网络层数较多的结构中。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604225010750.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




上图左中的残差块的实现如下，可以设定输出通道数，是否使用1*1的卷积及卷积层的步幅。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605140158783.png#pic_center)



```python
class Residual(tf.keras.Model):
    # 定义网络结构
    def __init__(self,num_channels,use_1x1conv=False,strides=1):
        super(Residual,self).__init__()
        # 卷积层
        self.conv1 = layers.Conv2D(num_channels,kernel_size=3,padding="same",strides=strides)
        # 卷积层
        self.conv2 = layers.Conv2D(num_channels,kernel_size=3,padding="same")
        # 是否使用1*1的卷积
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        # BN层
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    # 定义前向传播过程
    def call(self,x):
        Y = activations.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        outputs = activations.relu(Y + x)
        return outputs
```



#### 3.3残差模块

ResNet模型的构成如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/202106042250419.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




ResNet网络中按照残差块的通道数分为不同的模块。第一个模块前使用了步幅为2的最大池化层，所以无须减小高和宽。之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。

下面来实现这些模块。注意，这里对第一个模块做了特别处理。

```python
class ResnetBlock(tf.keras.layers.Layer):
    # 定义所需的网络结构
    def __init__(self,num_channels,num_res,first_block=False):
        super(ResnetBlock,self).__init__()
        # 存储残差块
        self.listLayers=[]
        # 遍历残差数目生成模块
        for i in range(num_res):
            # 如果是第一个残差块而且不是第一个模块时
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels,use_1x1conv=True,strides=2))
            else:
                self.listLayers.append(Residual(num_channels))
    # 定义前向传播
    def call(self,X):
        for layers in self.listLayers.layers:
            X = layers(X)
        return X
```

#### 3.4ResNet模型

ResNet的前两层跟之前提出的GoogLeNet中的一样：在输出通道数为64、步幅为2的7×77×7卷积层后接步幅为2的3×33×3的最大池化层。不同之处在于ResNet每个卷积层后增加了BN层,接着是所有残差模块，最后，与GoogLeNet一样，加入全局平均池化层（GAP）后接上全连接层输出。

```python
class ResNet(tf.keras.Model):
    # 定义网络的构成
    def __init__(self,num_blocks):
        super(ResNet,self).__init__()
        # 输入层
        self.conv = layers.Conv2D(64,kernel_size=3,strides=1,padding="same")
        # BN层
        self.bn = layers.BatchNormalization()
        # 激活层
        self.relu = layers.Activation("relu")
        # 池化层
        self.mp = layers.MaxPool2D(pool_size=2,strides=1,padding="same")
        self.res_block1 = ResnetBlock(64,num_blocks[0],first_block=True)
        self.res_block2 = ResnetBlock(128,num_blocks[1])
        self.res_block3 = ResnetBlock(256,num_blocks[2])
        self.res_block4 = ResnetBlock(512,num_blocks[3])
        # GAP
        self.gap = layers.GlobalAveragePooling2D()
        # 全连接层
        self.fc = layers.Dense(units=10,activation=tf.keras.activations.softmax)
    # 定义前向传播过程
    def call(self,x):
        # 输入部分的传输过程
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        # block
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # 输出部分的传输
        x = self.gap(x)
        x = self.fc(x)
        return x
```

这里每个模块里有4个卷积层（不计算 1×1卷积层），加上最开始的卷积层和最后的全连接层，共计18层。这个模型被称为ResNet-18。通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。虽然ResNet的主体架构跟GoogLeNet的类似，但ResNet结构更简单，修改也更方便。这些因素都导致了ResNet迅速被广泛使用。 在训练ResNet之前，我们来观察一下输入形状在ResNe的架构：

```python
# 实例化
mynet = ResNet([2,2,2,2])
X = tf.random.uniform(shape=(1,32,32,3))
y = mynet(X)
mynet.summary()
```

```python
Model: "res_net"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_20 (Conv2D)           multiple                  1792      
_________________________________________________________________
batch_normalization_17 (Batc multiple                  256       
_________________________________________________________________
activation_1 (Activation)    multiple                  0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 multiple                  0         
_________________________________________________________________
resnet_block_4 (ResnetBlock) multiple                  148736    
_________________________________________________________________
resnet_block_5 (ResnetBlock) multiple                  526976    
_________________________________________________________________
resnet_block_6 (ResnetBlock) multiple                  2102528   
_________________________________________________________________
resnet_block_7 (ResnetBlock) multiple                  8399360   
_________________________________________________________________
global_average_pooling2d_1 ( multiple                  0         
_________________________________________________________________
dense_1 (Dense)              multiple                  5130      
=================================================================
Total params: 11,184,778
Trainable params: 11,176,970
Non-trainable params: 7,808
_________________________________________________________________
```

---

### 4.模型编译

```python
# 优化器，损失函数，评价指标
mynet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005,momentum=0.9),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics = ["accuracy",tf.keras.metrics.sparse_top_k_categorical_accuracy],loss_weights=[1,0.3,0.3])
```



### 5.模型训练

```python
# 模型训练：指定训练数据，batchsize,epoch,验证集
history = mynet.fit(train_images,train_labels,batch_size=100,epochs=100,verbose=1,validation_split=0.1)
```



```python
Epoch 1/100
450/450 [==============================] - 50s 102ms/step - loss: 1.6243 - accuracy: 0.4205 - sparse_top_k_categorical_accuracy: 0.8757 - val_loss: 1.5159 - val_accuracy: 0.4866 - val_sparse_top_k_categorical_accuracy: 0.9206s - loss: 2.1511 - accuracy: 0.2566 - sparse_top_k_categorical_accuracy: 0.753 - ETA: 36s - loss: 2.1472 - accura - ETA: 30s - loss: 1.9909 - accuracy: 0.3021 - sparse_top_k_catego - ETA: 28s - loss: 1.9448 - accuracy: 0.3161 - sparse_t - ETA: 25s - loss: 1.8873 - accuracy: 0.3339 - sparse_top_k_categorical_accurac - ETA: 24s - loss: 1.8735 - accuracy: 0.3381 - sparse_top_k - E - ETA: 13s - loss: 1.7369 - accuracy: 0.3821 - sparse_top_k_categorical_accuracy: 0.8 - ETA: 13s - loss: 1.7340 - accuracy: 0.3831 - sparse_top_k_cate - ETA: 6s - loss: 1.6709 - accuracy: 0.4044 - s
Epoch 2/100
450/450 [==============================] - 45s 100ms/step - loss: 0.8699 - accuracy: 0.6918 - sparse_top_k_categorical_accuracy: 0.9763 - val_loss: 1.0755 - val_accuracy: 0.6096 - val_sparse_top_k_categorical_accuracy: 0.96864s - loss: 0.8930 - accuracy: 0.6784 - sparse_top_k_categorical_accur - ETA: 33s - loss: 0.8925 - accuracy: 0.6792 - sparse_top_k_categorical_accura - ETA: 32s - loss: 0.8917 - accuracy:  - ETA: 27s - loss: 0.8875 - accuracy: 0.6830 - sp - ETA: 23s - loss: 0.8849 - accuracy: 0.6847 - sparse_top_k_categoric - ETA: 21s - loss: 0.8839 - accuracy: 0.6854 - sparse_top_k_categorical_accur - ETA: 20s - loss: 0.8834 - accuracy: 0.6857 - sparse_top_k_cat - ETA: 18s - loss: 0.8820 - accuracy: 0.6866 - sparse_top_k_categorical_accurac - ETA: 17s - loss: 0.8816 - accuracy: 0.6868 - sparse_top_k_categorical_accuracy: 0.97 - ETA: 17s - loss: 0.8815 - accuracy: 0.6868 - sparse_top_k_categorical_accuracy - ETA: 16s - loss: 0.8811 - accuracy: 0.6871 - sparse_top_k_categorical_accura - ETA: 15s - loss: 0.8805 - accuracy: 0.6873 - sparse_top_k_cat - ETA: 13s - loss: 0.8791 - accuracy: 0.6881 - sparse_top_k_categorical_accuracy - ETA: 12 - ETA: 7s - loss: 0.8753  - ETA: 4s - loss: 0.8731  - ETA: 1s - loss: 0.8709 - accuracy: 0.6914 - sparse_top_
Epoch 3/100
450/450 [==============================] - 45s 100ms/step - loss: 0.6026 - accuracy: 0.7886 - sparse_top_k_categorical_accuracy: 0.9901 - val_loss: 1.3170 - val_accuracy: 0.5724 - val_sparse_top_k_categorical_accuracy: 0.9614ac
=========================================================================================================================================================================================================================
Epoch 98/100
450/450 [==============================] - 51s 113ms/step - loss: 3.1858e-05 - accuracy: 1.0000 - sparse_top_k_categorical_accuracy: 1.0000 - val_loss: 0.9684 - val_accuracy: 0.8122 - val_sparse_top_k_categorical_accuracy: 0.9880
Epoch 99/100
450/450 [==============================] - 52s 116ms/step - loss: 3.4269e-05 - accuracy: 1.0000 - sparse_top_k_categorical_accuracy: 1.0000 - val_loss: 0.9706 - val_accuracy: 0.8120 - val_sparse_top_k_categorical_accuracy: 0.9882
Epoch 100/100
450/450 [==============================] - 51s 112ms/step - loss: 3.3821e-05 - accuracy: 1.0000 - sparse_top_k_categorical_accuracy: 1.0000 - val_loss: 0.9698 - val_accuracy: 0.8116 - val_sparse_top_k_categorical_accuracy: 0.9882
```



### 6.模型评估

```python
mynet.evaluate(test_images,test_labels,verbose=1)
```

```python
313/313 [==============================] - 5s 15ms/step - loss: 0.9799 - accuracy: 0.8123 - sparse_top_k_categorical_accuracy: 0.9864
```

#### 6.1loss

```python
# 损失函数绘制

plt.figure()
plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="val")
plt.legend()
plt.grid()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605135839349.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




#### 6.2acc_top1

```python
# top1准确率

plt.figure()
plt.plot(history.history["accuracy"],label="train")
plt.plot(history.history["val_accuracy"],label="val")
plt.legend()
plt.grid()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605140035345.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




#### 6.3acc_top5

```python
# top5准确率

plt.figure()
plt.plot(history.history["sparse_top_k_categorical_accuracy"],label="train")
plt.plot(history.history["val_sparse_top_k_categorical_accuracy"],label="val")
plt.legend()
plt.grid()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605140100189.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)



### 7.预测

```python
image = Image.open("./img/ship.jpg")
plt.imshow(image)
newpic = np.array(image.resize((32, 32)))/255
print("下面的图预测结结果是",class_names[mynet.predict(np.array([newpic])).argmax()])
```



> 下面的图预测结结果是 ship
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604225134559.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)




### 

```python
image = Image.open("./img/bird.jpg")
plt.imshow(image)
newpic = np.array(image.resize((32, 32)))/255
print("下面的图预测结结果是",class_names[mynet.predict(np.array([newpic])).argmax()])
```



> 下面的图预测结结果是 bird
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604225144668.png?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU3NDQ2OQ==,size_16,color_FFFFFF,t_70#pic_center)
---
完整的代码已经上传到github：[https://github.com/a5116638/cifar10-resnet18](https://github.com/a5116638/cifar10-resnet18)






