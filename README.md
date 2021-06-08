# Kaggle_ClassifyLeaves
Study Note
动手学深度学习_竞赛2_20210602

#### Competition Address
* [竞赛地址](#https://www.kaggle.com/c/classify-leaves)<br>
    * https://www.kaggle.com/c/classify-leaves
* [B站视频](#https://www.bilibili.com/video/BV1z64y1o7iz/)<br>
    * https://www.bilibili.com/video/BV1z64y1o7iz/
* beasline:
    * [Datawhale 零基础入门CV赛事-Baseline：(主体结构)](https://tianchi.aliyun.com/notebook-ai/detail?postId=108342)
    * [simple-resnet-baseline：（细节）](https://www.kaggle.com/nekokiku/simple-resnet-baseline)


#### Model
* [ResNet18](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Model/ResNet18_model)
* [ResNet50](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Model/ResNet50_model)
* [MobileNet_V2](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Model/Mobilenet_model)


#### Record 

* 分数
  * [0.00522] : submit_2021060201.csv
    * 基于ResNet18
    * 步骤正确，Loss and Acc 都很不错。但是分数极低。
    * 引发问题：图像增强原因
    <br>
  * [0.88181] : [submit_2021060306.csv](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/csv/submit_2021060306.csv)
    * 基于ResNet18
    * 根据beasline(细节)
      * 修改图像增强仅Resize(),ToTensor(),RandomHorizontalFlip(p=0.5),RandomRotation(5)
    * lr:8e-3,epoch:10
    * criterion = nn.CrossEntropyLoss()
    * optimizer = torch.optim.SGD
    <br>
  * [0.92181] : [submit_2021060401.csv](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/csv/submit_2021060401.csv)
    * 基于ResNet18
    * lr:2e-3,epoch:160
    <br>
  * [0.91090] : [submit_2021060402.csv](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/csv/submit_2021060402.csv)
    * 基于ResNet50
    * lr:2e-3,epoch:60
  * [[0.93545](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Code/0.93545_Mobinet_model)] : [submit_2021060601.csv](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/csv/submit_2021060601.csv)
    * MobileNet_V2
    * lr:1e-3,epoch:160
  * [[0.95181](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Code/0.95181_ResNet50_model)] : [submit_2021060801_loss.csv](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/csv/submit_2021060801_loss.csv)
    * ResNet50
    * lr = 0.01
    * momentum = 0.9                          # 动量(momentum)的引入就是为了加快SGD学习过程
    * weights_dacay = 1e-4
    * lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,eta_min=0)
    * epoch = 34                              # 11-th epoch 出现最佳loss-0.15,acc-95.57

#### 提升点

   * Transform：
      * 图像增强
         * 20210608: 
            * transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ：导致过拟合（trainAcc High, valAcc low）
   * Color:
      * RGB -> L
   * Model:
      * 1. Resnet50 :改最后两层全连接层为单层：直接输出分类数
   * lr:
      
      * 学习率下降
         * 正则项
         * CosineAnnealing(余旋退火)
         * 热重启 SGDR
      * 利用fastai.lr_find()查找最合适学习率
   * Criterion:
      * CrossEntropyLoss()
   * Optimizer:
      * Adam 收敛快
      * SGD + monmentum(0.8) 效果好
   * Cross Validation:
      * k折交叉验证：常用5折
   * Image fusion:
      * 多模型融合
         * 同样的参数,不同的初始化方式
         * 同样的参数,模型训练的不同阶段，即不同迭代次数的模型。
         * 不同的参数,通过cross-validation,选取最好的几组
         * 不同的模型,进行线性融合. 例如RNN和传统模型
      * Example
         * 1. model1 probs + model2 probs + model3 probs ==> final label
         * 2. model1 label , model2 label , model3 label ==> voting ==> final label
         * 3. model1_1 probs + ... + model1_n probs ==> mode1 label, <br>
              model2 label与model3获取的label方式与1相同  ==> voting ==> final label
    
      * 参考：<br>
      [1]. 2021版调参上分手册：https://mp.weixin.qq.com/s/oi3wXRv9NC5lL831s7AsQg<br>
   
   
####  知识点

   * [模型保存](https://www.cnblogs.com/zkweb/p/12843741.html)
```
      import gzip
      torch.save(model.state_dict(), gzip.GzipFile("model.pt.gz", "wb"))            #save
      new_model.load_state_dict(torch.load(gzip.GzipFile("model.pt.gz", "rb")))     #load
```
   

