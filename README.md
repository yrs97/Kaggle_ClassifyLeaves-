# Kaggle_ClassifyLeaves
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
    * 改善问题：图像增强原因
    <br>
  * [0.88181] : submit_2021060306.csv
    * 基于ResNet18
    * 根据beasline(细节)
      * 修改图像增强仅Resize(),ToTensor(),RandomHorizontalFlip(p=0.5),RandomRotation(5)
    * lr:8e-3,epoch:10
    * criterion = nn.CrossEntropyLoss()
    * optimizer = torch.optim.SGD
    <br>
  * [0.92181] : submit_2021060401.csv
    * 基于ResNet18
    * lr:2e-3,epoch:160
    <br>
  * [0.91090] : submit_2021060402.csv 
    * 基于ResNet50
    * lr:2e-3,epoch:60
  * [[0.93545](https://github.com/standbyme-ge/Kaggle_ClassifyLeaves-/blob/main/Code/0.93545_Mobinet_model)] : submit_2021060601.csv 
    * MobileNet_V2
    * lr:1e-3,epoch:160


#### 提升点

   * Transform：
      * 图像增强
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
