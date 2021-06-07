# Kaggle_ClassifyLeaves
动手学深度学习_竞赛2_20210602

#### Competition Address
* [竞赛地址](#https://www.kaggle.com/c/classify-leaves)<br>
    * https://www.kaggle.com/c/classify-leaves
* [B站视频](#https://www.bilibili.com/video/BV1z64y1o7iz/)<br>
    * https://www.bilibili.com/video/BV1z64y1o7iz/
* [beasline: ](#)
    * [Datawhale 零基础入门CV赛事-Baseline：(主体结构)](https://tianchi.aliyun.com/notebook-ai/detail?postId=108342)
    * [simple-resnet-baseline：（细节）](https://www.kaggle.com/nekokiku/simple-resnet-baseline)


#### Model
* [ResNet18](#)
* [ResNet50](#)
* [MobileNet_V2](#)


#### Record 

* [分数](#)
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
  * [0.93545] : submit_2021060601.csv
    * MobileNet_V2
    * lr:1e-3,epoch:160
