# MetaPruning

This is the pytorch implementation of our paper "MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning", https://arxiv.org/abs/1903.10258, published in ICCV 2019. 

<img width=60% src="https://github.com/liuzechun0216/images/blob/master/figure1.jpg"/>

在网络模型压缩中，channel pruning是一种十分有效的压缩方式。比较常见的剪枝pipeline是：训练一个完整模型 -> 减掉冗余的部分 -> finetune或retrain压缩的模型。

在以往的方法中，channel pruning大多是data-driven的稀疏化或者人工设计规则来选择保留哪些channel，减掉哪些channel；而最近有很多AutoML风格的剪枝方式，通过一些反馈机制或利用强化学习来自动的进行剪枝。

同时，以往的方法通常会保留之前完整模型中的参数；而最近也有一些研究表明，剪枝之后模型的性能，并不是由这些继承来的参数决定的，而是由结构本身决定的。

结合以上观点，本文提出了一种meta learning的方法——MetaPruning，利用meta learning训练一个Pruning Network用来生成压缩网络的参数，而不是继承或者重新训练。

它提出一种用于通道剪裁的元学习方法——MetaPruning，其核心是最前沿的AutoML 算法，旨在打破传统通道剪裁需人工设定每层剪裁比例，再算法迭代决定裁剪哪些通道的过程，直接搜索最优的已剪裁网络各层通道数。它的主要算法是通过学习一个元网络 PruningNet，为不同的剪裁结构生成权重，极大程度加速最优剪裁网络的搜索过程。

通道剪裁（Channel Pruning）作为一种神经网络压缩/加速方法，其有效性已深获认可，并广泛应用于工业界。

一个经典的剪裁方法包含三步：1）训练一个参数过多的大型网络；2）剪裁较不重要的权重或通道；3）微调或再训练已剪裁的网络。其中第二个阶段是关键，它通常借助迭代式逐层剪裁、快速微调或者权重重建以保持精度。

卷积通道剪裁方法主要依赖于数据驱动的稀疏约束（sparsity constraints）或者人工设计的策略。最近，一些基于反馈闭环或者强化学习的 AutoML 方法可自动剪裁一个迭代模型中的通道。

相较于传统剪裁方法， AutoML 方法不仅可以节省人力，还可以帮助人们在不用知道硬件底层实现的情况下，直接为特定硬件定制化设计在满足该硬件上速度限制的最优网络结构。

MetaPruning 作为利用 AutoML 进行网络裁剪的算法之一，有着 AutoML 所共有的省时省力，硬件定制等诸多优势，同时也创新性地加入了先前 AutoML pruning 所不具备的功能，如轻松裁剪 shortcut 中的通道。

过去的研究往往通过逐层裁剪一个已训练好模型中带有不重要权重的通道来达到裁剪的目的。而一项最新研究发现，不管继不继承原始网络的权重，已剪裁的网络都可获得相同精度。

这一发现表明，通道剪裁的本质是决定逐层的通道数量。基于这个，MetaPruning 跳过选择剪裁哪些通道，而直接决定每层剪裁多少通道——好的剪裁结构。

然而，可能的每层通道数组合数巨大，暴力寻找最优的剪裁结构是计算量所不支的。

受到近期的神经网络架构搜索（NAS）的启发，尤其是 One-Shot 模型，以及 HyperNetwork 中的权重预测机制，旷视研究院提出训练一个 PruningNet，它可生成所有候选的已剪裁网络结构的权重，从而仅仅评估其在验证集上的精度，即可搜索表现良好的结构。这极其有效。

PruningNet 的训练采用随机采样网络结构策略，如图 1 所示，它为带有相应网络编码向量（其数量等于每一层的通道数量）的已剪裁网络生成权重。通过在网络编码向量中的随机输入，PruningNet 逐渐学习为不同的已剪裁结构生成权重。


Traditional pruning decides pruning which channel in each layer and pays human effort in setting the pruning ratio of each layer. MetaPruning can automatically search for the best pruning ratio of each layer (i.e., number of channels in each layer). 

MetaPruning contains two steps: 
1. train a meta-net (PruningNet), to provide reliable weights for all the possible combinations of channel numbers in each layer (Pruned Net structures).
2. search for the best Pruned Net by evolutional algorithm and evaluate one best Pruned Net via training it from scratch.

# Citation

If you use the code in your research, please cite:

	@article{liu2019metapruning,
	  title={MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning},
	  author={Liu, Zechun and Mu, Haoyuan and Zhang, Xiangyu and Guo, Zichao and Yang, Xin and Cheng, Tim Kwang-Ting and Sun, Jian},
	  journal={arXiv preprint arXiv:1903.10258},
	  year={2019}
	}

# Run

1. Requirements:
    * python3, pytorch 1.1.0, torchvision 0.3.0

2. ImageNet data:
    * You need to split the original training images into sub-validation dataset,  which contains 50000 images randomly selected from the training images with 50 images in each 1000-class, and sub-training dataset with the rest of images. Training the PruningNet with the sub-training dataset and searching the pruned network with the sub-validation dataset for inferring model accuracy. 

3. Steps to run:
    * Step1:  training
    * Step2:  searching 
    * Step3:  evaluating
    
    * After training the Pruning Net, checkpioint.pth.tar will be generated in the training folder, which will be loaded by the searching algorithm. After searching is done, the top1 encoding vector will be shown in the log. By simply copying the encoding vector to the rngs = \[ \] in evaluate.py, you can evaluate the Pruned Network corresponding to this encoding vector. 

# Models

MobileNet v1

| | Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | --- | 
| Ratio | Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 1x | 70.6% | 569M | - | - | - |
| 0.75x | 68.4% | 325M | 70.9% | 316M | [Model-MetaP-Mbv1-0.75](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EXuAHjKVTa9Gkj4ZHET0s58BU9QGI9O88iEVLopWu-usdw?e=b0VcpJ) |
| 0.5x  | 63.7% | 149M | 66.1% | 142M | [Model-MetaP-Mbv1-0.5 ](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ERb0bJ7ggL5Du8v4mrLeVlkBEontkyhTWdDKIoMZQwHC2w?e=5pXdDh) |
| 0.25x | 50.6% | 41M  | 57.2% | 41M  | [Model-MetaP-Mbv1-0.25](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EQpBwbDTmCxLmpG8BCzG3xMBJsRoYOURAwG53HzkIqOzKQ?e=UrABgZ) |


MobileNet v2

| Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | 
| Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 74.7% | 585M | - | - | - |
| 72.0% | 313M | 72.7% | 303M | [Model-MetaP-Mbv2-300M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EWtiXOwKblRNnQs_xtzpM8oBuQ7wlAXGzrlJgEPZ7aXc7Q?e=h1vn4s) |
| 67.2% | 140M | 68.2% | 140M | [Model-MetaP-Mbv2-140M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcCbtJSanrdJs9xANHx6qSgBF3FzCN00uNTlDv2vJlZlNw?e=HoQmtY) |
| 54.6% | 43M  | 58.3% | 43M  | [Model-MetaP-Mbv2-40M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EX_Lna862JpHhLz7eWUBRqABdCh1_7wyxtaW4bE7PC3wuw?e=dEkWyv)  |


ResNet

| | Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | --- | 
| Ratio | Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 1x | 76.6% | 4.1G | - | - | - |
| 0.75x | 74.8% | 2.3G | 75.4% | 2.0G | [Model-MetaP-ResN-0.75](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EUpHJYfrtaFMn46Af94vqn4BgNr9AAZ6hskoWahtA8r5Tg?e=8ovp6p) |
| 0.5x  | 72.0% | 1.1G | 73.4% | 1.0G | [Model-MetaP-ResN-0.5 ](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EX8NhySdkw1NrUx9EYVCH0sBEJzgwM4ZS0Opv6WG0intJA?e=xMLY07) |
