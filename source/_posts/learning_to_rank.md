---
title: learning to rank算法学习
date: 2019-01-29
toc: true
categories: 模型与算法
tags: [l2r, learning2rank]
description: learning to rank算法笔记
mathjax: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX"],
    tex2jax: {
      inlineMath: [ ['$','$'], ['\\(','\\)'] ],
      displayMath: [ ['$$','$$']],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML,http://myserver.com/MathJax/config/local/local.js">
</script>


L2R(learning to rank)是指利用机器学习的技术，去完成排序的任务。在模型训练阶段，该算法最终优化的目标是一个更加合理的排序指标。L2R技术已经成功应用在信息检索(IR)、NLP、和数据挖掘(DM)等领域【1】。根据具体优化的目标不同，L2R算法主要分为Pointwise，Pairwise，Listwise三类。本文首先介绍L2R建模的整体框架，然后针对三类方法分别进行介绍。


# L2R建模

L2R建模的基本框架如下所示：

![“L2R基本框架”](learning_to_rank/l2r_framework.png) 
<center/>图1：L2R基本框架</center>

如图1所示，模型具体分为训练和预测两个解读。

**训练阶段：** 训练数据包括多个group, 其中每个group由1个query和1组document共同构成。该group中，$q\_i$表示第$i$个query，$d\_{i,j}$表示对应第$i$个query的第$j$个document, 不仅包含了document的各种属性，也包括了对应document和query之间的相关性标签$y\_i$。具体如下所示：

$S=[((q\_1, D\_1), y\_1), ((q\_2, D\_2), y\_2), ..., ((q\_m, D\_m), y\_m)]$
 
$D\_i=[d\_{i,1}, d\_{i,2}, ..., d\_{i,n}]$

$y\_i=[y\_{i,1}, y\_{i,2}, ..., y\_{i,n}]$

其中$m$表示query个数，$n$表示每个group的document个数（此处假设每个group的document个数相同为$n$）

在训练阶段，每个pair$(q\_i$, $d\_{i,j})$ 都提取相关的特征，作为特征向量$x\_{i,j}$。 模型学习的目标是，对每个pair$(q\_i$, $d\_{i,j})$，预测其对应的分数$f(x\_{i,j})$, 使得根据这些分数得到的每个$q\_i$对应的所有$d\_{i,j}$排序尽量接近真实排序。

**预测阶段：** 输入query $q\_{m+1}$和document集合$D=[d\_1, d\_2, ... , d\_N]$, 利用训练得到的model，预测query $q\_{m+1}$和每个document $d\_i$的相关性分数，并根据预测的分数对document进行排序，输出排序列表。

**和分类回归的关系：** 传统的分类和回归方法，通过学习相关模型，预测样本的类别或者分数值；而排序模型，则是通过模型，预测样本相关性(或者其它分数)的相对顺序。在分类问题中，有一种问题是序数分类(Ordinal Classification)，序数分类问题和排序问题有点类似，不同之处在于序数分类的目标是预测顺序的类别(ordered-categorization of objects)，而排序问题的目标是预测相对顺序(ordering)。

# L2R评估

L2R的评估基于预测的rank list和真实的rank list比较，主要有DCG(Discounted Cumulative Gain)，NDCG(Normalized Discounted Cumulative Gain)，MAP(Mean Average Precision)等评估指标。

## DCG

对于给定的query, TOP T的返回结果对应的DCG值如下所示：

$DCG@T = \sum\_{i=1}^T \frac{2^{l\_i} - 1}{log(1 + i)} $

其中$i$表示预测结果列表中第$i$个位置，$l\_i$表示预测结果中第$i$个位置的document的真实相关性值。分子部分描述了饭回结果的相关性，分母部分针对位置进行加权，排序越靠前，其对应的相关性值权重系数越大。

所有query的平均DCG值作为最终排序系统DCG评估值。

## NDCG

NDCG在DCG指标基础上进行了扩展，通过将DCG值除于DCG最佳排序对应的DCG值，将其归一化到0-1的范围，其定义如下所示：

$NDCG@T = \frac{DCG@T}{maxDCG@T} $

所有query的平均NDCG值作为最终排序系统DCG评估值。

## MAP

map是L2R中另一种评估指标，其对应的相关性标签只有0和1。对于给定的query，AP的定义如下所示：
$AP = \frac{\sum\_{i=1}^{n} P(i) * l\_i}{\sum\_{i=1}^{n}l\_i}$

其中$i$表示预测结果列表中第$i$个位置，$l\_i$表示预测结果中第$i$个位置的document的真实相关性值（在MAP中，相关性值取0或1）。$n$表示排序列表的长度，$P(i)$表示从列表第一个位置到第$i$个位置预测结果的平均准确率，其定义如下：
 $P(i) = \frac{\sum\_{k=1}^i l\_k}{i}$

所有query的平均AP值作为最终排序系统MAP评估值。



# Pointwise

在Pointwise方法中，排序问题可以转化成分类或回归问题，分类(包括序数分类)或回归的方法都可以使用。由于建模没有使用样本的相对顺序，group也不需要构建。

此处以OC SVM(SVM for Ordinal Classification)【2】为例，说明如何利用分类方法解决排序问题。该方法优化的目标是，对于任何相邻的2个类别，最大化其对应的分类间隔。实现层面，如图2所示，对于类别为$l$的序数分类问题，引用$l-1$个分类器 $⟨w, x⟩−b\_r(r = 1,···,l − 1)$, 其中$ b\_1≤···≤b\_{l−1}≤b\_l=\inf $。$⟨w, x⟩−b\_r = 0$用于划分第$r$和$r-1$个类别，如果$⟨w, x⟩ + b\_{r-1} >= 0$并且$⟨w, x⟩ + b\_{r} < 0$, 则样本标签属于$y=r$。建模的目标函数如下所示：

$min\_{w, b, \xi} = \frac{1}{2}||w||^2+C\sum\_{r=1}^{l-1}\sum\_{i=1}^{m\_r}(\xi\_{r,i}+\xi\_{r+1,i}^*)$ 

约束如下：

$⟨w, x\_{r, i}⟩ + b\_r < -(1 - \xi\_{r,i})$

$⟨w, x\_{r+1, i}⟩ + b\_r >= 1 - \xi\_{r, i}^*$

$\xi\_{r,i} >= 0$

$\xi\_{r+1,i}^* >= 0$

$i = 1, 2, ... , m\_r$

$r = 1, 2, ..., l-1$

$m = m\_1 + m\_2 + ... + m\_l$

其中 $x\_{r,i}$表示第$r$个类别的第$i$个样本，$\xi\_{r,i}$和$\xi\_{r+1,i}^*$表示对应的松弛变量，$m$是样本的个数, $m\_i$表示第$i$类样本的个数。


![“OC SVM”](learning_to_rank/oc_svm.png) 
<center/>图2：SVM for Ordinal Classification【2】</center>



# Pairwise

基于pairwise的rank方法中，将排序问题转化为pairwise的分类或回归问题进行求解。通常情况下，针对一个query对应的document pair, 利用分类器对pair的order进行判断。常见的pairwise rank方法有rank net、rank svm等，此处以rank net为例进行说明。

## rank net原理及求解
**rank net建模**

rank net使用的打分模型要求对参数可导，训练数据根据query分为多个组，对于1个给定的query，选择2个不同相关性label的document pair，计算相关性分数$s\_i=f(x\_i)$和$s\_j=f(x\_j)$，rank net对其对应的特征向量进行打分。$d\_i>d\_j$表示document $d\_i$的相关性大于$d\_j$。

document $d\_i$的相关性大于document $d\_j$的概率如下:

$P\_{ij}=P(d\_i>d\_j)=\frac{1}{1+e^{-\sigma (s\_i-s\_j)}}  (1)$ 

其中$\sigma$是常数，决定sigmoid函数的形状。rank net采用交叉熵函数训练模型，如下所示。其中$P'\_{ij}$表示真实的$d\_i$相关性大于$d\_j$的概率。

$C=-P'\_{ij}logP\_{ij}-(1-P'\_{ij})log(1-P\_{ij}))  (2)$

**rank net求解**

为方便后续描述，针对1个给定的query，我们定义变量$S\_{ij}$：

$S\_{ij} =
\begin{cases}
1,  & d\_i比d\_j更相关\\\\
-1,  & d\_j比d\_i更相关\\\\
0  & d\_i和d\_j相关性相同
\end{cases}
 (3)$

在本文中，假定对于每个query，其对应所有document的相关性顺序都是完全确定的。

因此，

$P'\_{ij}=\frac{1}{2}(1+S\_{ij}). (4)$

由上述式2和式4的到，$C=\frac{1}{2}(1-S\_{ij})\sigma (s\_i - s\_j) + log(1+e^{-\sigma(s\_i-s\_j)}) (5)$

$C =
\begin{cases}
log(1+e^{-\sigma(s\_i-s\_j)}),  & 当S\_{ij}=1\\\\
log(1+e^{-\sigma(s\_j-s\_i)}),  & 当S\_{ij}=-1
\end{cases}
(6)$

$C$对$s$求导，结果如下：
$ \frac{\varphi C}{\varphi s\_i}=\sigma (\frac{1}{2}(1-S\_{ij})-\frac{1}{1+e^{-\sigma(s\_i-s\_j)}})=-\frac{\varphi C}{\varphi s\_j} (7)$

通过SGD的方式进行求解

$w\_k=w\_k-\eta (\frac{\varphi C}{\varphi s\_i}\frac{\varphi s\_i}{\varphi w\_k}+\frac{\varphi C}{\varphi s\_j}\frac{\varphi s\_j}{\varphi w\_k}) (8)$

其中$\eta > 0$为学习率。

**rank net求解加速**

对于给定的文档对$d\_i$和$d\_j$，

$ \frac{\varphi C}{\varphi w\_k}=\frac{\varphi C}{\varphi s\_i}\frac{\varphi s\_i}{w\_k}+\frac{\varphi C}{\varphi s\_j}\frac{\varphi s\_j}{\varphi w\_k} = \sigma (\frac{1}{2}(1-S\_{ij})-\frac{1}{1+e^{-\sigma(s\_i-s\_j)}}) (\frac{\varphi s\_i}{\varphi w\_k} - \frac{\varphi s\_j}{\varphi w\_k})=\lambda\_{ij}(\frac{\varphi s\_i}{\varphi w\_k} - \frac{\varphi s\_j}{\varphi w\_k})(9)$

其中$\lambda\_{ij}=\frac{\varphi C}{\varphi s\_i}= \sigma (\frac{1}{2}(1-S\_{ij})-\frac{1}{1+e^{-\sigma(s\_i-s\_j)}})$

我们定义$I$为索引对$(i,j)$的集合(其中$doc\_i$的相关性大于$doc\_j$)，汇总来自所有文档对的贡献，

$\delta w\_k=-\eta \sum\_{(i,j) \in I} \lambda\_{ij}(\frac{\varphi s\_i}{\varphi w\_k} - \frac{\varphi s\_j}{\varphi w\_k})=-\eta\sum\_i\lambda\_i\frac{\varphi s\_i}{\varphi w\_k}(10)$

其中$\lambda\_i = \sum\_{j:(i,j) \in I}\lambda\_{ij}-\sum\_{j:(j,i) \in I}\lambda\_{ij}$，每个document对应一个$\lambda\_i$，方向表示梯度更新的方向，大小表示梯度更新的幅度。每个$\lambda\_i$的计算都来自该document对应的所有pair。在实际计算时，可以对每个文档计算其对应的$\lambda\_i$，然后用于更新模型参数。这种mini-batch的梯度更新方式和问题分解方法，显著提升了ranknet学习的效率。



# Listwise

# 参考资料

[1] A Short Introduction to Learning to Rank
[2] A. Shashua and A. Levin, “Ranking with large margin prin- ciple: Two approaches,” in Advances in Neural Information Processing Systems 15, ed. S.T. S. Becker and K. Ober- mayer, MIT Press.


 
