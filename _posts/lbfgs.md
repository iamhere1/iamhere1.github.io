---
title: lbfgs算法与源码学习
date: 2018-1-13
toc: true
categories: 模型与算法
tags: [lbfgs, 拟牛顿算法, 非线性优化]
description: lbfgs算法具备牛顿法收敛速度快的优点，同时又不需要存储和计算完整的hessian矩阵，能够节省大量的存储和计算资源，非常适用于解决无约束的大规模的非线性优化问题。

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


LBFGS（limited-memory BFGS或limited-strorate BFGS）算法具备牛顿法收敛速度快的优点，同时又不需要存储和计算完整的hessian矩阵，能够节省大量的存储和计算资源，非常适用于解决无约束的大规模非线性优化问题。

本文从牛顿法出发，先简要介绍牛顿法、拟牛顿法，然后从分别从原理和源码实现的角度介绍lbfgs优化算法。其源码主要来自chokkan等人贡献[1]。

# 牛顿法
**原始牛顿法:**

目标函数: $min\;\;f(x)\;(1)$

函数$f(x)$在$x=x\_k$附近进行二阶泰勒展开，如下所示：

$f(x) \approx f(x\_k) + \bigtriangledown f(x\_k)(x-x\_k) + \frac{1}{2}(x-x\_k)^T \bigtriangledown^2 f''(x\_k)(x-x\_k)\;(2)$

为了求$f(x)$的极小值,令$f(x)$的导数为0，得到：

$x = x\_k - H\_k^{-1}g\_k\;(3)$

其中$g\_k$为函数$f$在$x=x\_k$的一阶导数， $H\_k$为函数$f$在$x=x\_k$的二阶导数(hessian矩阵)。
因此，为求解下次迭代结果，可直接令:

$x\_{k+1}=x\_k-H\_k^{-1}g\_k\;(4)$

算法在利用牛顿法求解时，从$x=x\_0$出发，逐步迭代直到终止。终止的条件可以是梯度的二范数小于一定值，或者达到最大迭代次数等。

**阻尼牛顿法:**

原始牛顿法是固定步长迭代，对于非二次型目标函数，不能保证目标函数值稳定下降。严重情况下可能造成迭代点序列发散，使得计算失败。为消除该缺点，采用阻尼牛顿法，在更新迭代点时寻求最优步长$ \lambda\_k$。

$\lambda\_k=argmin\_{\lambda}f(x\_k+\lambda H\_k^{-1}g\_k)\;(5)$

$x\_{k+1}=x\_k+\lambda H\_k^{-1}g\_k\;(6)$

**牛顿法及阻尼牛顿法优点：**当目标函数$f$为二次函数，且hessian矩阵正定时，通过牛顿法一步就可以得到最优解。当目标函数$f$为非二次函数，但是其二次性较强或迭代点已进入极小点附近，其收敛速度也很快。

**牛顿法及阻尼牛顿法缺点:**要求目标函数$f$需要具有连续的一、二阶导数，且hessian矩阵正定；当特征维度很高时，hessian矩阵存储需要很大空间，求逆计算量也很大，不适合用于大规模问题的优化。


# 拟牛顿法

拟牛顿法的核心思想是：直接构造hessian矩阵或hessian矩阵的逆，从而在构造的近似hessian矩阵基础上按照式4或式6进行迭代求解。

## 拟牛顿条件

函数$f(x)$在$x=x\_{k+1}$附近进行二阶泰勒展开，如下所示：

$f(x) \approx f(x\_{k+1}) + \bigtriangledown f(x\_{k+1})(x-x\_{k+1}) + \frac{1}{2}(x-x\_{k+1})^T \bigtriangledown^2 f(x\_{k+1})(x-x\_{k+1})\;(7)$

对上式两边对$x$求导，如下所示:

$\bigtriangledown f(x) \approx \bigtriangledown f(x\_{k+1}) + \bigtriangledown^2 f''(x\_{k+1})(x-x\_{k+1})\;(7)$


取$x=x\_k$,则由式7可以得到：

$g\_{k+1}-g\_k \approx  H\_{k+1}(x_{k+1}-x\_k)\;(8)$

其中$g\_k$为函数$f$在$x=x\_k$的一阶导数， $H\_{k+1}$为函数$f$在$x=x\_{k+1}$的二阶导数(hessian矩阵)。令$y\_k=g\_{k+1}-g\_k$, $s\_k=x_{k+1}-x\_k$, $G\_{k+1}=H^{-1}\_{k+1}$ 得：

$y\_k \approx H\_{k+1} s\_k\;(9)$

$s\_k \approx G\_{k+1}y\_k\;(10)$

式9和式10是拟牛顿条件，在迭代过程中对hessian矩阵$H\_{k+1}$做近似，或者对hessian矩阵的逆$G\_{k+1}$做近似，而不是直接求解hessian矩阵，就是拟牛顿法。比较常用的拟牛顿法包括DFP算法和BFGS算法。

## DFP算法

DFP算法的核心是通过迭代对hessian矩阵的逆进行近似，迭代公式：
$G\_{k+1}=G\_k + \bigtriangleup G\_k, \; k = 0, 1, 2, ... \;(11)$

其中$G\_k$可以通过单位矩阵构造，关键在于如何构造$\bigtriangleup G\_k$，其构造过程如下：
为保证对称性，我们假定:

$\bigtriangleup G\_k=\alpha uu^T + \beta vv^T\;(12)$

将式11和式12代入式10，可得：

$s\_k \approx G\_{k+1}y\_k$

$=>s\_k = (G\_k + \bigtriangleup G\_k)y\_k=G\_ky\_k+\alpha u^Ty\_ku+\beta v^Ty\_kv$

$=>s\_k-G\_ky\_k=\alpha u^Ty\_ku+\beta v^Ty\_kv；（13）$

为使得式12成立，直接使$\alpha u^Ty\_k=1$, $\beta v^Ty\_k=-1, u=s\_k, v=G\_ky\_k$，得到$\alpha=\frac{1}{s^Ty\_k}$, $\beta = -\frac{1}{y^T\_kG\_ky\_k}$, 将$\alpha,\beta,u,v$代入式12，得：

$\bigtriangleup G\_k=\frac{s\_ks^T\_k}{s^T\_ky\_k}-\frac{G\_ky\_ky^T\_kG\_k}{y^T\_kG\_ky\_k} \;(14)$

DFP算法根据式11和式14，迭代求解hessian矩阵的逆$G\_k$,其他步骤同牛顿法（或阻尼牛顿法）。

## BFGS算法
BFGS算法核心思想是通过迭代对hessian矩阵进行近似（和DFP算法不同之处在于，DFP算法是对hessian矩阵的逆进行近似）。相对于DFP算法，BFGS算法性能更佳，具有完善的局部收敛理论，在全局收敛性研究也取得重要进展[4]。

BFGS算法和DFP算法推导类似，迭代公式：
$H\_{k+1}=H\_k + \bigtriangleup H\_k, \; k = 0, 1, 2, ... \;(15)$

其中H\_0可以用单位矩阵进行构造，对于$\bigtriangleup H\_k$的构造如下：

$\bigtriangleup H\_k= \alpha uu^T + \beta vv^T\;(16)$

将式15和式16代入式9，得：

$y\_k \approx H\_{k+1}s\_k $

$=> y\_k= H\_ks\_k+\alpha u^Ts\_ku + \beta v^Ts\_kv$

$=>y\_k-H\_ks\_k=\alpha u^Ts\_ku + \beta v^Ts\_kv\;(17)$

为使式17成立，直接令$u=y\_k$, $v=H\_ks\_k, \alpha u^Ts\_k=1, \beta v^Ts\_k=-1$,  将$\alpha,\beta,u,v$代入式15，得：

$\bigtriangleup H\_k=\frac{y\_ky\_k^T}{y\_k^Ts\_k}-\frac{H\_ks\_ks\_k^TH\_k^T}{s\_k^TH\_ks\_k}\;(18)$

BFGS算法通过式18更新hessian矩阵的求解过程，在求解搜索方向$d\_k=H\_k^{-1}g\_k$时，通过求解线性方程组$H\_kd\_k=g\_k$得到$d\_k$的值。

更一般的解法是通过sherman-morrison公式[6],直接得到$H\_{k+1}^{-1}$和$H\_k^{-1}$之间的关系如式19所示，并根据该关系迭代求解hessian矩阵的逆:
$H\_{k+1}^{-1}=(I-\frac{s\_ky_k^T}{y\_k^Ts\_k})H\_k^{-1}(I-\frac{y\_ks\_k^T}{y\_k^Ts\_k})+\frac{s\_ks\_k^T}{y\_k^Ts\_k}\;($

$=> G\_{k+1}=(I-\frac{s\_ky_k^T}{y\_k^Ts\_k})G\_k(I-\frac{y\_ks\_k^T}{y\_k^Ts\_k})+\frac{s\_ks\_k^T}{y\_k^Ts\_k}\;(19)$


# LBFGS算法

BFGS算法需要存储完整的$H\_k^{-1}$矩阵。因此，当矩阵的维度比较大时，需要占用大量的存储空间(空间复杂度为$O(N^2)$)。LBFGS算法通过使用最近$m$次迭代过程中的$s$和$y$向量，使得其存储复杂度由$O(N^2)$下降到$O(m\times N)$[2]。

本章节首先介绍lbfgs算法和求解推导、然后介绍带有L1正则的LBFGS算法求解（OWLQN算法）、最后介绍lbfgs算法在liblbfgs库[1]中的实现。

## LBFGS算法及求解
对于式19，我们令$\rho\_k=\frac{1}{y\_k^Ts\_k}$, $v\_k=(I-\rho\_ky\_ks\_k^T)$, 得：

$G\_{k+1}=v\_k^TG\_kv\_k+\rho\_ks\_ks\_k^T\;(20)$

假定$G\_0$是正定矩阵，则：

$G\_1=v\_0^TG\_0v\_0+\rho\_0s\_0s\_0^T$

$G\_2=v\_1^TG\_1v\_1+\rho\_1s\_1s\_1^T=v\_1^Tv\_0^TG\_0v\_0v\_1+v\_1^T\rho\_0s\_0s\_0^Tv\_1+\rho\_1s\_1s\_1^T$

$G\_3=v\_2^Tv\_1^TG\_2v\_1v\_2+\rho\_2s\_2s\_2^T=v\_2^Tv\_1^Tv\_0^TG\_0v\_0v\_1v\_2+v\_2^Tv\_1^T\rho\_0s\_0s\_0^Tv\_1v\_2+v\_2^T\rho\_1s\_1s\_1^Tv\_2+\rho\_2s\_2s\_2^T$

通过递归式20，可得：

$G\_{k+1}=v\_k^Tv\_{k-1}^T...v\_0^TG\_0v\_0...v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^Tv\_{k-1}^T...v\_1^T\rho\_0 s\_0 s\_0^Tv\_1...v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ ...$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^Tv\_{k-1}^T\rho\_{k-2} s\_{k-2} s\_{k-2}v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^T\rho\_{k-1} s\_{k-1} s\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + \rho\_ks\_ks\_k^T\;(21)$

由式21可以得出，$G\_{k+1}$的计算需要用到$G\_0$,$s\_i$, $y\_i$,其中$i=0,1,2,...k$。而lbfgs算法最关键的点在于，通过使用距离当前迭代最近的$m$个$s$向量和$y$向量，近似求解$G\_{k+1}$。当$k+1<=m$,则根据式21直接求解$G\_{k+1}$, 当$k+1>m$时，只保留最近的$k$个$s$向量和$y$向量,具体计算如式22所示:

$G\_{k+1}=v\_k^Tv\_{k-1}^T...v\_{k-m+1}^TG\_0v\_{k-m+1}...v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^Tv\_{k-1}^T...v\_{k-m+2}^T\rho\_{k-m+1} s\_{k-m+1} s\_{k-m+1}^Tv\_{k-m+2}...v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ ...$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^Tv\_{k-1}^T\rho\_{k-2} s\_{k-2} s\_{k-2}v\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + v\_k^T\rho\_{k-1} s\_{k-1} s\_{k-1}v\_k$

$\ \ \ \ \ \ \ \ \ \ \ \ + \rho\_ks\_ks\_k^T\;(22)$

虽然式21和式22可用于在任何情况下，对hessian矩阵的逆进行迭代求近似解，进而用于lbfgs算法求解。然而，仅仅通过式21和式22，依然需要存储hessian矩阵的逆，并不能节省存储空间。实际上，我们只要能求出$G\_kg\_k$(或$H\_k^{-1}g\_k$)，就可以避开存储完整的$G\_{k+1}$,将存储空间由$O(N^2)$下降至$O(m\times N)$。[2]提供了有效计算$G\_kg\_k$的一个迭代算法，如下所示：

**算法1:**

1） $if\;iter < M: incr = 0, bound= iter$

$\;\;\;\; else \; incr= iter - m, bound = m$
   
2) $q\_{bound} = g\_{iter}$

3) $for \;i = (bound-1), ... , 0$
  
$\;\;\;\;\;\;\;\;j = i + incr$

$\;\;\;\;\;\;\;\;\alpha\_i = \rho\_js\_j^Tq\_{i+1} (存储每个\alpha\_i)$

$\;\;\;\;\;\;\;\;q\_i=q\_{i+1}-\alpha\_iy\_j$

$\;\;\;\;r\_0=G\_0.q\_0$

$\;\;\;\;for\;i=0, 1, ..., (bound - 1)$

$\;\;\;\;\;\;\;\;j=i+incr$

$\;\;\;\;\;\;\;\;\beta\_i = \rho\_jy\_j^Tr\_i$

$\;\;\;\;\;\;\;\;r\_{i+1}=r\_i+s\_j(\alpha\_i-\beta\_i)$


**算法1的证明：**

$q\_{bound}=g\_{iter}$

对于$0<i<bound$,

$q\_i=q\_{i+1}-\alpha\_iy\_i \\ $

$=q\_{i+1}-\rho\_jy\_js\_j^Tq\_{i+1}$

$=(I-\rho\_jy\_js\_j^T)q\_{i+1}$

$=v\_j^Tq\_{i+1}$

$=v\_{inc+i}^Tq\_{i+1}$

$=v\_{inc+i}v\_{inc+i+1}q\_{i+2}$

$=v\_{inc+i}v\_{inc+i+1}v\_{inc+i+2}...v\_{inc+bound-1}q\_{bound}\;(23)$



$\alpha\_i=\rho\_js\_j^Tq\_{i+1}$

$=\rho\_{inc+i}s\_{inc+i}^Tv\_{inc+i+1}v\_{inc+i+2}...v\_{inc+bound-1}q\_{bound}\;(24)$

$r\_0=G\_0q\_0=G\_0v\_{inc}v\_{inc+1}...v\_{inc+bound-1}q\_{bound}(25)$

$r\_{i+1}=r\_i+s\_j(\alpha\_i-\beta\_i)$


$=r\_i+s\_j\alpha\_j-s\_j\rho\_jy\_j^Tr\_i=(I-s\_j\rho\_jy\_j^T)r\_i+s\_j\alpha\_i=v\_{inc+i}^Tr\_i+s\_{inc+i}\alpha\_i(26)$

由式26可得：
$r\_{bound}=s\_{inc+bound-1}\alpha\_{bound-1}+v\_{inc+bound-1}r\_{bound-1}$

$=s\_{inc+bound-1}\rho\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}+v\_{inc+bound-1}r\_{bound-1}$

$=s\_{inc+bound-1}\rho\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}$

$\;\;\;+v\_{inc+bound-1}^T(s\_{inc+bound-2}\alpha\_{bound-2}+v\_{inc+bound-2}^Tr\_{bound-2})$

$=\rho\_{inc+bound-1}s\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}$

$\;\;\;+v\_{inc+bound-1}^T\rho\_{inc+bound-2}s\_{inc+bound-2}s\_{inc+bound-2}^Tv\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^Tr\_{round-2}$

$=\rho\_{inc+bound-1}s\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}$

$\;\;\;+v\_{inc+bound-1}^T\rho\_{inc+bound-2}s\_{inc+bound-2}s\_{inc+bound-2}^Tv\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\rho\_{inc+bound-3}s\_{inc+bound-3}s\_{inc+bound-3}^Tv\_{inc+bound-2}v\_{inc+bound-1}q\_{bound}$


$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^Tv\_{inc+bound-3}^Tr\_{bound-3}$

$=\rho\_{inc+bound-1}s\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}$

$\;\;\;+v\_{inc+bound-1}^T\rho\_{inc+bound-2}s\_{inc+bound-2}s\_{inc+bound-2}^Tv\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\rho\_{inc+bound-3}s\_{inc+bound-3}s\_{inc+bound-3}^Tv\_{inc+bound-2}v\_{inc+bound-1}q\_{bound}$

$\;\;\;...$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\;...\;v\_{inc+1}^T\rho\_{inc}s\_{inc}s\_{inc}^Tv\_{inc+1}\;...\;v\_{inc+bound-2}v\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\;...\;v\_{inc+1}^Tv\_{inc}^Tr\_0$

$=\rho\_{inc+bound-1}s\_{inc+bound-1}s\_{inc+bound-1}^Tq\_{bound}$

$\;\;\;+v\_{inc+bound-1}^T\rho\_{inc+bound-2}s\_{inc+bound-2}s\_{inc+bound-2}^Tv\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\rho\_{inc+bound-3}s\_{inc+bound-3}s\_{inc+bound-3}^Tv\_{inc+bound-2}v\_{inc+bound-1}q\_{bound}$

$\;\;\;...$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\;...\;v\_{inc+1}^T\rho\_{inc}s\_{inc}s\_{inc}^Tv\_{inc+1}\;...\;v\_{inc+bound-2}v\_{inc+bound-1}q\_{bound}$

$\;\;\;+v\_{inc+bound-1}^Tv\_{inc+bound-2}^T\;...\;v\_{inc+1}^Tv\_{inc}^TG\_0v\_{inc}v\_{inc+1}...v\_{inc+bound-1}q\_{bound}$

$=G\_{iter}g\_{iter}$

到此，lbfgs迭代求解证明完毕，其中[1]中实现的lbfgs求解，就是用的该迭代算法。

## OWL-QN算法及求解 

为了减少模型过拟合，我们在进行优化求解时，通常的方式是加入正则项。常见的正则因子包括$l1$正则和$l2$正则。相对于$l2$正则，$l1$正则的优势在于[3]:(1)当大多数特征之间不相关时，$l1$正则在理论和实践上都能够学习更好的模型;(2)$l1$正则能够学到更稀疏的参数空间，有更好的可解释型，在模型计算时能更高效的进行计算。

由于$l1$正则的一阶导数是常数，迭代时使得每个变量尽量被更新为0（$l2$正则是一个比例值，使得每个变量逐渐接近0而不是直接更行为0）。由于$l1$正则在零点不可导，使得基于梯度的优化算法如lbfgs算法无法使用。 针对该问题，Galen Andrew等人提出了OWL-QN(Orthant-Wise Limited-memory Quasi-Newton)算法，用于求解带$l1$正则的log-linear model。

### 相关定义

为方便描述OWL-QN算法，我们做如下一些定义：

$f(x)$对$x\_i$的右导数：$\partial\_i^+=lim\_\{\alpha->0}\frac{f(x+\alpha e\_i)-f(x)}{\alpha}$

$f(x)$对$x\_i$的左导数：$\partial\_i^-=lim\_\{\alpha->0}\frac{f(x)-f(x+\alpha e\_i)}{\alpha}$

其中$e\_i$是第$i$个维度的基向量。

$f(x)$对方向$d$的偏导数：$f′(x;d)=lim\_\{\alpha->0}\frac{f(x+\alpha d)-f(x)}{\alpha}$

符号函数:$\sigma(x\_i) =  \begin{cases} 
1,  & x\_i>0\\\\
-1,  & x\_i<0\\\\
0,  & x\_i=0
\end{cases}
$

象限投影函数:$\pi(x\_i,y\_i) =  \begin{cases} 
x\_i,  & \sigma(x\_i) = \sigma(y\_i)\\\\
0,  & otherwise
\end{cases}
$

### OWL-QN算法

**基于象限建模**

考虑L1正则，要求解的目标函数为：

$F(x)=f(x)+C ||x||\_{1}\;(27)$

其中$f(x)$为原始损失，$C ||x||\_{1}$为正则惩罚。

对于包含$L1$正则目标函数，当数据点集合在某个特定的象限内部（所有维度的符号保持不变），它是可导的。$L1$正则部分是参数的线性函数，且目标函数的二阶导数只取决于原始损失(不包括正则)的部分。基于这点，对于目标函数，可构建包括当前点的某个象限二阶泰勒展开（固定该象限时梯度可以求解，hessian矩阵只根据原始损失部分求解即可），并限制搜索的点，使得迭代后参数对应象限对于当前的近似依然是合法的。

对于向量$\varepsilon \in \lbrace -1, 0 , 1 \rbrace ^n$, 我们定义其对应象限区域为：

$\Omega\_\varepsilon=\lbrace x \in R^n: \pi(x;\varepsilon)=x\rbrace$

对于该象限内的任意点$x$，$F(x)=f(x)+C \varepsilon^Tx\;(28)$

我们在式28基础上，扩展定义$F\_\varepsilon$为定义在$R^n$上函数，在每个象限具有和$R\_\varepsilon$空间类似的导数。通过损失函数的hessian矩阵的逆$H\_k$，以及$F\_\varepsilon$的负梯度在$\Omega\_\varepsilon$的投影$v^k$，可以近似$F\_\varepsilon$在$\Omega\_\varepsilon$的投影。为迭代求$F\_\varepsilon$最小值，出于技术原因，限制搜索的方向和$v^k$所在象限一致。

$p^k=\pi(H\_kv^k;v^k)$

**选择投影象限：**

为了选择投影的象限，我们定义伪梯度：

$\diamond\_iF(x)=\begin{cases} 
\partial\_i^{-}F(x),  & if\;\partial\_i^{-}F(x)>0\\\\
\partial\_i^{+}F(x),  & if\;\partial\_i^{+}F(x)<0\\\\
0,  & otherwise
\end{cases}\;(29)
$

其中，$\partial\_i^{+/-}F(x)$定义如下：
$\partial\_i^{+/-}F(x)=\frac{\partial}{\partial x\_i} f(x)+\begin{cases}
C \sigma(x\_i) & if\;x\_i\neq 0\\\\
+/-C & if\;x\_i=0
\end{cases}\;(30)$

由式30可得，$\partial\_i^{-}F(x)\leq \partial\_i^{+}F(x)$，因此式29能够精确定义。伪梯度是对梯度信息的泛化，$x$是极小值的充要条件是$\diamond\_iF(x)=0$

一个合理的象限选择可以定义如下：

$\varepsilon\_i^k=\begin{cases}\sigma(x\_i^k) &if(x\_i^k\neq0)\\\\
\sigma(-\diamond\_iF(x)) & if (x\_i^k = 0)
\end{cases}\;(31)$

这样选择象限的理由是：-$\diamond\_iF(x)$和$F\_\varepsilon$的负梯度在$\Omega\_\varepsilon$的投影$v^k$相等。因此，在利用owl-qn算法求解时，并不需要显示的计算$\varepsilon\_i$,直接计算$-\diamond\_iF(x)$, 就等价于按照式31设置$\varepsilon$,并代入式28求解梯度的投影。

**有约束的线性搜索**

为了确保每次迭代没有离开合法的象限空间，owl-qns算法对搜索的点重新投影到象限$\Omega\_\varepsilon$，对于符号发生变化的每个维度，均置为0.如式32所示。

$x\_{k+1}=\pi(x^k+\alpha p^k; \varepsilon^k)\;(32)$

有很多的线性搜索方法，[3]采用的方法是：

**算法1:有约束的线性搜索**

(1) $设置\;\beta,\gamma \in (0,1)$

(2) $for\;\;n = 0, 1, 2...$

$\;\;\;\;\;\;\;\;\;\; \alpha=\beta^n$
  
$\;\;\;\;\;\; \;\;\;\;if\;\;f(x^{k+1})\leq f(x^k)-\gamma v^T(x^{k+1}-x^k)$

$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;找到下个最优解 $
    
$\;\;\;\;\;\; \;\;\;\;else\;\;continue$

**owl-qn算法**

owl-qn算法同普通的lbfgs算法基本相同，不同之处主要在于：（1）需要计算伪梯度；（2）搜索方向对$v^k$对应的象限做做投影；（3）搜索的点需要限制在上次迭代点对应的象限。（4）目标函数的非正则部分的梯度用于更新$y$向量集合,而不是用伪梯度去更新$y$向量集合。

**算法2:owl-qn算法描述**

$初始化x\_0,s=\lbrace\rbrace,y=\lbrace\rbrace$

$for\;\; k = 0 \;to \;MaxIters$

$\;\;\;\;计算梯度v^k=-\diamond f(x^k)$

$\;\;\;\;通过s,y向量集合,计算d^k=H\_kv^k$

$\;\;\;\;p^k=\pi(d^k;v^k)$

$\;\;\;\;根据算法1求解x\_{k+1}$

$\;\;\;\;如果达到终止条件，则终止算法，否则更新s^k=x\_{k+1}-x\_{k},y\_{k+1}=\triangledown f(x^{k+1})-\triangledown f(x^{k}) 向量集合$

## LBFGS在liblbfgs开源库的实现

本章节主要介绍LBFGS算法在liblbfgs开源库[1]的实现，[1]不仅实现了普通的lbfgs算法，也实现了上个章节介绍的owl-qn算法。

**相关数据结构:**
```c++
//定义callback_data_t结构
struct tag_callback_data {
    int n;  //变量个数
    void *instance; //实例
    lbfgs_evaluate_t proc_evaluate; //计算目标函数及梯度的回调函数
    lbfgs_progress_t proc_progress; //接受优化过程进度的的回调函数
};
typedef struct tag_callback_data callback_data_t;

//定义iteration_data_t，存储lbfgs迭代需要的s,y向量
struct tag_iteration_data {
    lbfgsfloatval_t alpha;  //算法1迭代需要的alpha变量
    lbfgsfloatval_t *s;     //x(k+1) - x(k)
    lbfgsfloatval_t *y;     //g(k+1) - g(k)
    lbfgsfloatval_t ys;     //vecdot(y, s)
};
typedef struct tag_iteration_data iteration_data_t;

//定义lbfgs参数
static const lbfgs_parameter_t _defparam = {
    6, 1e-5, 0, 1e-5,
    0, LBFGS_LINESEARCH_DEFAULT, 40,
    1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
    0.0, 0, -1,
};

```

**lbfgs算法:**
```c++
//lbfgs算法求解核心过程，为描述lbfgs算法核心流程，此处只保留主要代码
int lbfgs(
    int n, //变量个数
    lbfgsfloatval_t *x, //变量值
    lbfgsfloatval_t *ptr_fx, // 函数值
    lbfgs_evaluate_t proc_evaluate, //计算目标函数及梯度的回调函数
    lbfgs_progress_t proc_progress, //接受优化过程进度的的回调函数
    void *instance, //实例变量
    lbfgs_parameter_t *_param //lbfgs优化永的的参数变量
    )
{
    ... 
    //构建callback_data_t
    callback_data_t cd;
    cd.n = n; //参数的维度
    cd.instance = instance; //实例变量
    cd.proc_evaluate = proc_evaluate; //计算目标函数及梯度的回调函数
    cd.proc_progress = proc_progress; //接受优化过程进度的的回调函数
   ...
    /* Allocate working space. */
    xp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));//上次迭代的变量值
    g = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));//本次迭代对应的梯度值
    gp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));//上次迭代的梯度址
    d = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));//迭代方向变量
    w = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
    //对l1正则，分配OW-LQN算法伪梯度需要的存储空间 */
    if (param.orthantwise_c != 0.) {
        pg = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
        if (pg == NULL) {
            ret = LBFGSERR_OUTOFMEMORY;
            goto lbfgs_exit;
        }
    }    
    //最近m次迭代相关向量的存储
    lm = (iteration_data_t*)vecalloc(m * sizeof(iteration_data_t));
    if (lm == NULL) {
        ret = LBFGSERR_OUTOFMEMORY;
        goto lbfgs_exit;
    }    
    //最近m次迭代相关向量的初始化
    for (i = 0;i < m;++i) {
        it = &lm[i];
        it->alpha = 0;
        it->ys = 0;
        it->s = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
        it->y = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
        if (it->s == NULL || it->y == NULL) {
            ret = LBFGSERR_OUTOFMEMORY;
            goto lbfgs_exit;
        }
    }
    //最近的m次迭代的目标函数值
    if (0 < param.past) {
        pf = (lbfgsfloatval_t*)vecalloc(param.past * sizeof(lbfgsfloatval_t));
    }
    //计算目标函数的值和梯度
    fx = cd.proc_evaluate(cd.instance, x, g, cd.n, 0);
    //如果有l1正则，计算带l1正则的目标函数值和伪梯度信息 
    if (0. != param.orthantwise_c) {
        //有l1正则，计算l1正则对应的norm
        xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
        //将l1z正则对应的值加入目标函数
        fx += xnorm * param.orthantwise_c;
        //计算伪梯度信息
        owlqn_pseudo_gradient(
            pg, x, g, n,
            param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
            );
    }
    //存储目标函数值到pf[0]
    if (pf != NULL) {
        pf[0] = fx;
    }

    //存储迭代方向到d变量, 假定原始hessian矩阵G0为单位矩阵，G0 g = g
    if (param.orthantwise_c == 0.) {
        vecncpy(d, g, n);
    } else {
        vecncpy(d, pg, n);
    }
    
    //通过比较g_norm / max(1, x_norm)是否小于param.epsilon，确定是否已经达到极小值
    vec2norm(&xnorm, x, n);
    if (param.orthantwise_c == 0.) {
        vec2norm(&gnorm, g, n);
    } else {
        vec2norm(&gnorm, pg, n);
    }
    if (xnorm < 1.0) xnorm = 1.0;
    if (gnorm / xnorm <= param.epsilon) {
        ret = LBFGS_ALREADY_MINIMIZED;
        goto lbfgs_exit;
    }
    //初始化最优步长 step: 1.0 / sqrt(vecdot(d, d, n)) */
    vec2norminv(&step, d, n);
    k = 1;
    end = 0;
    for (;;) {
        veccpy(xp, x, n);//存储变量值到xp
        veccpy(gp, g, n);//存储梯度值到gp
        /* Search for an optimal step. */
        if (param.orthantwise_c == 0.) {//无l1正则，在d方向搜索最优解
            ls = linesearch(n, x, &fx, g, d, &step, xp, gp, w, &cd, &param);
        } else { //有l1正则，在d方向搜索最优解
            ls = linesearch(n, x, &fx, g, d, &step, xp, pg, w, &cd, &param);
            //计算伪梯度
            owlqn_pseudo_gradient(
                pg, x, g, n,
                param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
                );
        }
        //达到终止条件
        if (ls < 0) {
            /* Revert to the previous point. */
            veccpy(x, xp, n);
            veccpy(g, gp, n);
            ret = ls;
            goto lbfgs_exit;
        }

        /* Compute x and g norms. */
        //计算x范数，g范数
        vec2norm(&xnorm, x, n);
        if (param.orthantwise_c == 0.) {
            vec2norm(&gnorm, g, n);
        } else {
            vec2norm(&gnorm, pg, n);
        }

        //输出进度信息
        if (cd.proc_progress) {
            if ((ret = cd.proc_progress(cd.instance, x, g, fx, xnorm, gnorm, step, cd.n, k, ls))) {
                goto lbfgs_exit;
            }
        }

        //收敛测试， |g(x)| / \max(1, |x|) < \epsil
        if (xnorm < 1.0) xnorm = 1.0;
        if (gnorm / xnorm <= param.epsilon) {
            ret = LBFGS_SUCCESS;
            break;
        }

        //以past为周期，根据当前函数值和1个周期之前的函数值判断是否停止迭代
        //停止条件：(f(past_x) - f(x)) / f(x) < \delta
        if (pf != NULL) {
            /* We don't test the stopping criterion while k < past. */
            if (param.past <= k) {
                /* Compute the relative improvement from the past. */
                rate = (pf[k % param.past] - fx) / fx;
                /* The stopping criterion. */
                if (rate < param.delta) {
                    ret = LBFGS_STOP;
                    break;
                }
            }
            /* Store the current value of the objective function. */
            pf[k % param.past] = fx;
        }
        //达到最大迭代次数
        if (param.max_iterations != 0 && param.max_iterations < k+1) {
            /* Maximum number of iterations. */
            ret = LBFGSERR_MAXIMUMITERATION;
            break;
        }

        //更新向量s, y  s_{k+1} = x_{k+1} - x_{k}，y_{k+1} = g_{k+1} - g_{k}
        it = &lm[end];
        vecdiff(it->s, x, xp, n);
        vecdiff(it->y, g, gp, n);

        vecdot(&ys, it->y, it->s, n); //ys = y^t \cdot s; 1 / \rho
        vecdot(&yy, it->y, it->y, n); //yy = y^t \cdot y
        it->ys = ys;// y^t \cdot s

        /*
           Recursive formula to compute dir = -(H \cdot g).
               This is described in page 779 of:
               Jorge Nocedal.
               Updating Quasi-Newton Matrices with Limited Storage.
               Mathematics of Computation, Vol. 35, No. 151,
               pp. 773--782, 1980.
        */
        //根据文献[1]中算法（对应本文算法1），计算 -(G \cdot g)
        bound = (m <= k) ? m : k;
        ++k;
        end = (end + 1) % m;

        /* Compute the steepest direction. */
        if (param.orthantwise_c == 0.) {
            /* Compute the negative of gradients. */
            vecncpy(d, g, n);
        } else {
            vecncpy(d, pg, n);
        }

        j = end;
        for (i = 0;i < bound;++i) {
            j = (j + m - 1) % m;    /* if (--j == -1) j = m-1; */
            it = &lm[j];
            /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
            vecdot(&it->alpha, it->s, d, n);
            it->alpha /= it->ys;
            /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
            vecadd(d, it->y, -it->alpha, n);
        }
        vecscale(d, ys / yy, n);

        for (i = 0;i < bound;++i) {
            it = &lm[j];
            /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
            vecdot(&beta, it->y, d, n);
            beta /= it->ys;
            /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
            vecadd(d, it->s, it->alpha - beta, n);
            j = (j + 1) % m;        /* if (++j == m) j = 0; */
        }        

```


# 参考资料
[1] chokkan, https://github.com/chokkan/liblbfgs

[2] Jorge Nocedal, Updating Quasi-Newton Matrices With Limited Storage

[3] Galen Andrew, Jianfeng Gao, Scalable Training of L1-Regularized Log-Linear Models

[4] 皮果提, http://blog.csdn.net/itplus/article/details/21896453

[5] http://blog.sina.com.cn/s/blog_eb3aea990101gflj.html

[6] https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

