---
date: 2019-12-23 20:35
status: public
title: XGBOOST
---

[TOC]

# Boosting: 加法模型+前向分步算法

boosting实际就是加法模型（即基函数的线性组合）+ 前向分步算法。

加法模型为
$$
\hat y_i=\sum_{m=1}^{M}f_m(x_i)
$$
其中，$f_m(x)$为基函数。

在给定训练数据和损失函数$L(y_i,\hat y_i))$的条件下，学习加法模型成为**经验风险最小化**即损失函数最小化问题：
$$
\mathop{\min}\_{}\sum_{i=1}^{N}L\left (y_i,\sum_{m=1}^Mf_m(x_i)  \right)
$$
通常这是一个复杂的优化问题，前向分步算法求解这一优化问题的想法是：因为学习的是加法模型，如果能从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式，那么就可以简化优化的复杂度，具体的，每一步只需极小化如下损失函数就可以得到每一步的$f_m$，这样，前向分步算法将同时求解从m=1到M所有模型$f(x)$的优化问题转化为逐次求解各个$f(x)$的优化问题。

 在前向分步算法的第m步，给定当前模型$\sum_{t=1}^{m-1}f_t(x_i)$，需求解
$$
f_m=\mathop{\arg \min}\_{f}\sum_{i=1}^{N}L\left (y_i,\sum_{t=1}^{m-1}f_t(x_i)+f(x_i)  \right)
$$

$$
\hat{\Theta}\_m=\mathop{\arg \min}\_{\Theta}\sum_{i=1}^{N}L\left (y_i,F_{m-1}(x_i)+f_m(x_i;\Theta)\right)
$$

得到$\hat{\Theta}_m$，即第m个基模型的参数。

# Gradient Boosting: 加法模型+梯度提升(逐步优化)

如果通过上式无法求得第m步模型$f_m$ 的闭式解，那么我们可以借鉴梯度下降的思想，只要保证第m轮的损失比第m-1轮的小，那么经过一定次数T的迭代后，就可以保证损失达到极小，如果是凸函数，就可以使损失达到最小值。即第m步的模型为：
$$
f_m(x_i)=-\left [ \frac{\partial L(y_i,\sum_{t=1}^{m-1}f_t(x_i)}{\partial \sum_{t=1}^{m-1}f_t(x_i) }\right ]
$$
如果基模型$f_m(x)$为决策树的话，就是GBDT。

# Newton Boost: XGBoost

## 目标函数

xgboost中，优化目标由经验风险最小化变成结构风险最小化：
$$
\mathop{\min}\_{}\sum_{i}L\left (y_i,\sum_{m=1}^Mf_m(x_i)  \right)+\sum_{m}\Omega (f_m)
$$
再来看正则项$\Omega$ , 这里必须设定其为如下具体形式，注意这个具体形式是跟后面的二阶泰勒展开紧密联系的，如果没有这样的而具体形式，后面就无法推导出叶子节点的权重$w_j$的解析解。
$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2
$$
加入正则项，是XGBoost相对于传统GBDT的一个改良。后一项容易理解：叶子节点的L2-norm，常见的惩罚项设置方法，不多说了；而前一项$\gamma T$  则是XGBoost的一个主要Contribution: **传统GBDT模型中没有正则项，而是通过剪枝等“启发式”(heuristic，其实就是拍脑袋)方法来控制模型复杂度。后面的推导会证明，gamma参数的作用等同于预剪枝，这就将难以解释的启发式方法纳入到了标准的 supervised leanring 框架中，让我们更清楚的知道what you are learning**。

对于一般的损失函数$l(y_i, \hat{y_i})$来说，与梯度下降（一阶泰勒展开）不同的是，我们对$l(y_i, \hat{y_i})$在当前模型$f(x)=\sum_{m=1}^{t-1}f_m(x)$处做二阶泰勒展开，
$$
obj^{(t)}=\sum_{i=1}^{n}[l(y_i,\hat{y_i}^{(t-1)})+g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\Omega (f_t)+constant
$$
其中，
$$
\begin{align} 
g_i=\frac{\partial l(y_i,\hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}}\\ 
h_i=\frac{\partial^2 l(y_i,\hat{y_i}^{(t-1)})}{\partial (\hat{y_i}^{(t-1)})^2}\ 
\end{align}
$$
分别表示损失函数在当前模型的一阶导和二阶导，每一个样本都可以计算出该样本点的$g_i$和$h_i$，而且样本点之间的计算可以独立进行，互不影响，也就是说，可以并行计算。

对上式进行化简，去掉与$f_t$无关的项，得到
$$
obj^{(t)}=\sum_{i=1}^{n}[g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\Omega (f_t)
$$
上式就是每一步要优化的目标函数。

## 重写目标函数

每一步学到的CART树可以表示成
$$
f_m(x) = w_{q(x)} , w \in R^T , q : R^d \rightarrow  { 1,2,...,T  }
$$
其中T为叶子节点个数，$q(x)$是一个映射，用来将样本映射成1到T的某个值，也就是把它分到某个叶子节点，$q(x)$其实就代表了CART树的结构。$w_{q(x)}$自然就是这棵树对样本x的预测值了。

因此，树的复杂度可以表示为：
$$
\Omega(f)=\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2
$$
将复杂度$\Omega(f)$代入目标函数并做变形，得到
$$
\begin{align} 
obj^{(t)}&=\sum_{i=1}^{n}[g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}]+\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2\\ 
&=\sum_{j=1}^{T}[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2]+\gamma T 
\end{align}
$$
$I_j$代表一个集合，集合中每个值代表一个训练样本的序号，整个集合就是被第t棵CART树分到了第j个叶子节点上的所有训练样本。令$G_j=\sum_{i\in I_j}g_i$和$H_j=\sum_{i\in I_j}h_i$，目标函数可以写为
$$
obj^{(t)}=\sum_{j=1}^{T}[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2]+\gamma T
$$
对于第t棵CART树的某一个确定的结构（可用q(x)表示），所有的$G_j$和$H_j$都是确定的。而且上式中各个叶子节点的值$w_j $之间是互相独立的。上式其实就是一个简单的一元二次式，我们很容易求出各个叶子节点的最佳值以及此时目标函数的值：
$$
w_j^\*=-\frac{G_j}{H_j +\lambda}
$$

$$
obj^\*=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+1}+\gamma T
$$

$obj$表示了这棵树的结构有多好，值越小的话，代表这样结构越好！也就是说，它是衡量第t棵CART树的结构好坏的标准。这个值仅仅是用来衡量结构的好坏的，与叶子节点的值是无关的，因为$obj^ *$只和$G_j $和$H_j $和T有关，而它们又只和树的结构(q(x))有关，与叶子节点的值可是半毛关系没有。

Note：这里，我们对$w_j $给出一个直觉的解释，以便能获得感性的认识。我们假设分到j这个叶子节点上的样本只有一个。那么，$w_j $就变成如下这个样子：
$$
w_j=\underbrace{\frac{1}{h_j+\lambda}}\_{学习率}\cdot \underbrace{-g_j}_{负梯度}
$$
这个式子告诉我们，$w_j $的最佳值就是负的梯度乘以一个权重系数，该系数类似于随机梯度下降中的学习率。观察这个权重系数，我们发现，$h_j$越大，这个系数越小，也就是学习率越小。$h_j$越大代表什么意思呢？代表在该点附近梯度变化非常剧烈，可能只要一点点的改变，梯度就从10000变到了1，所以，此时，我们在使用反向梯度更新时步子就要小而又小，也就是权重系数要更小。

## 最优树结构

现在的问题变成了：如何得到树结构？ 这是一个NP-hard问题，无法在多项式时间内求解，因此我们采取贪心算法求解：从根节点开始，遍历所有可能的 (feature, split)，找到使loss reduction最大的那个，作为树的下一步生长方式，Loss reduction（Gain）的计算方式为：
$$
Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma
$$
这个Gain实际上就是单节点的$obj$减去切分后的两个节点的树$obj$，Gain如果是正的，并且值越大，表示切分后$obj$越小于单节点的$obj$，就越值得切分。同时，我们还可以观察到，Gain的左半部分如果小于右侧的$\gamma$，则Gain就是负的，表明切分后$obj$反而变大了。$γ$在这里实际上是一个临界值，它的值越大，表示我们对切分后$obj$下降幅度要求越严。

## 思考

Q:xgb怎么处理缺失值？

A：xgb处理缺失值的方法和其他树模型不同,xgboost把缺失值当做稀疏矩阵来对待，本身的在节点分裂时不考虑的缺失值的数值。缺失值数据会被分到左子树和右子树分别计算损失，选择较优的那一个。如果训练中没有数据缺失，预测时出现了数据缺失，那么默认被分类到右子树。

PS:随机森林怎么处理缺失值？

1. 数值型变量用中值代替，类别型变量用众数代替。
2. 引入了权重。即对需要替换的数据先和其他数据做相似度测量,补全缺失点是相似的点的数据会有更高的权重W

Q:xgb为什么用二阶展开项？

**二阶导其实可以看成是梯度下降的学习率，二阶导反映了梯度变化的情况，二阶导越大，梯度变化越剧烈，那么学习的时候越需要小心！！！**

Q:如何保证一元二次式（7）是开口向上的呢？即如果12(Hj+λ)<012(Hj+λ)<0,那么不是岂可以使损失函数达到无限小？

A:这个二阶导数值一定是大于0的，因为如果二阶导不大于0，那么损失函数就不是凸函数，也就没有最小值，（损失函数要大于0），而xgb要求自定义损失函数要二阶可导，因此二阶导数值必须要大于0

Q:xgb多分类是怎么处理的？每个类别的概率是怎么来的！！！！！？？？？

A:one vs all 有n个类别，就训练n个模型，最后将这n个模型的输出做softmax归一化

## xgb调包侠

### Dmatrix和Booster

xgboost的数据结构（类）只有两个： `DMarix` 和 `Booster` 。 `DMatrix`是为XGB算法专门设计的、高效存储和计算的数据结构，而`Booster`顾名思义是boosting集成模型对象。可以通过原生的Learning API `xgb.train()` 得到。booster对象有一个`.trees_to_dataframe()` 方法，将集成树模型的信息返回成一个DataFrame，包括树的分裂节点、每个节点的Gain, Cover （cover表示该节点影响的样本数量）。注意叶子节点上的Gain字段值，表示的是叶子节点权重，即解析解$w_j^\*=-\frac{G_j}{H_j +\lambda}$  , 与`plot_tree()` 中叶子节点的值一致：

我们将所有树相加（对于每个样本，在每个树的叶子节点score相加），再加上初始值$f_0(x)$ (对应的参数为`base_score`，默认值0.5），就得到了 `bst.predict()` 返回的值。

### feature_importance

**xgb特征重要性有三种不同的计算方式：**

- `weight`**注意这里的权重既不是 leaf-node-score, 也不是二阶导数，而是定义为“该特征用于分裂的次数”。** 
- `gain` 前面的公式中的 loss reduction
- `cover` 定义为”受其影响的样本数数量“

```python
xgb.plot_importance(bst,importance_type='cover',ax=ax)
```

### min_child_weight

`min_child_weight`这个参数虽然直接调的不多，但它的推导和数学意义却很有趣。它的含义是节点上所有样本的在当前模型输出处的二阶导数值之和 $H_j=\sum_{i\epsilon I_j}h_i$，这个值必须大于一定的阈值才继续分裂。但为什么是$H_j$呢？

原因可以有2个。

1.$w_j=\underbrace{\frac{1}{h_j+\lambda}}\_{学习率}\cdot \underbrace{-g_j}_{负梯度}$，二阶导数大，说明一阶导数变化率快，所以需要继续学习。

2.将$obj^{(t)}$改写成如下形式，
$$
obj^{(t)}=\sum_{i=1}^{n}\frac{1}{2}h_i(f_t(x_i)-\frac{g_i}{h_i})^2+\Omega (f_t)
$$
会发现，（lambda=0情况下）我们在做的事情是 “以二阶导$h_i$为权重，以$\frac{g_i}{h_i}$为真实label，以RMSE最小化为目标，拟合$f_t(x)$"。而$f_t(x)$的解，跟我们之前推导出来的$w_j $解析解也是一致的。另外再多说一句，如果你之前熟悉牛顿法，会发现这个形式
$$
f_t(x_i)=w_j^\*=-\frac{G_j}{H_j}
$$
跟牛顿法参数更新公式
$$
\theta_t=-H_t^{-1}G_t
$$
有神奇的相似之处。因此也有人把XBG这种方法称之为 Newton Boosting。

**感觉这两种解释其实是一个意思。**

### random forest

XGB可以用来训练随机森林。随机森林模型与boosting tree 形式上完全相同，只需要设置三个参数`num_parallel_trees=100`， `num_boosting_round=1, learning_rate=1`，`colsample_by_*` < 1 即可

## 参考文献

[深入xgb](https://zhuanlan.zhihu.com/p/91817667)

[xgboost如何使用MAE或MAPE作为目标函数?](https://zhuanlan.zhihu.com/p/34373008)

[How to get or see xgboost's gradient statistics value?](https://stackoverflow.com/questions/44916391/how-to-get-or-see-xgboosts-gradient-statistics-value?r=SearchResults)