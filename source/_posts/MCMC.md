---
title: 蒙特卡罗采样
date: 2022-11-20 17:22:57
tags: 
    - 采样算法

categories: 
    - 基础知识
    - 白板系列

mathjax: true
---

# 1. 背景知识

## 1.1 采样的动机

1. 采样本身就是常见的任务，例如从生成模型 $P(\mathbf{x})$ 中采样出样本。
2. 求积分或者复杂求和。例如，求 $\int p(\mathbf{x})f(\mathbf{x})dx$ 可以转化为 $\mathbb{E}_{x \sim p(\mathbf{x})} \left[f(\mathbf{x})\right]$，从 $p(\mathbf{x})$ 中采样出 $N$ 个样本，求解 $\frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{i})$

> 根据李航老师的《统计学习方法》，假设多元随机变量 $\mathbf{x}\in\mathcal{X}$， 其概率密度函数为 $p(\mathbf{x})$， $f(\mathbf{x})$ 为定义在 $\mathbf{x}\in\mathcal{X}$ 上的函数，采样的目标是获得概率分布为 $p(\mathbf{x})$ 的样本集合，以及求函数 $f(\mathbf{x})$ 的数学期望 $\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})]$。

## 1.2 如何衡量采样的质量？（什么是好的样本？）

1. 样本趋向于高概率区域，同时兼顾其他区域。以高斯分布为例，我们希望绝大多数样本位于 $3\sigma$ 之间。
2. 样本之间要相互独立的，即相关性不强。

以选举为例，我们希望按照人数比例选举代表。例如绝大多数是工人和农民，我们希望绝大多数代表是工人和农民（性质1）；而且这些代表应该是不同工厂和农场的，不是同一家工厂或同一个家族的（性质2）。

## 1.3 采样是困难的

1. 配分函数是不可解的。即 $p(\mathbf{x}) = \frac{\hat{p}(\mathbf{x})}{Z} = \frac{\hat{p}(\mathbf{x})}{\int \hat{p}(\mathbf{x}) d \mathbf{x}}$
2. 高维环境下，状态空间过于复杂，无法通过遍历的方式覆盖所有状态空间。直接采样不可行。

# 2. 一些采样方法介绍

回想推断问题，推断问题可以分为以下几种：

+ 精确推断
+ 近似推断
    - 确定性近似 （变分推断）
    - 随机近似 （MCMC）

例如观测数据 $X$ 是根据隐变量 $Z$ 生成的，我们希望根据观测变量推断隐变量的分布 $P(Z|X)$ 并求解 $\mathbb{E}_{\mathbf{z}}\left[f(\mathbf{z})\right]$。通常我们可以通过积分的方式来实现：
$$\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})] = \int p(\mathbf{z}|\mathbf{x})f(\mathbf{z}) d \mathbf{z} \tag{1}$$
通过从分布 $p(\mathbf{z}|\mathbf{x})$ 中采样 N个点，求解 $\frac{1}{N} \sum_{i=1}^N f(\mathbf{z}^{(i)})$。所以问题就转化成如何从分布中采样出N个样本。

## 2.1 概率分布采样

对于概率分布函数 $p(\mathbf{z})$ （PDF），可以计算其CDF，其CDF属于[0,1]上的均匀分布，可以从[0,1]中采样N个点，并根据CDF的逆计算这N个点对应的输入值 $\mathbf{z}$。

这种方法的缺陷在于CDF难以求解，适用情况少。

## 2.2 接受-拒绝采样

![图1 接受拒绝采样](./MCMC/AccRejSamp.png)

接受-拒绝采样的核心思想是对于难以直接采样的分布 $p(\mathbf{z})$， 构造易于采样的 proposal distribution （建议分布） $q(\mathbf{z})$，使得 $kq(\mathbf{z})$ 处处不小于 $p(\mathbf{z})$。采样时，如果采到的样本位于灰色阴影区域，就拒绝，否则接受。我们可以定义接受率 $\alpha=\frac{p(\mathbf{z}^{(i)})}{kq(\mathbf{z}^{(i)})}$，接受率 $\alpha \in [0,1]$。

具体采样步骤如下：

1). 从分布$q$中采样出样本 $\mathbf{z}^{(i)}$ 
2). 从均匀分布 $U(0,1)$ 中采样 $u_i$
3). 如果 $u_i \le \alpha$ 接受样本，否则拒绝

我们可以看出，建议分布 $q$ 越接近 $p$， 接受率越高。该方法存在的主要问题在于一是合适的q(x)分布比较难以找到，二是很难确定一个合理的 k 值。这两个问题会导致拒绝率很高，无用计算增加。

## 2.3 重要性采样
重要性采样不是从概率分布中采样，而是从期望中采样。根据公式（1），我们可以做如下变换：
$$\begin{split}
\mathbb{E}_{p(\mathbf{z})} [f(\mathbf{z})] &= \int p(\mathbf{z}) f(\mathbf{z}) d \mathbf{z} \\
&= \int \frac{p(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z}) f(\mathbf{z}) d \mathbf{z} \\
&= \int f(\mathbf{z}) \frac{p(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z}) d\mathbf{z} \\
& \approx \frac{1}{N} \sum_{i=1}^N f(\mathbf{z}^{(i)})\frac{p(\mathbf{z}^{(i)})}{q(\mathbf{z}^{(i)})}
\end{split}\tag{2}$$
重要性体现在系数 $\frac{p(\mathbf{z}^{(i)})}{q(\mathbf{z}^{(i)})}$ 上，标明二者的分布越接近，权重越大。

### 2.3.1 Sampling-Importance-Resampling
由于直接使用重要性采样效率也可能很低。因此可以将系数 $\frac{p(\mathbf{z}^{(i)})}{q(\mathbf{z}^{(i)})}$ 看作权重，对采出来的样本进行重采样。简单理解就是，权重越大，样本重复次数越多。

# 3. 马尔可夫链

## 3.1 基本定义

假设在时刻0的随机变量 $\mathbf{x}_0$ 遵循概率分布 $P(\mathbf{x}_0)=\pi_0$，称为初始状态分布，在某个时刻 $t \ge 1$ 的随机变量 $\mathbf{x}_t$ 与前一个时刻的随机变量 $\mathbf{x}_{t-1}$ 之间有条件分布 $P(\mathbf{x}_t |\mathbf{x}_{t-1})$， 如果 $\mathbf{x}_t$ 只依赖于 $\mathbf{x}_{t-1}$， 而不依赖于过去的随机变量 $\{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{t-2}\}$，这一性质称为马尔可夫性，即：

$$P(\mathbf{x}_t|\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_{t-1}) = P(\mathbf{x}_t |\mathbf{x}_{t-1}) \quad t=1,2,\cdots \tag{3}$$

具有马尔可夫性的随机序列 $\mathbf{X} = \{\mathbf{x}_0,\mathbf{x}_1, \cdots, \mathbf{x}_{t}, \cdots\}$ 称为马尔可夫链（Markov Chain）或者马尔可夫过程（Markov Process）。

马尔可夫性的直观解释是“未来只依赖于现在（假设现在已知），而与过去无关”。

## 3.2 离散状态马尔可夫链

### 3.2.1 转移概率矩阵和状态分布

离散状态马尔可夫链 $\mathbf{X} = \{\mathbf{x}_0,\mathbf{x}_1, \cdots, \mathbf{x}_{t}, \cdots\}$， 随机变量 $\mathbf{x}_t (t=0,1,2,\cdots)$ 定义在离散空间 $\mathcal{S}$，转移概率分布可以用矩阵表示。如果马尔可夫链在时刻 $(t-1)$ 处于状态 $i$, 在 $t$ 时刻移动到状态 $j$， 则转移概率记作：
$$p_{ij} = (\mathbf{x}_t = j | \mathbf{x}_{t-1} = i), \quad i=1,2,\cdots; \quad j=1,2,\cdots \tag{4}$$
满足
$$p_{ij}\ge0, \sum_j p_{ij} = 1$$
转移概率矩阵可以写为：
$$\boldsymbol{P} = \left[\begin{matrix}
p_{11} & p_{12} & p_{13} & \cdots \\
p_{21} & p_{22} & p_{23} & \cdots \\
p_{31} & p_{32} & p_{33} & \cdots \\
\cdots & \cdots & \cdots & \cdots \\
\end{matrix}\right] \tag{5}$$

转移概率矩阵 $\boldsymbol{P}$ 满足公式（2）性质，且满足这两个条件的矩阵通常被称为随机矩阵（Stochastic Matrix）。注意，这里矩阵行之和为1，表示从状态 $i$ 转换至状态空间 $\mathcal{S}$ 中任意状态的概率之和为1. 

将马尔可夫链在 $t$ 时刻的概率分布称为 $t$ 时刻的状态分布，记作：
$$\pi(t)=\left[\begin{matrix}
\pi_1(t)\\
\pi_2(t)\\
\vdots
\end{matrix}\right] \tag{6}$$
其中 $\pi_i(t)$ 表示时刻 $t$ 状态为 $i$ 的概率 $P(\mathbf{x}_t = i)$，
$$\pi_i(t) = P(\mathbf{x}_t = i) \quad i=1,2,\cdots \tag{7}$$
这里 $\sum_i \pi_i(t) = 1$。特别地，马尔可夫链的初始状态分布可以表示为
$$\pi(0)=\left[\begin{matrix}
\pi_1(0)\\
\pi_2(0)\\
\vdots
\end{matrix}\right] \tag{8}$$

马尔可夫链 $X$ 在 $t$ 时刻的状态分布，可以由 $(t-1)$ 时刻的状态分布及转移概率分布决定：
$$\pi(t) = P\pi(t-1) \tag{9}$$
这是因为
$$\begin{split}
\pi_i(t) &= P(\mathbf{x}_t = i) \\
&= \sum_{m\in \mathcal{S}} P(\mathbf{x}_t = i|\mathbf{x}_{t-1}=m) P(\mathbf{x}_{t-1} = m) \\
&= \sum_{m\in\mathcal{S}} p_{mi} \pi_m(t-1) 
\end{split}\tag{10}$$

马尔可夫链在 $t$ 时刻的状态分布可以通过递推得到，根据公式（7），我们可以得到
$$\pi(t)=\boldsymbol{P} \pi(t-1)=\boldsymbol{P} (\boldsymbol{P} \pi(t-2))=\boldsymbol{P} ^2\pi(t-2)=\cdots=\boldsymbol{P} ^t\pi(0) \tag{11}$$
其中 $P^t$ 称为 $t$ 步转移概率矩阵，
$$p_{ij}^t = P(\mathbf{x}_t=j|\mathbf{x}_0=i)\tag{12}$$
表示时刻0从状态 $i$ 触发，时刻 $t$ 到达状态 $j$ 的 $t$ 步转移概率。

尤其是，当t=0时，做如下规定：
$$p_{ij}^0 = \begin{cases}
0, \quad & i \neq j\\
1, \quad & i = j
\end{cases}\tag{13}$$
我们可以认为初始状态下，节点状态保持不变，不会发生状态转移。

### 3.2.2 Champman-Kolmogorov 方程，简称CK方程
CK方程旨在描述 $p_{ij}^t$ 和 $p_{ij}$ 之间的关系，对于一切 $t,n \ge 0, i,j \in \mathcal{S}$，有以下二式成立：

+ $$p_{ij}^{t+n} = \sum_{k\in\mathcal{S}} p_{ik}^t p_{kj}^n \tag{14}$$
+ $$\boldsymbol{P}^t = \boldsymbol{P} \cdot \boldsymbol{P}^{t-1} \tag{15} $$ 

## 3.3 马尔可夫链的性质

![图2 马尔可夫链状态集合](./MCMC/StateClass.png)

> **前言**
> 马尔可夫链的性质是与马尔可夫链的状态划分密切相关的，所有状态如图1所示，通俗解读如下：
>
> + 非常返： 不确定事情不一定会发生，例如买彩票中特等奖
> + 常返：事情一定会发生，例如抛硬币早晚会抛到正面
> + 正常返：事情经过有限步会发生
> + 零常返：事情要经过无穷步才会发生，例如取到标准正态分布的一个点
> + 周期的：在马尔可夫链中，从该状态触发，再返回该状态时时间间隔为 $d \ge 2$（注意，这里不一定是完整的周期）
> + 非周期：周期为1
> + 遍历态： 不可约，非周期，且正常返



### 3.3.1 可约（连通）/不可约

$\color{green}{\textbf{定义 1 }}$如果存在 $n \ge 0$，使得 $p_{ij}^{(n)}>0$，记为 $i \rightarrow j$，称为状态 $i$ 可达状态 $j$（$i,j\in\mathcal{S}$）。如果同时有 $j \rightarrow i$，则称状态 $j$ 与 $i$ 互通，记为 $ j \leftrightarrow i$。

$\color{red}{\textbf{定理 1 }}$ 互通是一种等价关系，即满足以下三个性质：

+ 自返性： $i \leftrightarrow i$

+ 对称性： $j \leftrightarrow i$ 则 $i \leftrightarrow j$

+ 传递性： $k \leftrightarrow j, j \leftrightarrow i$，则 $k\leftrightarrow i$

$\color{green}{\textbf{定义 2 }}$ 我们把任何两个互通状态归为一类，由上述定理可知，同在一类的状态应该都是互通的，并且任何一个状态不能同属于两个不同的类。**如果Markov链只存在一个类，就称它是不可约的（irreducible），否则称为可约的（reducible）。**

### 3.3.2 周期/非周期

$\color{green}{\textbf{定义 3 }}$ 如果集合 $\{n: n\ge 1, p_{ii}^{(n)}>0\}$ 非空，则称它的最大公约数 $d=d(i)$ 为 $i$ 的周期。**如果 $d>i$，则称 $i$ 是周期的（periodic）；如果 $d=1$， 则称 $i$ 是非周期的（aperiodic）**。并且特别规定，当上述集合为空集时，$i$ 的周期无穷大。

对于一个马尔可夫链 $X$ 来说，若所有状态的周期均为1，称该链是非周期的，否则是周期的。

$\color{red}{\textbf{定理 2 }}$ 如果状态 $j,i$同属一类，则 $d(i)=d(j)$。

$\textbf{证明} \quad$ 由类的定义可知 $i \leftrightarrow j$，即存在 $m,n \ge 0$，使得 $p_{ij}^{(m)} > 0, p_{ji}^{(n)}>0$，则 $p_{ii}^{(n+m)} = \sum_{k\in\mathcal{S}} p_{ik}^{(m)} p_{ki}^{(n)} \ge p_{ij}^{(m)} p_{ji}^{(n)} >0$。对于所有 $p_{jj}^{(s)} >0$的 $s$，有 $p_{ii}^{(m+s+n)} \ge p_{ij}^{(m)} p_{jj}^{(s)}  p_{ji}^{(n)} >0$。显然 $d(i)$ 应该同时整除 $m+n$ 和 $m+s+n$，则它必定整除 $s$。而 $d(j)$是 $j$ 的周期，所以 $d(i)$必定整除 $d(j)$。反过来亦可证 $d(j)$ 整除 $d(i)$，于是 $d(i) = d(j)$。

### 3.3.3 常返（Recurrent）/非常返（Transient）

$\color{green}{\textbf{定义 4 }}$ 对于任何状态$i,j$，以 $f_{ij}^{(n)}$ 记录从 $i$ 出发经过 $n$ 步后首次到达 $j$ 的概率。令 $f_{ij} = \sum_{n=1}^{\infty}f_{ij}^{(n)}$，如果 $f_{jj}=1$，则称状态 $j$为常返状态；若 $f_{jj}<1$，则称状态 $j$ 为非常返状态或瞬过状态。

> 常返与否可以理解为从 $j$ 出发未来某个时刻是否一定能回到 $j$。

### 3.3.4 正常返（Positive Recurrent）/零常返（Null Recurrent）

对于常返的状态 $j$，定义
$$\mu_i = \sum_{n=1}^{\infty} n f_{ii}^{(n)} \tag{16}$$
为从状态 $i$ 出发返回状态 $i$ 所需的平均步数。

$\color{green}{\textbf{定义 5 }}$ 对于常返状态 $i$， 如果 $\mu_i < +\infty$，则称 $i$ 为正常返（positive recurrent）状态；若 $\mu_i = + \infty$，则称 $i$ 为零常返状态。

![图2 正常返与零常返举例](./MCMC/PosRecurrent.png)

举例： 根据级数的发散思维，如果 $p=\frac{1}{2}$ 时， $f_{11}^{(2)} = \frac{1}{2}, f_{11}^{(4)} = \frac{1}{2^2}, f_{11}^{(8)} = \frac{1}{2^3}, \cdots$，其余为0，所以
$$f_{11} = 0+\frac{1}{2} + 0+ \frac{1}{2^2} + 0 + 0 +0 + \frac{1}{2^3} + \cdots = 1$$
$$\mu_1 = 0 + \frac{2}{2} + 0 + \frac{4}{2^2} + 0+ 0+ 0+ \frac{8}{2^3} + \cdots = \infty$$
因此是零常返。如果令 $p=\frac{1}{4}$，$f_{11}^{(2)} = \frac{3}{4}, f_{11}^{(4)} = \frac{3}{4^2}, f_{11}^{(8)} = \frac{3}{4^3}, \cdots$，其余为0，所以
$$f_{11} = 0+\frac{3}{4} + 0+ \frac{3}{4^2} + 0 + 0 +0 + \frac{3}{4^3} + \cdots = 1$$
$$\mu_1 = 3\times ( 0 + \frac{2}{4} + 0 + \frac{4}{4^2} + 0+ 0+ 0+ \frac{8}{4^3} + \cdots) = 3$$

，因此是正常返的。**本质原因还是因为返回的概率增大了。**


### 3.3.5 关于上述状态之间的关系


$\color{blue}{\textbf{引理 1 }}$ 对于任意状态 $i,j, 1\le n < +\infty$ 有
$$ f_{ij}^{(n)} = \sum_{k\neq j \land k \in \mathcal{S}} f_{ik}^{(1)}f_{kj}^{(n-1)} \tag{17}$$

$\textbf{证明}\quad$ 当 $n=1$ 时，显然有 $f_{ij}^{(1)} = p_{ij}$，下面考虑 $n>1$的情况。令 $T_{ij}$表示从状态 $i$出发，首次到达状态 $j$的时间。由于 
$$ \{T_{ij}=n, X_0=i\} =\cup_{k\neq j} \{ X_0=i, X_1 =k, X_l \neq j, 2\le l \le n-1, X_n=j \}$$
因此有
$$\begin{split}P(T_{ij}=n|X_0=i) &= \sum_{k\neq j} P(X_1 = k|X_0=i) \times \\
&P(X_n=j|X_1=k, X_0=i, X_l\neq j, 2\le l\le n-1)\end{split}$$
由马尔科夫定义，我们可以得到
$$f_{ij}^{(n)}=\sum_{k\neq j} p_{ik}f_{kj}^{(n-1)}=\sum_{k\neq j}f_{ij}^{(1)}f_{ij}^{(n-1)}  $$


$\color{blue}{\textbf{引理 2 }}$ 对于任意状态 $i,j, 1 \le n \le + \infty$，有
$$ p_{ij}^{(n)} = \sum_{l=1}^n f_{ij}^{(l)} p_{jj}^{(n-l)} \tag{18}$$

$\textbf{证明}\quad$ 根据首次进入法，注意到 $X_n=j$ 时 $T_{ij} \le n$，
$$\begin{split}
P_{ij}^{(n)} &= P(X_0=i, X_n=j) = \sum_{l=1}^n P(X_0=i, X_n=j, T_{ij}=l) \\
&= \sum_{l=1}^n P(X_0=i, X_n=j, X_l=j, X_k\neq j,0<k<l)\\
&= \sum_{l=1}^n P(X_0=i, X_l=j, X_k \neq j, 0<k<l)P(X_n=j|X_l=j)\\
&= \sum_{v=1}^n f_{ij}^{(l)}P_{jj}^{(n-l)}
\end{split}$$


从状态 $i$ 出发经过 $r$ 步以概率 $f_{ij}^{(r)}, r\le n$ 首次到达状态 $j$ ，对于 $1 \le r \le n$ 剩下 $n-r$ 步再次到达状态 $j$ 的概率为 $ p_{ii}^{(n-r)} $。对于这些彼此不相容的概率求和，我们可以得到一个重要的关系式：
$$ p_{ij}^{(n)} = \sum_{r=1}^n f_{ij}^{r} p_{jj}^{(n-r)} \quad n\ge 1 \tag{19}$$
其中 
$$f_{ij}^{(0)} = 0 \quad p_{jj}^{(0)}=1 \quad p_{ij}^{0}=0 \quad  f_{ij}^{(1)}=p_{ij} \quad i\neq j \tag{20}$$
设 $P_{ij}(z)$和 $F_{ij}(z)$分别表示序列 $\{p_{ij}^{(n)}\}$ 和  $\{f_{ij}^{(n)}\}$ 的矩生成函数，那么我们可以得到下式
$$\begin{split} 
P_{ij}(z) &= \sum_{n=0}^\infty p_{ij}^{(n)} z^n = p_{ij}^{(0)} + \sum_{n=1}^\infty \sum_{r=1}^n f_{ij}^{(r)} p_{jj}^{(n-r)} z^n \\
&= p_{ij}^{(0)} + \sum_{r=1}^\infty f_{ij}^{(r)}z^r \sum_{k=0}^\infty p_{jj}^{(k)} z^k = p_{ij}^{(0)} + F_{ij}(z)P_{jj}(z)
\end{split}\tag{21}$$
特别地，对于 $i=j$， 我们得到有用的关系式
$$P_{ii}(z) = 1+ F_{ii}(z)P_{ii}(z) \tag{22} $$
或
$$ P_{ii}(z) = \frac{1}{1-F_{ii}(z)} \tag{23}$$
显然，
$$f_{ij} = \sum_{n=1}^\infty f_{ij}^{(n)} =F_{ij}(1) \tag{24} $$


$\color{blue}{\textbf{引理 3 }}$ 状态 $i$ 是常返的，当且仅当 
$$\sum_{n=0}^\infty p_{ii}^{(n)} =\infty \tag{25}$$

$\textbf{证明}\quad$ 根据定义，如果状态 $i$ 是常返的，那么 
$$f_{ii}=\sum_{n=1}^{\infty} f_{ii}^{(n)} = \lim\limits_{z\rightarrow 1} F_{ii}(z) = 1$$
那么根据公式（23）可得
$$ \sum_{n=1}^\infty p_{ii}^{(n)} = \lim\limits_{z\rightarrow 1} P_{ii}(z) = \lim\limits_{z\rightarrow 1} \frac{1}{1-F_{ii}(z)} = \infty $$


$\color{blue}{\textbf{引理 4 }}$ 状态 $i$ 是非常返的，当且仅当 
$$\sum_{n=0}^\infty p_{ii}^{(n)} <\infty \tag{26}$$
若状态 $j$ 是非常返状态，对于所有的状态 $i$， 
$$\sum_{n=0}^\infty p_{ij}^{(n)} < \infty \tag{27} $$

$\textbf{证明 }\quad$ 假定 
$$\sum_{n=0}^\infty p_{ii}^{(n)}<\infty $$
由于 $p_{ii}^{(n)}$ 非负，当 $z\rightarrow 1$ 时，$P_{ii}(z)$ 单调递增，并且对每一个 $N$
$$\sum_{n=0}^N p_{ii}^{(n)} \le \lim\limits_{z\rightarrow 1} P_{ii}(z) \le \sum_{n=0}^\infty p_{ii}^{(0)} $$
因此，当$ N \rightarrow \infty $, 取极限，我们可以得到
$$\lim\limits_{z\rightarrow 1} P_{ii}^{(n)} = \sum_{n=0}^\infty p_{ii}^{(n)} \lt \infty  $$

对于公式（27），我们利用公式（19）证明：
$$ \sum_{n=0}^\infty p_{ij}^{(n)} = \sum_{n=0}^\infty \sum_{r=0}^n f_{ij}^{(r)} p_{jj}^{(n-r)} = \sum_{m=0}^\infty p_{jj}^{(m)} \sum_{r=0}^\infty f_{ij}^{(r)} \le \sum_{m=0}^\infty p_{jj}^{(m)} \lt \infty $$

$\color{blue}{\textbf{引理 5 }}$ 当且仅当
$$ \sum_{n=0}^\infty p_{jj}^{(n)} = \infty \quad \text{并且当} n\rightarrow \infty \text{时，} p_{jj}^{(n)} \rightarrow 0 \tag{28}$$ 
状态 $j$ 是零常返的，在这种情况下，对所有的 $i$， 
$$\text{当}n\rightarrow \infty \text{时} \quad p_{ij}^{(n)} \rightarrow 0 \tag{29}$$




$\color{blue}{\textbf{引理 6 }}$ 当且仅当 $\mu_j \lt \infty$ 非周期态 $j$ 是遍历的，在这种情况下：
$$\lim\limits_{n\rightarrow \infty} p_{ij}^{(n)} \rightarrow \frac{f_{ij}}{\mu_{ij}} \tag{30}$$

$\textbf{证明}\quad$ 为证明引理5 和引理6， 我们定义
$$v_n = p_{jj}^{(n)} -p_{jj}^{(n-1)} \quad n\ge 1 \quad, v_0 = p_{jj}^{(0)} $$
所以，
$$ \sum_{k=0}^n v_k = p_{jj}^{(n)} $$
定义 
$$ V(z) = \sum_{n=0}^\infty v_n z^n = P_{jj}(z) -zP_{jj}(z) = \frac{1-z}{1-F_{jj}(z)} $$
于是
$$\lim\limits_{z\rightarrow 1} V(z) = \lim\limits_{z\rightarrow 1} \frac{1}{(1-F_{jj}(z))/(1-z)}=\frac{1}{F_{jj}^{'}(1)} = \frac{1}{\mu_j} $$
且
$$\lim\limits_{z\rightarrow 1} V(z) = \lim\limits_{z\rightarrow 1}\sum_{k=0}^\infty v_k z^k = \lim\limits_{n\rightarrow \infty} \sum_{k=0}^{n} v_k = \lim\limits_{n\rightarrow \infty} p_{jj}^{(n)} $$
因此，我们可以得到
$$\lim\limits_{n\rightarrow\infty} p_{jj}^{(n)} = \frac{1}{\mu_j}  $$
引理5中公式（28）得证。

> **步骤解释1：**
> $$\begin{split} 
> P_{jj}(z) - zP_{jj}(z) &= \{p_{jj}^{(0)}z^0 + p_{jj}^{(1)}z^1 + p_{jj}^{(2)}z^2 + \cdots \} - \{ p_{jj}^{(0)}z^1 + p_{jj}^{(1)}z^2 + p_{jj}^{(2)}z^3 + \cdots \} \\
&= p_{jj}^{(0)}z^0 + (p_{jj}^{(1)}-p_{jj}^{(0)})z^1 + (p_{jj}^{(2)}-p_{jj}^{(1)})z^2 + \cdots \\
&= \sum_{n=0}^\infty v_n z^n
>\end{split}$$

> **步骤解释2:**
> 根据导数定义
> $$F^{'}(1)=\lim\limits_{z\rightarrow 1^-} \frac{F_{jj}(1)-F_{jj}(z)}{1-z^-}$$


接下来证明引理5中公式（29）。根据公式（19），我们可以得到
$$\lim\limits_{n\rightarrow\infty} p_{ij}^{(n)} = \lim\limits_{n\rightarrow\infty}\sum_{k=1}^n f_{ij}^{(k)}p_{jj}^{(n-k)} = \sum_{k=1}^\infty \frac{f_{ij}^{(k)}}{\mu_j} = \frac{f_{ij}}{\mu_j}  $$
如果 $j$ 是零常返状态，那么 $\mu_j = \infty$，所以 
$$\lim\limits_{n\rightarrow\infty} p_{ij}^{(n)} \rightarrow 0 $$
因此，引理5中公式（29）得证。

接下来证明引理6。如果状态 $j$ 是非周期的常返遍历态，那么根据定义 $\mu_j \lt \infty$，且根据引理5， 当 $n\rightarrow \infty$ 时， $p_{JJ}^{(n)}$ 趋于一个非零常数，因此 $\lim\limits_{n\rightarrow\infty} \sum_{k=0}^n p_{jj}^{(n)} = \infty$，从而状态 $j$ 是常返和遍历的，因而证明了引理6.


$\color{blue}{\textbf{引理 7 }}$ 设状态 $j$ 是周期为 $T$ 的常返状态，那么
$$p_{jj}^{(nd)} \rightarrow \frac{d}{\mu_j} \tag{31}$$

$\textbf{证明}\quad$ 状态 $j$ 是周期为 $d$ 的，除非 $n$ 是周期 $d$ 的整数倍，否则 $f_{jj}^{(n)}=0$，因此$F_{ii}(z)$ 仅包含 $z^d$ 的幂。设
$$F_{jj}(z) = \varphi(z^d) $$
所以，根据公式（23），我们可以得到
$$P_{jj}(z) = \frac{1}{1-\varphi(z^d)} $$
或
$$ P_{jj}(z^{1/d}) = \frac{1}{1-\varphi(z)} = \sum_{n=0}^\infty p_{jj}^{(nd)}z^n $$
且用类似于引理5证明方式中的推理，我们可以定义
$$v_n = p_{jj}^{(nd)} - p_{jj}^{((n-1)d)} $$
且可以得到 $$ \sum_{k=0}^{n} v_k = p_{jj}^{(nd)} $$，令
$$ \begin{split}
V(z) &= \sum_{n=0}^\infty v_n z^n =P_{jj}(z) - zP_{jj}(z) \\
&=\frac{1}{1-\varphi({z^d})} - \frac{z}{1-\varphi(z^d)} \\
&= \frac{1-z}{1-\varphi(z^d)}
\end{split}$$
于是
$$\begin{split} 
\lim\limits_{z\rightarrow 1} V(z) &= \lim\limits_{z\rightarrow 1} \frac{1-z}{1-\varphi(z^d)} \cdot \frac{1-z^d}{1-z^d} = \lim\limits_{z\rightarrow 1} \frac{1-z^d}{1-\varphi(z^d)} \cdot \frac{1-z}{1-z^d} \\
&= \lim\limits_{z\rightarrow 1} \frac{1}{(1-\varphi(z^d))/(1-z^d)} \cdot \frac{1}{(1-z^d)/(1-z)} \\
&= \lim\limits_{z\rightarrow 1} \frac{1}{\varphi^{'}(z^d)\cdot (\partial z^d)/(\partial z)} = \lim\limits_{z\rightarrow 1} \frac{1}{\varphi^{'}(z)}
\end{split}$$
对 $F_{jj}(z) = \varphi(z^d)$ 做变换，得到  $F_{jj}(z^{\frac{1}{d}}) = \varphi(z)$  两边求导，得到
$$ F_{jj}^{'} (z^{\frac{1}{d}}) \cdot \frac{1}{d} z^{\frac{1}{d} - 1} = \varphi^{'} (z) $$
带回公式，我们可以得到
$$\begin{split}
\lim\limits_{z\rightarrow 1} V(z) &= \lim\limits_{z\rightarrow 1} \sum_{n=0}^\infty v_n z^n = \lim\limits_{z\rightarrow 1} \sum_{k=0}^n v_k = \lim\limits_{z\rightarrow 1} p_{jj}^{(nd)} \\
&= \lim\limits_{z\rightarrow 1} \frac{1}{\varphi^{'}(z)} = \lim\limits_{z\rightarrow 1} \frac{d}{F_{jj}^{'}(z)}
\end{split}$$
因此，引理7的证
$$p_{ij}^{nT} \rightarrow \frac{1}{\varphi^{'}(1)} = \frac{T}{F^{'}_{jj}(1)} = \frac{T}{\mu_j} $$


## 3.4 平稳分布

$\color{green}{\textbf{定义 6 }}$ 对于Markov链，概率分布 $\{p_j, j \in \mathcal{S}\}$ 称为平稳分布，如果
$$p_j = \sum_{i\in\mathcal{S}} p_i p_{ij} \tag{32}$$
可以理解为，对于 $(t-1)$ 时刻的概率分布 $\pi(t-1)$，通过概率转移矩阵 $\pi(t)=\mathbf{P}\pi(t-1)$ 计算得到新的概率分布，如果 $\pi(t)=\pi(t-1)$，则称为是平稳分布。

$\color{green}{\textbf{定义 7 }}$ 对于不可约非周期的马尔可夫链：

(1) 若它是遍历的，则存在以下唯一的平稳分布 
$$\pi_j = \lim\limits_{n\rightarrow\infty} p_{jj}^{(n)} >0 (j\in\mathcal{S}) $$ 

(2) 若状态都是瞬过的或者全为零常返的，则平稳分布不存在。

$\textbf{证明}\quad$ 对于第（1）条定理，对于遍历的Markov链，根据引理6可知 $$\lim\limits_{n\rightarrow\infty} p_{ij}^{(n)} \ge 0 $$ 存在，记为 $\pi_j$。首先证明 $\{\pi_j, j\in\mathcal{S}\}$ 是平稳分布。

由于 $$\sum_{j\in\mathcal{S}} p_{ij}^{(n)} =1 $$，则有 $$\lim\limits_{n\rightarrow\infty} \sum_{j\in\mathcal{S}} p_{ij}^{(n)} =1 $$，因此可以得到 $\sum_{j\in\mathcal{S}}\pi_j=1$。

利用C-K方程，可以得到 $$p_{ij}^{(n+1)} = \sum_{k\in\mathcal{S}} p_{ik}^{(n)} p_{kj} $$。两边取极限可以得到
$$\lim\limits_{n\rightarrow\infty}p_{ij}^{(n+1)} = \lim\limits_{n\rightarrow\infty}\sum_{k\in\mathcal{S}} p_{ik}^{(n)}p_{kj} = \sum_{k\in\mathcal{S}}\left[\lim\limits_{n\rightarrow\infty} p_{ik}^{(n)} \right] p_{kj} $$
因此可以得到 $\pi_j = \sum_{k\in\mathcal{S}} \pi_k p_{kj}$，因此 $\{\pi_j, j\in\mathcal{S}\}$ 是平稳分布。

再证明 $\{\pi_j, j\in\mathcal{S}\}$  是唯一的平稳分布。假设还有另外一个平稳分布 $\{\tilde{\pi}_j, j\in\mathcal{S}\}$，则根据  $\tilde{pi}_j = \sum_{k\in\mathcal{S}} \tilde{\pi}_k p_{kj}$ 归纳可以得到
$$\tilde{pi}_j = \sum_{k\in\mathcal{S}} \tilde{\pi}_k p_{kj}^{(n)}, \quad n=1,2,\cdots$$
令 $n\rightarrow\infty$，对上式两端取极限，可以得到
$$\tilde{\pi}_j = \sum_{i\in\mathcal{S}} \tilde{\pi}_i \lim\limits_{\rightarrow\infty} p_{ij}^{(n)} = \sum_{i\in\mathcal{S}} \tilde{\pi}_i\pi_j $$
因为 $\sum_{i\in\mathcal{S}} \tilde{\pi}_{i} = 1$，所以 $\tilde{\pi}_j = \pi_j$，得证平稳分布唯一。

对于第（2）条定理，假设存在一个平稳分布 $\{\pi_j, j\in\mathcal{S} \}$，则根据（1）中的证明可知，
$$ \pi_j \sum_{i\in\mathcal{S}} \pi_i p_{ij}^{(n)}, \quad n=1,2,\cdots $$
成立，令 $n\rightarrow \infty$ 知道 $p_{ij}^{(n)}\rightarrow 0$，则可以推导出 $\pi_j = 0 (j\in \mathcal{S})$，这是不可能的。因此对于非常返或零常返的Markov链不存在平稳分布。


# 4 Markov Chain and Monte Carlo

## 4.1 核心思想:基于采样的随机近似方法

在随机变量 $\mathbf{x}$ 的状态空间 $\mathcal{S}$ 上定义一个满足遍历定理的马尔可夫链 $\mathbf{X}=\{\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_t, \cdots\}$，使其平稳分布就是抽样的目标分布 $p(\mathbf{x})$。这样当时间趋于无穷时，样本分布趋近于平稳分布，样本的函数值的均值趋近于函数的数学期望。所以，当时间足够长时（时刻大于某个正整数 $m$）zhihou de shijian (时刻小于某个正整数 $n$, $n>m$)里随机游走得到的样本集合 $\{\mathbf{x}_{m+1}, \mathbf{x}_{m+2}, \cdots, \mathbf{x}_{n}\}$ 就是目标概率分布的抽样结果，得到的函数均值就是要计算的数学期望：
$$\hat{\mathbf{E}}[f(\mathbf{x})] = \frac{1}{n-m} \sum_{i=m+1}^n f(\mathbf{x}_i)$$
到时刻 $m$ 为止的时间段被称为燃烧期。

我们希望构造的马尔可夫链是细致平稳的，即
$$p(\mathbf{z}) q(\mathbf{z} \rightarrow \mathbf{z}^*) \alpha (\mathbf{z}^*) = p(\mathbf{z}^*) q(\mathbf{z}^* \rightarrow \mathbf{z}) \alpha (\mathbf{z})$$
这样，我们可以将 $p$ 看作目标分布，$q$ 看作建议分布， $\alpha$ 看作接受分布。

假设要抽样的分布是 $p(\mathbf{x})$，我们构建以 $p(\mathbf{x}, \mathbf{x}^{'})$ 为转移核的马尔可夫链。
$$p(\mathbf{x}, \mathbf{x}^{'}) = q(\mathbf{x}, \mathbf{x}^{'}) \alpha(\mathbf{x}, \mathbf{x}^{'})$$
其中 $q(\mathbf{x}, \mathbf{x}^{'})$ 是不可约的，即其概率值恒不为0，同时是一个易于抽样的分布。接受分布 $\alpha(\mathbf{x}, \mathbf{x}^{'})$ 可以表示如下：
$$\alpha(\mathbf{x}, \mathbf{x}^{'}) = \min \left\{1, \frac{p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})}{p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'})}\right\}$$
这样，转移核可以写为：
$$p(\mathbf{x}, \mathbf{x}^{'}) = \begin{cases}
q(\mathbf{x}, \mathbf{x}^{'}), \quad &p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x}) \ge p(x)q(\mathbf{x}, \mathbf{x}^{'}) \\
q(\mathbf{x}^{'}, \mathbf{x}) \frac{p(\mathbf{'})}{p(\mathbf{x})}, \quad & p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x}) < p(x)q(\mathbf{x}, \mathbf{x}^{'}) 
\end{cases}$$
个人理解，之所以要加入 $\min$ 限制，是因为转移核作为概率值要始终不超过1。

$\color{red}{\textbf{定理 3 }}$ 通过上述方式构造的马尔可夫链是可逆的，即
$$p(\mathbf{x})p(\mathbf{x}, \mathbf{x}^{'}) = p(\mathbf{x}^{'})p(\mathbf{x}^{'}, \mathbf{x})$$
且 $p(\mathbf{x})$ 是该马尔可夫链的平稳分布。

$\textbf{证明} \quad$ 若 $\mathbf{x} = \mathbf{x}^{'}$，显然成立

如果 $\mathbf{x}\neq\mathbf{x}^{'}$， 则
$$\begin{split}
p(\mathbf{x})p(\mathbf{x}, \mathbf{x}^{'}) &= p(\mathbf{x}) q(\mathbf{x}, \mathbf{x}^{'}) \min \left\{1, \frac{p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})}{p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'})}\right\} \\
&= \min \left\{ p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'}), p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x}) \right\}\\
&= p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x}) \min \left\{ \frac{p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'})}{p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})}, 1 \right\}\\
&= p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})
\end{split}$$
公式中导数第二行到最后一行的原因是因为按照接受概率的定义，$\alpha(\mathbf{x}, \mathbf{x}^{'}) \le 1$。 而上式中导数第二行中的左侧项则是反过来的，因此其值大于等于1且最小值为1。
显然定理中马尔可夫链可逆得证。
$$\begin{split}
\int p(\mathbf{x}) p(\mathbf{x}, \mathbf{x}^{'}) d \mathbf{x} &= \int p(\mathbf{x}^{'}) p(\mathbf{x}^{'}, \mathbf{x}) d \mathbf{x} \\
&= p(\mathbf{x}^{'}) \int p(\mathbf{x}^{'}, \mathbf{x}) d\mathbf{x} \\
&= p(\mathbf{x}^{'})
\end{split}$$
因此定理得证。

这就是常用的 Metroplis-Hasting算法。

## 4.1 Metropolis-Hasting 算法

输入： 抽样的目标分布的密度函数 $p(\mathbf{x})$， 函数 $f(\mathbf{x})$

输出： $p(\mathbf{x})$ 的随机样本 $\{\mathbf{x}_{m+1}, \mathbf{x}_{m+2}, \cdots, \mathbf{x}_{n}\}$， 函数样本均值 $f_{mn}$

参数：收敛步数 $m$, 迭代步数 $n$

（1）对于任意一个初始值 $\mathbf{x}_0$

（2）对 $i=1,2,\cdots, n$ 循环执行

$\quad$ （a）设状态 $\mathbf{x}_{i-1} = \mathbf{x}$，按照建议分布随机抽取一个候选状态 $\mathbf{x}^{'}$

$\quad$ （b） 计算接受概率 $\alpha(\mathbf{x}, \mathbf{x}^{'}) = \min \left\{1, \frac{p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})}{p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'})}\right\} $

$\quad$ （c）从区间（0，1）中按照均匀分布随机抽取一个数 $u$，如果 $u \le \alpha(\mathbf{x}, \mathbf{x}^{'})$，则状态 $\mathbf{x}_i = \mathbf{x}^{'}$，否则 $\mathbf{x}_i = \mathbf{x}$

（3）得到样本集合 $\{\mathbf{x}_{m+1}, \mathbf{x}_{m+2}, \cdots, \mathbf{x}_{n}\}$

计算 $$f_{mn} = \frac{1}{n-m} \sum_{i=m+1}^n f(\mathbf{x}_i)$$

这里要注意的是，我们这里使用的 $p(\mathbf{x})$ 实际上是 $\hat{p}(\mathbf{x})$，是一个根据先验知识及似然概率推断出来的近似值，按理说还需要除以一个归一化因子，但实际上归一化因子很难求解，因此常常默认 $p(\mathbf{x}) = \hat{p}(\mathbf{x})$

## 4.2 Gibbs 采样
Gibbs采样是MH采样的一个特例，通常假设要采样的目标分布 $p(\mathbf{x}^{(i)}) = p(z_1^{(i)},z_2^{(i)} , \cdots, z_K^{(i)})$ 是高维的，需要一维一维的采，**每次采样第$i$维时，固定其他维**。

假设多元变量联合概率分布为 $p(\mathbf{x}) = p(x_1, x_2, \cdots, x_k)$， 吉布斯采样从一个初始样本 $x^{(0)} = (x^{(0)}_1, x^{(0)}_2, \cdots, x^{(0)}_k)^T$ 出发不断迭代，每一次迭代得到联合分布的一个样本 $\mathbf{x}^{(i)} = (x^{(i)}_1, x^{(i)}_2, \cdots, x^{(i)}_k)^T$，最终样本序列为 $\{\mathbf{x}^{(0)}, \mathbf{x}^{(1)}, \cdots, \mathbf{x}^{(k)} \}$。

在每次迭代中，依次对 $k$ 个随机变量中的一个变量进行随机抽样。如果在第 $i$ 次迭代中， 对滴 $j$ 个变量进行随机抽样，那么抽样的分布是满条件概率分布 $p(x_j | x^{(i)}_{-j})$，这里 $x^{(i)}_{-j}$ 表示第 $i$ 次迭代中，变量 $j$ 以外的其他变量。

吉布斯采样算法如下：

输入： 目标概率分布的密度函数 $p(\mathbf{x})$，函数 $f(\mathbf{x})$

输出： $p(\mathbf{x})$ 的随机样本 $\{\mathbf{x}_{m+1}, \mathbf{x}_{m+2}, \cdots, \mathbf{x}_{n}\}$，函数样本均值 $f_{mn}$

参数： 收敛步数 $m$， 迭代步数 $n$

（1） 初始化初始样本 $x^{(0)} = (x^{(0)}_1, x^{(0)}_2, \cdots, x^{(0)}_k)^T$

（2） 对 $i$ 循环执行。设第 $(i-1)$ 迭代结束时的样本为 $\mathbf{x}^{(i-1)} = (x^{(i-1)}_1, x^{(i-1)}_2, \cdots, x^{(i-1)}_k)^T$，则第 $i$ 次的迭代进行如下几步操作：

$\quad$ 1) 由满条件分布 $p(x_1|x^{(i-1)}_2, \cdots, x^{(i-1)}_k)$ 抽取 $x^{(i)}_1$

$\quad \quad \vdots$

$\quad$ j) 由满条件分布 $p(x_j|x^{(i)}_1, \cdots, x^{(i)}_{j-1}, x^{(i-1)}_{j+1}, \cdots, x^{(i-1)}_k)$ 抽取 $x^{(i)}_j$

$\quad \quad \vdots$

$\quad$ k) 由满条件分布 $p(x_k|x^{(i)}_1, \cdots, x^{(i)}_{k-1})$ 抽取 $x^{(i)}_k$

得到第 $i$ 次的迭代值 $\mathbf{x}^{(i)} = (x^{(i)}_1, x^{(i)}_2, \cdots, x^{(i)}_k)^T$

(3) 得到样本集合  $\{\mathbf{x}_{m+1}, \mathbf{x}_{m+2}, \cdots, \mathbf{x}_{n}\}$

(4) 计算 $$f_{mn} = \frac{1}{n-m} \sum_{i=m+1}^{n} f(\mathbf{x}^{(i)})$$

定义的减一分部是当前变量的满条件概率分布：
$$ q(\mathbf{x}, \mathbf{x}^{'}) = p(x^{'}_{j}|x_{-j}) $$
这时，接受概率为 $\alpha = 1$
$$\begin{split}
\alpha(\mathbf{x}, \mathbf{x}^{'}) &= \min \left\{ 1, \frac{p(\mathbf{x}^{'})q(\mathbf{x}^{'}, \mathbf{x})}{p(\mathbf{x})q(\mathbf{x}, \mathbf{x}^{'})} \right\} \\
&= \min \left\{ 1, \frac{p(x^{'}_{-j}) p(x^{'}_j|x^{'}_{-j}) p(x_j|x^{'}_{-j}) }{p(x_{-j}) p(x_j|x_{-j}) p(x^{'}_j | x_{-j}) } \right\} = 1
\end{split}$$
上式是用到了 $p(x_{-j}) = p(x^{'}_{-j})$ 和 $p(\cdot | x_{-j}) = p(\cdot|x^{'}_{-j})$，注意，这里的点表示的是当二者任意一致时成立。
转移核就是满条件概率分布
$$p(\mathbf{x}, \mathbf{x}^{'}) = p(x^{'}_j|x_{-j})$$



# 参考

1. [李航，统计机器学习（第2版），清华大学出版社](http://www.tup.tsinghua.edu.cn/booksCenter/book_08132901.html)
2. [张波，应用随机过程（第5版），中国人民大学出版社](http://www.crup.com.cn/Book/TextDetail?doi=1520e7e7-ac46-40b9-9864-1a917d9677dc)
3. [帕普里斯（著），保铮（译），概率、随机变量与随机过程（第4版），西安交通大学出版社](https://baike.baidu.com/item/%E6%A6%82%E7%8E%87%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%E4%B8%8E%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B/56914646?fr=aladdin)
3. [白板推导](https://space.bilibili.com/97068901)
4. [如何理解马尔科夫链中的常返态，非常返态，零常返，正常反，周期和非周期，有什么直观意义？ - uplow的回答 - 知乎](https://www.zhihu.com/question/46539491/answer/263442039)
5. [如何理解马尔科夫链中的常返态，非常返态，零常返，正常反，周期和非周期，有什么直观意义？ - LittleHealth的回答 - 知乎](https://www.zhihu.com/question/46539491/answer/2177451474)