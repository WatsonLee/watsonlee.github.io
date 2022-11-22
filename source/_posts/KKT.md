---
title: 约束优化
date: 2022-11-18 12:09:33

tags: 
    - 约束优化
    - 最优化

categories: 
    - 基础知识

mathjax: true
---

Karush-Kuhn-Tucker（KKT）条件是非线性规划最佳解的必要条件。KKT条件将Lagrange乘数法所涉及到的等式约束优化问题推广至不等式。在实际应用上，KKT条件一般不存在代数解。许多优化算法可提供数值计算选用。

# 1. 原始问题

## 1.1 等式约束优化问题
<a id="sec1.1"></a>

给定一个目标函数 $f:\mathbb{R}^n \rightarrow \mathbb{R} $，我们希望找到 $\mathbf{x}\in \mathbb{R}^n$，在满足约束条件 $g(\mathbf{x})=0$ 的前提下，使得 $f(\mathbf{x})$ 有最小值，这个约束优化问题记为：

$$\begin{split}
& \min \quad f(\mathbf{x}) \\
& \text{s.t.} \quad g(\mathbf{x})=0
\end{split}\tag{1}$$

为了方便分析，假设 $f$ 与 $g$ 均为连续可导函数。Lagrange乘数法是等式约束优化问题的典型解法。定义Lagrange函数：

$$L(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x}) \tag{2}$$

其中 $\lambda$ 为Lagrange乘数。Lagrange乘数法将原本的约束优化问题转化为等价的无约束优化问题：

$$\min\limits_{\mathbf{x},\lambda} L(\mathbf{x}, \lambda) \tag{3}$$

计算 $L$ 对 $\mathbf{x}$ 和 $\lambda$ 的偏导数并设为零，可以得到最优解的必要条件：

$$\begin{split}
& \nabla_\mathbf{x} L = \frac{\partial L}{\partial \mathbf{x}} = \nabla f + \lambda \nabla g = 0\\
& \nabla_\lambda L = \frac{\partial L}{\partial \lambda} = g(\mathbf{x}) = 0
\end{split}\tag{4}$$

其中，公式（4）第一式为定常方程式（Stationary Equation），第二式为约束条件，求解上面 n+1个方程式可以得到 $L(\mathbf{x}, \lambda)$ 的Stationary point $\mathbf{x}^{*}$ 以及 $\lambda$ 的值（正负均有可能）。

## 1.2 不等式约束优化问题
<a id="sec1.2"></a>

将约束等式 $g(\mathbf{x})=0$ 推广为不等式 $g(\mathbf{x}) \le 0$，考虑如下问题：
$$\begin{split}
& \min \quad f(\mathcal{x})\\
& \text{s.t.} \quad g(\mathcal{x}) \le 0
\end{split}\tag{5}$$
约束不等式 $g(\mathbf{x}) \le 0$称为原始可行性（primal feasibility），据此我们定义可行域（feasible region）$K = \{\mathbf{x}\in\mathbb{R}^n | g(\mathbf{x}) \le 0\}$，假设Stationary Point $\mathbf{x}^*$ 为满足约束条件的最佳解，分为以下两种情况讨论：

+ $g(\mathbf{x}^*) < 0$，表示最佳解位于可行域 $K$ 内部，称为内部解（interior solution），这时约束条件是不起作用的（inactive）
+ $g(\mathbf{x}^*) = 0$，表示最佳解落在可行域 $K$ 边界，称为边界解（boundary solution），此时约束条件发挥作用（active）

这两种情况的最佳解具有不同的必要条件：

+ 内部解：在约束条件不发挥作用（inactive）的情况下， $g(\mathbf{x})$ 不起作用，约束问题退化为无约束优化问题，因此驻点 $\mathbf{x}^*$ 满足 $\nabla_\mathbf{x} f = 0$ 且 $\lambda=0$。

+ 边界解：在约束条件发挥作用（active）的情形下，约束不等式变成等式 $g(\mathbf{x})=0$，这与Lagrange乘数法的情况相同。这里可以认为存在 $\lambda$ 使得 $\nabla_\mathbf{x} f = -\lambda \nabla_\mathbf{x}g$。这里 $\lambda$ 的正负号是有其意义的。因此我们希望最小化 $f$，梯度 $\nabla_\mathbf{x} f$ （函数 $f$ 在 $\mathbf{x}^*$ 点方向导数最大值，即最陡上升方向）应该指向可行域 $K$ 的内部（因为最优解最小值是在边界取得的），但 $\nabla_\mathbf{x} g$ 指向可行域 $K$ 外部（即 $g(\mathbf{x})>0$ 的区域，因为约束是小于等于0，继续向外走才能持续使目标函数 $f$ 的值下降），因此 $\lambda \ge 0$， 称为对偶可行性（dual feasibility）。

## 1.3 多个约束等式与约束不等式

根据[章节1.1](#sec1.1)和[章节1.2](#sec1.2)，我们可以推广至多个约束等式与约束不等式的情况，考虑标准约束优化（或者称非线性规划）：

$$\begin{split}
\min\limits_{\mathbf{x}\in \mathbb{R}^n} \quad &f(\mathbf{x})\\
\text{s.t.} \quad & c_i(\mathbf{x}) \le 0, \quad i=1,2,\ldots,k \\
& h_j(\mathbf{x}) = 0, \quad j = 1,2,\ldots, l
\end{split}\tag{6}$$
我们称上式为约束最优化问题为原始最优化问题或原始问题。

首先，引入Generalized Lagrange Function（广义拉格朗日函数）：
$$L(\mathbf{x}, \alpha, \beta) = f(\mathbf{x}) + \sum_{i=1}^k \alpha_i c_i(\mathbf{x}) + \sum_{j=1}^l \beta_j h_j(\mathbf{x}) \tag{7}$$
这里 $\alpha_i, \beta_j$ 是Lagrange乘子，$\alpha_i>\ge 0$，考虑 $\mathbf{x}$ 的函数：
$$\theta_P(\mathbf{x}) = \max\limits_{\alpha, \beta; \alpha_i \ge 0} L(\mathbf{x}, \alpha, \beta) \tag{8}$$ 
这里，下标 $P$ 表示原始问题。

假设给定某个 $\mathbf{x}$，如果它违反原始问题的约束条件，即存在某个 $i$ 使得 $c_i(\mathbf{x})>0$ 或者存在某个 $j$ 使得 $h_j(\mathbf{x}) \neq 0$，那么就有

$$\theta_P(\mathbf{x}) = \max\limits_{\alpha, \beta; \alpha_i\ge 0} \left[f(\mathbf{x}) + \sum_{i=1}^k \alpha_i c_i (\mathbf{x}) + \sum_{j=1}^l \beta_j h_j (\mathbf{x}) \right] = + \infty \tag{9}$$
因为如果某个 $i$ 使得约束 $c_i(\mathbf{x})>0$，则可令 $\alpha_i \rightarrow + \infty$；若某个 $j$ 使得 $h_j(\mathbf{x}) \neq 0$，则可令 $\beta_j$ 使 $\beta_j h_j (\mathbf{x}) \rightarrow + \infty$， 而将其余各个 $\alpha_i, \beta_j$ 均取值为0. 

相反地，如果 $\mathbf{x}$ 满足公式（6）中的约束条件式，则根据公式（7）和公式（8）可以得到：
$$\theta_P(\mathbf{x}) = \begin{cases}
f(\mathbf{x}), \quad & \mathbf{x} \text{满足原始问题约束}\\
+ \infty, \quad &\text{其他}
\end{cases}\tag{10}$$
所以，如果考虑极小化问题
$$\min\limits_{\mathbf{x}}\theta_P(\mathbf{x}) = \min\limits_{\mathbf{x}} \max\limits_{\alpha, \beta;\alpha_i \ge 0} L(\mathbf{x}, \alpha, \beta) \tag{11}$$
公式（11）是与公式（6）原始最优化问题是等价的，即它们有相同的解。问题 $\min\limits_{\mathbf{x}} \max\limits_{\alpha, \beta;\alpha_i \ge 0} L(\mathbf{x}, \alpha, \beta)$ 被称为广义拉格朗日函数的极小极大问题。这样一来，就把原始最优化问题表示为广义拉格朗日函数的极小极大问题。为了方便，可以定义原始问题的最优值
$$p^* = \min\limits_{\mathbf{x}} \theta_P(\mathbf{x}) \tag{12}$$
称为原始问题的值。

# 2. 对偶问题

定义
$$\theta_D(\alpha, \beta) = \min\limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta) \tag{13}$$
再考虑极大化公式（13），即
$$\max\limits_{\alpha, \beta;\alpha_i \ge 0} \theta_D(\alpha, \beta) = \max\limits_{\alpha,\beta;\alpha_i \ge 0} \min \limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta) \tag{14}$$
问题 $\max\limits_{\alpha,\beta;\alpha_i \ge 0} \min \limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta)$ 被称为广义拉格朗日函数的极大极小问题。

可以将广义拉格朗日函数的极大极小问题表示为约束最优化问题：
$$\begin{split}
&\max\limits_{\alpha, \beta} \theta_D(\alpha, \beta) = \max\limits_{\alpha, \beta}\min\limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta) \\
&\text{s.t.} \quad \alpha_i \ge 0, \quad i=1,2,\ldots,k
\end{split}\tag{15}$$
上式被称为原始问题的对偶问题，定义对偶问题的最优值
$$d^* = \max\limits_{\alpha, \beta; \alpha_i \ge 0} \theta_D (\alpha, \beta) \tag{16}$$
为对偶问题的值。

> **为什么要引入对偶问题？**
> 
> + 对偶问题交换了求极值的顺序，先求解的是函数 $f(\mathbf{x})$ 的极小值（自变量为 $\mathbf{x}$ ），等价成梯度为0的约束条件，并将难以求解的约束条件扔到目标函数的位置上去。
>
> + 在定义域为凸集的前提下，转换后的对偶问题的自变量是约束条件系数构成的线性函数，一定是凸问题。

# 3. 原问题与对偶问题之间的关系

> **定理1:** 如果原始问题和对偶问题都有最优值，则
> $$d^* = \max\limits_{\alpha, \beta;\alpha_i \ge 0} \min\limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta) \le \min\limits_{\mathbf{x}}\max\limits_{\alpha, \beta;\alpha_i \ge 0} L(\mathbf{x}, \alpha, \beta) = p^* \tag{17} $$

**证明：** 根据公式（9）和公式（13）的定义，我们可以得到：
$$\theta_D(\alpha, \beta) =\min\limits_{\mathbf{x}} L(\mathbf{x}, \alpha, \beta) \le L(\mathbf{x}, \alpha, \beta) \le \max\limits_{\alpha, \beta; \alpha_i \ge 0} L(\mathbf{x}, \alpha, \beta) = \theta_P(\mathbf{x}) \tag{18}$$
即
$$\theta_D(\alpha, \beta) \le \theta_P(\mathbf{x}) \tag{19}$$
由于原始问题和对偶问题都有最优值，所以：
$$\max\limits_{\alpha,\beta;\alpha_i \ge 0}\theta_D(\mathbf{x}) \le \min\limits_{\mathbf{x}} \theta_P(\mathbf{x}) \tag{20} $$
因此，定理得证。

> *推论1:* 设 $x^*, \alpha^*, \beta^*$ 分别是原始问题（公式（6））和对偶问题（公式（15））的可行解，并且 $d^* = p^*$， 则 $x^*, \alpha^*, \beta^*$ 分别是原始问题和对偶问题的最优解。

> **定理2:** 考虑原始问题（公式（6））和对偶问题（公式（15））。假设函数 $f(\mathbf{x})$ 和 $c_i(\mathbf{x})$ 是凸函数，$h_j(\mathbf{x})$ 是仿射函数，并且假设不等式约束 $c_i(\mathbf{x})$ 是严格执行的，即存在 $\mathbf{x}$ ，对所有 $i$ 有 $c_i(\mathbf{x}) <0$， 则存在 $x^*, \alpha^*, \beta^*$，使 $\mathbf{x}^*$ 是原始问题的解， $\alpha^*$ 和 $\beta^*$ 是对偶问题的解，并且
> $$p^* = d^* = L(\mathbf{x}^*, \alpha^*, \beta^*) \tag{21}$$

> **定理3:** 对原始问题（公式（6））和对偶问题（公式（15）），假设函数 $f(\mathbf{x})$ 和 $c_i(\mathbf{x})$ 是凸函数，$h_j(\mathbf{x})$ 是仿射函数，并且假设不等式约束 $c_i(\mathbf{x})$ 是严格执行的， 则存在 $x^*, \alpha^*, \beta^*$ 分别是原始问题和对偶问题的解的充分必要条件是下面的 Karush-Kuhn-Tucker（KKT）条件

$$ \nabla_\mathbf{x} L(\mathbf{x}^*, \alpha^*, \beta^*)=0 \tag{22-1} $$
$$ c_i(\mathbf{x}) \le 0, \quad i =1,2,\ldots,k \tag{22-2}$$
$$ h_j(\mathbf{x}^*) = 0, \quad j=1,2,\ldots, l \tag{22-3}$$
$$\alpha_i^* \ge 0, \quad i=1,2,\ldots,k \tag{22-4}$$
$$ \alpha_i^* c_i(\mathbf{x}^*)=0, \quad i=1,2,\ldots,k \tag{22-5} $$
特别指出，公式（22-5）被称为KKT的对偶互补条件，由此条件可知，如果 $\alpha_i^* >0$， 则 $c_i(\mathbf{x}^*) = 0$

# 4. KKT条件的解释

## 4.1 必要性证明

公式（22）为KKT条件，下面对这5个条件逐个进行解释：
+ 公式（22-1）为广义拉格朗日函数的梯度，表示最优解处的梯度为0.
+ 公式（22-2）和公式（22-3）分别是愿问题的不等式约束和等式约束，最优解显然应当满足
+ 公式（22-4）是对偶问题的不等式约束，表示对偶可行。即当 $\alpha \ge 0$ 时，$L(\mathbf{x}, \alpha, \beta) \le f(\mathbf{x})$，对偶函数才能给出愿问题的最优值下界。
+ 公式（22-5）被称为互补松弛性，推导过程如公式（23）所示：
    - 第一行：强对偶条件成立，对偶间隙为0
    - 第二行：根据公式（13）展开对偶函数
    - 第三行：函数的最小值不会超过定义域内任意一点函数值
    - 第四行：等式约束 $h_j(\mathbf{x})$ 为0， 而不等式约束 $c_i(\mathbf{x})\le 0$ 且拉格朗日乘子 $\alpha \ge 0$， 因此成立
$$\begin{split}
f(\mathbf{x}^*) &= \theta_D(\alpha^*, \beta^*) \\
&= \min\limits_{\mathbf{x}} \left( f(\mathbf{x}) + \sum_{i=1}^k \alpha_i^* c_i(\mathbf{x}) + \sum_{j=1}^l \beta_j^* h_j(\mathbf{x}) \right) \\
& \le f(\mathbf{x^*}) + \sum_{i=1}^k \alpha_i^* c_i(\mathbf{x}) + \sum_{j=1}^l \beta_j^* h_j(\mathbf{x^*}) \\
& \le f(\mathbf{x}^*)
\end{split}\tag{23}$$
    - 其中


## 4.2 充分性证明

+ 公式（22-1）是梯度为0的条件。
+ 公式（22-2）和公式（22-3）为原问题的不等式约束和等式约束，保证解的可行
+ 公式（22-4）为对偶可行条件，
+ 公式（22-5）为互补松弛条件

所以可以有公式（24）的推断：
+ 第一行为对偶函数在 $(\alpha^*, \beta^*)$ 处的取值
+ 第二行为拉格朗日函数的定义
+ 第三行是因为互补松弛条件和等式约束

$$\begin{split}
\theta_D(\alpha^*, \beta^*) &= L(\mathbf{x}^*, \alpha^*, \beta^*) \\
&= f(\mathbf{x}^*) + \sum_{i=1}^k \alpha_i^* c_i(\mathbf{x}) + \sum_{j=1}^l \beta_j^* h_j(\mathbf{x^*})\\ 
&= f(\mathbf{x}^*)
\end{split}\tag{24}$$

# 5. 举例
考虑如下问题：

$$\begin{split}
\min \quad & x_1^2 + x_2^2 \\
\text{s.t.} \quad & x_1 + x_2 = 1\\
& x_2 \le \eta
\end{split}\tag{25}$$
拉格朗日函数为：
$$L(x_1, x_2, \alpha, \beta) = x_1^2 + x_2^2 + \alpha(x_2 - \eta) + \beta(1 - x_1 -x_2)\tag{26}$$
KKT 方程组如下
$$\begin{split}
& \frac{\partial L}{\partial x_i} = 0, \quad i=1,2 \\
& x_1 + x_2 = 1\\
& x_2 - \eta \le 0 \\
& \alpha \ge 0 \\
& \alpha(x_2 - \eta) = 0
\end{split}\tag{27}$$
对公式（27）求解可得 $\frac{\partial L}{\partial x_1} = 2x_1 - \beta = 0$, $\frac{\partial L}{\partial x_2} = 2x_2-\beta+\alpha=0$。分别求解出 $x_1 = \frac{\beta}{2}$, $x_2 = \frac{\beta}{2} - \frac{\alpha}{2}$；代入约束等式，可以得到 $x_1 = \frac{\alpha}{4} + \frac{1}{2}$, $x_2 = -\frac{\alpha}{4} + \frac{1}{2}$；代入约束不等式 $-\frac{\alpha}{4} + \frac{1}{2} \le \eta$，以下分三种情况讨论：

+ $\eta > \frac{1}{2}$: 可以看出 $\alpha = 0 > 2-4\eta$ 满足所有KKT条件，约束不等式未发挥作用（inactive），$x_1^* = x_2^* = \frac{1}{2}$ 是内部解，目标函数的极小值为 $\frac{1}{2}$
+ $\eta = \frac{1}{2}$: $\alpha = 0= 2-4\eta$ 满足所有KKT条件，$x_1^* = x_2^* = \frac{1}{2}$ 是边界解，因此 $x_2^* = \eta$
+ $\eta < \frac{1}{2}$: 这时约束不等式是生效的（active），$\alpha = 2-4\eta >0$，则 $x_1^* = 1-\eta$ 且 $x_2^* = \eta$， 目标函数极小值是 $(1-\alpha)^2 + \alpha^2$

# 参考

1. [李航，统计学习方法（第二版）](http://www.tup.tsinghua.edu.cn/booksCenter/book_08132901.html)
2. [支持向量机原理详解(四): KKT条件(Part I) - 知乎](https://zhuanlan.zhihu.com/p/62420593)
3. [Karush-Kuhn-Tucker (KKT)条件 - 知乎](https://zhuanlan.zhihu.com/p/38163970)
