#import "@preview/drafting:0.2.0": margin-note, set-page-properties
#import "@preview/showybox:2.0.1": showybox
#import "@preview/lovelace:0.3.0": *
#import "@preview/codly:1.0.0": *

#set text(lang: "cn")

#set page(
  paper: "a4",
  numbering: "1",
  margin: (x:1.6in)
)

#set par(
  justify: true,
  first-line-indent: 0em
)

#set-page-properties()

#let hd1(in_text) = {
  text(size: 18pt)[
    #align(horizon)[
    #heading(level: 1)[#in_text]
    #v(10pt)
    ]
  ]
}
#let hd2(in_text) = {
  text(size: 16pt)[
    #align(center)[
    #heading(level: 2)[#in_text]
    #v(5pt)
    ]
  ]
}
#let hd3(in_text) = {
  text(size: 14pt)[
    #align(center)[
    #heading(level: 3)[#in_text]
    ]
  ]
}
#let hd4(in_text) = {
  text(size: 12pt)[
    #align(left)[
    #heading(level: 4)[#in_text]
    ]
  ]
}

#set heading(numbering: (..numbers) => {
  let level = numbers.pos().len()
  if (level == 2) {
    return numbering("第一章", numbers.pos().at(level - 1))
  } else if (level == 3) {
    return numbering("1.1", numbers.pos().at(level - 2), numbers.pos().at(level - 1))
  } else if (level == 4) {
    return numbering("1.1.1", numbers.pos().at(level - 3), numbers.pos().at(level - 2), numbers.pos().at(level - 1))
  }}
)

#show: codly-init.with()
#codly(number-format: (n) => none)

#align(horizon+center)[#text(size: 28pt)[机器学习]]

#pagebreak()

#outline(
  indent: 2em,
  depth: 3, 
  title: "大纲"
  )

#pagebreak()

#outline(
  indent: 2em, 
  depth: 4,
  title: "目录"
)

#show heading.where(level: 3): it => {
  counter(math.equation).update(0)
  it
}
#set math.equation(numbering: num =>
  numbering("(1.1)", counter(heading.where(level: 2)).get().last(), num),
  supplement: "Eq."
)
#set math.mat(delim: "[", )

#pagebreak()
#hd1("机器学习基础")
#pagebreak()

#hd2("基础知识")

#hd3("NFL定理")

#h(2em) 归纳偏好用于描述当特征相同时，哪些特征更为重要

假设样本空间 $cal(X)$ 和假设空间 $cal(H)$ 为离散。令 $P(h|X, xi_a)$ 代表算法 $xi_a$ 基于训练数据 $X$ 产生假设 $h$ 的概率；令 $f$ 代表希望学习的目标函数。因此，算法在训练集外产生的误差为：

$
E_(o t e) (xi_a|X,f) = sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)
$



其中 $bb(I)(dot)$ 为指示函数，当 $dot$ 为真时返回 1，否则返回 0。

若学习目标为二分类，则 $cal(X) arrow.bar {0,1}$ 且函数空间为 ${0, 1}^(|cal(X)|)$，其中 $|dot|$ 用于计算集合长度。

算法用于解决多个任务，则拥有多个学习的目标函数；假设这些目标函数均匀分布，则这些目标函数的误差总和为：

$
sum_f E_(o t e) (xi_a|X,f) &= sum_f sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)\
&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) sum_f bb(I)(h(bold(x)) eq.not f(bold(x)))\

&text(font: "STFangsong", "根据假设，总有一半是正确的，因此")\

&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) 1/2 2^(|cal(X|))\
&= 2^(|cal(X)|-1) sum_(x in cal(X)-X)P(X)
$

#h(2em) 因此可知，在多目标目标函数均匀分布的情况下，不同算法所得的误差总和相同。实际情况中，某一算法通常只用于解决单一问题，且其目标函数的分布不均匀（即目标函数重要性不同），因此不同算法所得的误差总和不同。

这告诉我们，在某一任务上表现好的算法在另一任务上表现不一定好。

#pagebreak()
#hd3("Monte-Carlo estimation")

Monte-Carlo estimation可以用于估计复杂积分，假设 $f: bb(R)^d arrow bb(R)$，以及一个定义在 $cal(D) in bb(R)^d$ 上的pdf $p: bb(R)^d arrow bb(R)$，期望计算积分：
$
  I = integral_D f(bold(x)) d bold(x)
$
对于上式，假设 $bold(x) tilde p(bold(x))$，则可以变形为：
$
  I = integral_D p(bold(x)) f(bold(x))/p(bold(x)) d bold(x) = bb(E)_p [ f(bold(x))/p(bold(x))]
$
因此可以设计 Monte-Carlo estimator 为：
$
  hat(I)_N = 1/N sum_(i=1)^N f(bold(x_i))/p(bold(x_i))
$
且具有无偏性
- 无偏性：
$
  bb(E)[hat(I)_N] = I
$
- 方差：
$
  "Var"(hat(I)) = 1/N (bb(E)[(f(bold(x))/p(bold(x)))^2] - I^2)
$
特别的，当从均匀分布中采样时，$p(bold(x)) = 1/V$，其中 $V$ 为 $cal(D)$ 的体积，则：
$
  hat(I)_N = V/N sum_(i=1)^N f(bold(x_i))
$
当积分代表期望时，可以使用Monte-Carlo estimation：
$
  I &= integral_D p(bold(x)) f(bold(x)) d bold(x)\
  &= 1/N sum_(i=1)^N f(bold(x_i)), space.quad bold(x_i) tilde p(bold(x))
$


#pagebreak()
#hd3("CNN")

#hd4("Convolution")
卷积通常是指：
$
Y_(i,j) = (X * W)_(i,j) = sum_(m=0)^(M-1) sum_(n=0)^(N-1) X_(i+m, j+n) W_(m,n)
$
卷积操作包括Conv1d和Conv2d

Conv1d: 通常对于一维特征图，给定输入 $X in bb(R)^(N times L times C_text("in"))$，filter $W in bb(R)^(K times C_text("in") times C_text("out"))$，卷积操作为：
$
  Y_(n, C_text("out"),l) = sum_(c_text("in")=0)^(C_text("in")-1) sum_(k=0)^(K-1) X_(n,l+k, c_text("in")) W_(k, c_text("in"), c_text("out"))
$

Conv2d: 通常对于图像或二维特征图，给定输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")) times C_(text("out")))$，其中 $C_(text("in"))$ 为输入通道数，$C_(text("out"))$ 为输出通道数，$K_H, K_W$ 为卷积核的高度和宽度，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c_text("in")=0)^(C_text("in") - 1) sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c_text("in")) W_(m, n, c_text("in"), c_text("out"))
$
其中，$Y in bb(R)^(N times H_text("out") times W_text("out") times C_text("out"))$，$h, w$ 为空间位置索引，$o$ 为输出通道索引

池化层类型包括：最大池化、平均池化。
$
  A_(i, j) = max_(m,n) (a, Y_i,j)\
  A_(i, j) = 1/(M*N) sum_(m=0)^(M-1) sum_(n=0)^(N-1) Y_(i+m, j+n)
$

```python
import torch.nn as nn
conv = nn.Conv1d(in_channels, out_channels, kernel_size)
conv = nn.Conv2d(in_channels, out_channels, kernel_size)
pool = nn.MaxPool2d(kernel_size, stride)
pool = nn.AvgPool2d(kernel_size, stride)
```

#hd4("Depthwise Separable Convolution")

Depthwise Separable Convolution @howard2017mobilenets 是一种轻量级卷积操作，其包括两个步骤：Depthwise Convolution 和 Pointwise Convolution.

Depthwise Convolution 对每一个通道单独使用卷积核 filter.对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")))$，卷积操作为：
$
  Y_(n, h, w, c) = sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c) W_(m, n, c)
$
Pointwise Convolution 使用 $1 times 1$ 的卷积核，对每一个通道进行卷积操作。对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(1 times 1 times C_(text("in")) times C_(text("out")))$，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c=0)^(C_(text("in")) - 1) X_(n, h, w, c) W_(0, 0, c, o)
$
Depthwise Separable Convolution 结合以上两种卷积方式，首先使用 Depthwise Convolution ，然后使用 Pointwise Convolution. 

```py
import torch  
import torch.nn as nn  

depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                      groups=in_channels)
pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
deepwise_separable = pointwise(depthwise(input))
```

#hd3("State-Space model")

State-Space model (SSM) 是用于描述时间序列数据的模型。对于任意时间序列输入 $u(t)$，SSM首先将其映射到 hidden space $x(t)$，然后进一步映射为输出空间 $y(t)$：
$
  u(t) arrow.bar x(t) arrow.bar y(t)
$
SSM以以下形式表示：
$
  x'(t) = A x(t) + B u(t)\
  y(t) = C x(t) + D u(t)
$
解为：
$
  y(t) = sum_(n=0)^t (C A^(t - n) B + D delta(t-n))u(n)
$ <SSM-soultion>
其中，$delta(t-n)$ 为Kronecker delta函数。

#hd3("Fourier Transform")

#hd4("思想")

傅里叶变换的基本思想是：利用无穷个不同频率周期函数的线性组合来表示一个非周期函数。即：
$
  f(t) = sum_i a_i f_i (t)
$

最简单的周期函数为圆周运动，根据欧拉公式，我们可以得到：
$
  e^(i omega t) = cos(omega t) + i sin(omega t)
$
其中 $omega$ 表示旋转速度，正数时为逆时针旋转，负数时为顺时针旋转。圆周运动的频率为 $T = (2 pi)\/omega$. 同时注意到：旋转整数倍周期后回到原点，即
$
  integral_0^(n T) e^(i omega t) d t = 0
$

为了计算方便，我们可以令所有的周期都是 $2 pi \/omega$ 的整数倍，即：
$
  f(t) = sum_(-infinity)^(+infinity) c_k e^(i k omega_0 t)
$

这样一来，我们设定的是正交基：
$
  mat(dots, e^(-2i omega_0 t), e^(-i omega_0 t), 1, e^(i omega_0 t), e^(2i omega_0 t), dots)
$

两边同乘 $e^(i -n omega_0 t)$ 并积分：
$
  integral_0^T f(t) e^(-i n omega_0 t) d t &= sum_(-infinity)^(+infinity) c_k integral_0^T e^(i (k-n) omega_0 t) d t\
  &=T c_n
$
因此：
$
  c_n = 1/T integral_0^T f(t) e^(-i n omega_0 t) d t
$

这里的 $c_n$ 为不同角频率的圆周运动的系数

#hd4("傅里叶变换")

对于一个连续信号 $f: bb(R)^d arrow bb(C)$，其连续傅里叶变换 (CFT) 为 $cal(F): bb(R)^d arrow bb(C)$：
$
  cal(F)(f)(bold(K)) = integral_(bb(R)^d) f(bold(x)) e^(-2 pi i bold(K) dot bold(X)) d bold(X)
$
同时，可以进行逆变换：
$
  cal(F)^(-1)(f)(bold(X)) = integral_(bb(R)^d) f(bold(K)) e^(2 pi i bold(K) dot bold(X)) d bold(K)
$
对于不连续点序列 ${x[n]}_(0 <= n <= N)$，其离散傅里叶变换 (DFT) 为：
$
  cal(F) x[n] = sum_(n=0)^(N-1) x[n] e^(-2 pi i n k \/ N), space.quad k = 0, 1, dots, N-1
$
同理，可以进行逆变换：
$
  cal(F)^(-1) x[k] = 1/N sum_(k=0)^(N-1) x[k] e^(2 pi i n k \/ N), space.quad n = 0, 1, dots, N-1
$
同时，可以通过矩阵乘法表示：
$
  cal(F) x = W dot x
$
其中 $W$ 为 DFT 矩阵，${W_(i,j) = e^(-2 pi i n k \/ N)}_(n,j = 0, dots, N-1)$:
$
  W = 1/sqrt(N) mat(
    1 , 1 , 1 , dots , 1;
    1 , e^(-2 pi i 1 \/ N) , e^(-2 pi i 2 \/ N) , dots , e^(-2 pi i (N-1) \/ N);
    1 , e^(-2 pi i 2 \/ N) , e^(-2 pi i 4 \/ N) , dots , e^(-2 pi i 2(N-1) \/ N);
    dots , dots , dots , dots , dots;
    1 , e^(-2 pi i (N-1) \/ N) , e^(-2 pi i 2(N-1) \/ N) , dots , e^(-2 pi i (N-1)^2 \/ N)
  )
$
其中 $1\/sqrt(N)$ 为归一化系数，使得 $W$ 具有以下性质：
$
  W^(-1) dot W = I
$

#hd4("卷积和卷积定理")
时域上的卷积等价于频域上的乘积，若 $cal(F)[f(t)] = f(omega), cal(F)[g(t)] = g(omega)$，则：
$
  cal(F)[f(t) * g(t)] = F(omega) G(omega)\
  cal(F)[f(t) g(t)] = 1/(2 pi) F(omega) * G(omega)
$ <FFT-conv>
注意：这里的乘法不是矩阵的点乘，而是element-wise的乘法。即对于函数来说，每一点值的乘积作为新的函数值。

#pagebreak()

#hd2("贝叶斯")

贝叶斯涉及以下组件：

似然函数（likelihoo）：表示观测数据在参数 $theta$ 给定情况下的概率，通常记作 $p(D|theta)$，其中 $D$ 为观测数据

先验分布（prior distribution）：表示在没有观测数据时对参数 $theta$ 的信念，记作 $p(theta)$.

后验分布（posterior distribution）：表示在观测数据更新后参数分布，记作 $p(theta|D)$，通常由贝叶斯定理进行计算

#hd3("基础")

#hd4("贝叶斯优点")

贝叶斯的基础形式为：
$
p(y|x) = p(x,y)/p(x) = (p(x|y)p(y))/(p(x)) = (p(x|y)p(y))/(integral p(x|y)p(y) d y)
$
即：
$
text("Posterior") = (text("Likelihood") times text("Prior"))/text("Evidence")
$
考虑一系列观测数据 $X=(x_1,x_2,dots,x_n)$，i.i.d.来自某个分布 $p(x|theta)$，其中 $theta$ 为参数。我们希期通过观测数据来估计参数 $theta$，即获得 $p(theta|X)$.通常情况下我们可以利用MLE进行处理：
$
theta_(text("MLE")) = arg max_theta p(X|theta)=arg max_theta sum_i log p(x_i|theta)
$
如果利用贝叶斯方法，我们可以得到：
$
p(theta|X)=(p(X|theta)p(theta))/p(X) = (p(X|theta)p(theta))/(integral p(X|theta)p(theta) d theta) op("=", limits: #true)^(i i d) (product_i p(x_i|theta)p(theta))/(integral product_i p(x_i|theta)p(theta) d theta)
$
这里的性质在于，使用贝叶斯方法得到的后验概率分布 $p(theta|X)$ 包括了观测数据的信息，这样当我们有新的观测数据时，可以直接利用后验概率分布来估计参数，例如：
$
p(theta|X,x_(n+1)) = (p(x_(n+1)|theta)p(theta|X))/p(x_(n+1)|X) op("=", limits: #true)^(i i d) (p(x_(n+1)|theta)p(theta|X))/(p(x_(n+1)))
$
贝叶斯的优点在于：无论数据大小，都可以得到后验概率分布，这样可以避免过拟合问题。

#hd4("Probabilistic ML model")

判别式概率模型，Discriminative probabilistic ML model，用于分类和回归等任务。其特点是根据条件概率 $p(y|x,theta)$ 进行建模，而不是通过联合概率分布 $p(x,y)$. 即，根据 $x$ 预测 $y$。通常假设 $theta$ 的先验分布与 $x$ 无关，因此有：
$
p(y,theta|x) = p(y|x, theta) p(theta)
$
在这里，$p(y|x,theta)$ 是对与模型的选择，即函数 $y = f(x, theta)$.

生成式概率模型，Generative probabilistic ML model，则是可以根据联合概率分布 $p(x,y,theta)$ 进行建模，最终要获得的是 $p(x,y|theta)$，即
$
p(x,y,theta) = p(x,y|theta)p(theta)
$
贝叶斯模型，假设训练数据 $(X_(t r), Y_(t r))$ 和一个判别式模型 $p(y,theta|x)$，我们可以通过贝叶斯方法来估计参数 $theta$，在训练阶段，我们的 $theta$ 是由训练数据 $(X_(t r), Y_(t r))$ 估计得到的，即 $p(theta|X_(t r), Y_(t r))$. 根据贝叶斯定理：

$
p(theta|X_(t r), Y_(t r)) &= (p(X_(t r), Y_(t r),theta))/(p(X_(t r), Y_(t r)))\
&= (p(Y_(t r)|X_(t r),theta)p(X_(t r)|theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(X_(t r)|theta)p(theta) d theta)\
text("given: ") p(X_(t r)|theta) = P(X_(t r)) &=(p(Y_(t r)|X_(t r),theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(theta) d theta)
$ <BaysianBasic1>
通过训练，我们获得了后验分布 $p(theta|X_(t r), Y_(t r))$. 在测试阶段，加入新数据点 $x$，此时我们可以通过后验分布 $p(theta|X_(t r), Y_(t r))$ 来估计 $y$ 的概率分布：
$
p(y|x,X_(t r),Y_(t r)) = integral p(y|x,theta)p(theta|X_(t r),Y_(t r)) d theta
$ <BaysianBasic2>

这是对所有的模型 $theta$ 进行平均，其中 $p(y|x,theta)$ 代表每个模型（由 $theta$ 表示）的预测，而 $p(theta|X_(t r),Y_(t r))$ 代表这些模型的不确定性，衡量我们对不同参数的信心。

#hd4("Conjugate distribution")

在贝叶斯模型中， @BaysianBasic1 和 @BaysianBasic2 都存在积分计算，在大部分情况下是难以直接获得数值解的。但共轭分布（Conjugate distribution）可以简化这种计算。

共轭分布是指：对于先验分布 $p(theta)$ 、似然函数 $p(X|theta)$和后验分布 $p(theta|X)$，若先验分布和后验分布属于同一分布族（distribution family），则称 $p(theta)$ 和 $p(X|theta)$ 为共轭分布。即：
$
p(theta) in cal(A)(alpha), p(X|theta) in cal(B)(beta) arrow.double p(theta|X) in cal(A)(alpha^prime)
$
这样的好处在于，我们可以直接获得后验分布 $p(theta|X)$ 的形式，从而可以忽略积分的过程，例如：
$
p(theta|X) = (p(theta) p(X|theta)) / (integral p(theta) p(X|theta) d theta)
$ <conjugate>
我们知道 $p(theta|X)$ 的函数形式是与 $p(theta)$ 相同的，即确保了 $integral p(theta|X) d theta = 1$. 因此我们可以忽略积分，得到：
$
p(theta|X) prop p(theta) p(X|theta)
$
接着只需要计算参数即可。

常见的共轭分布：

#set table(stroke: (x, y) => (
  bottom: if y == 0 {1pt},
  right: if x == 0 or x == 1 {1pt},
  ))
#align(center)[#figure(
  table(
    align: horizon + center,
    columns: (40%, 20%, 40%),
    table.header[Likelihood $p(x|theta)$][$theta$][Conjugate prior $p(y)$],
    [Gaussian], [$mu$], [Gaussian],
    [Gaussian], [$sigma^(-2)$], [Gamma],
    [Gaussian], [$(mu, sigma^(-2))$], [Gaussian-Gamma],
    [Multivariate Gaussian], [$Sigma^(-1)$], [Wishart],
    [Bernoulli], [$p$], [Beta],
    [Multinomial], [$(p_1,dots,p_m)$], [Dirichlet],
    [Poisson], [$lambda$], [Gamma],
    [Uniform], [$theta$], [Pareto]
  ),
)]

共轭分布通常只适用于简单概率模型。

#hd4("Maximum posterior estimation")

当共轭分布不可用时，一种简单的方法是使用最大后验估计（maximum a posteriori probability estimate, MAP）.其思想是将分布估计转变为点估计，将参数取为后验分布的最大值，即：
$
theta_(M A P) &= arg max_(theta) p(theta|X_(t r), Y_(t r))\
&= arg max_(theta) (p(Y_(t r)|X_(t r),theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(theta) d theta)\
&integral p(Y_(t r)|X_(t r),theta)p(theta) d theta text("does not depend on ") theta\
&= arg max_(theta) p(Y_(t r)|X_(t r),theta)p(theta)
$
鉴于 $theta_(M A P)$ 为点估计值，此时测试阶段则转变为：
$
p(y|x, X_(t r), Y_(t r)) = p(y|x, theta_(M A P))
$

#pagebreak()
#hd3("Variational inference")

#hd4("KL-divergence")

变分推断（Variational inference）是一种近似推断方法，旨在处理复杂的后验分布。其基本思想是，将后验分布 $p(theta|X)$ 近似为一个简单的分布 $q(theta) in cal(Q)$，使得 $q(theta)$ 尽可能接近 $p(theta|X)$. 通常情况下，我们可以通过最小化两个分布的KL-divergence来获得 $q(theta)$，即：

$
q(theta) &= arg min_(q in cal(Q)) F(q) := K L(q(theta)||p(theta|X))
$

其中，KL-divergence为：
$
K L(q(theta)||p(theta|X)) = integral q(theta) log(q(theta)/p(theta|X)) d theta
$
注意：
1. KL-divergence是非负的，当且仅当 $q(theta) eq.triple p(theta|X)$ 时，KL-divergence为0
2. KL-divergence不对称，即 $K L(q(theta)||p(theta|X)) eq.not K L(p(theta|X)||q(theta))$
3. KL-divergence包含的两个分布必须是相同的支持集，即 $q(theta)$ 和 $p(theta|X)$ 必须在#margin-note()[
  两者必须拥有相同的信息量
]相同的空间上定义
  #showybox()[
    支撑集(support)：它是集合$X$的一个子集，要求对给定的$X$上定义的实值函数$f$在这个子集上恰好非$0$. 特别地，在概率论中，一个概率分布是随机变量的所有可能值组成的集合的闭包。
  ]

存在两个问题：
1. 未知的后验分布 $p(theta|X)$ 导致 KL-divergence 无法计算
2. KL-divergence的优化空间为分布空间，通常情况下是无法直接优化的

#hd4("Evidence lower bound")

证据下界（Evidece lower bound, ELOB）用于解决问题1. 我们知道后验分布的贝叶斯形式为：
$
p(theta|X) = (p(X|theta) p(theta))/ p(X) = (p(X|theta) p(theta))/ (integral p(X|theta) p(theta) d theta) = (text("Likelihood") times text("Prior"))/text("Evidence")
$
考虑 $log p(X)$，我们可以做出以下变形：

$
log p(X) &= integral q(theta) log(p(X)) d theta = integral q(theta) log p(X,theta)/(p(theta|X)) d theta\
&= integral q(theta) log (p(X,theta)q(theta))/((p(theta|X))q(theta)) d theta\
&= integral q(theta) log p(X,theta)/q(theta) d theta + integral q(theta) log q(theta)/p(theta|X) d theta\
&= cal(L)(q(theta)) + K L(q(theta)||p(theta|X))
$ <BaysianVariation1>

其中，$cal(L)(q(theta))$ 为证据下界（Evidence lower bound, ELBO），即：
$
log p(X) >= cal(L)(q(theta))
$

对于 @BaysianVariation1，我们注意到：$log p(x)$ 与 $q(theta)$ 无关，而 $cal(L)(p(theta))$ 和 $K L(q(theta)||p(theta|X))$ 与 $q(theta)$ 有关。因此，我们可以通过最大化 $cal(L)(q(theta))$ 来最小化 $K L(q(theta)||p(theta|X))$，即：
$
q(theta) &= arg min_(q in cal(Q))  K L(q(theta)||p(theta|X))\
&= arg max_(q in cal(Q)) cal(L)(q(theta))\
&= arg max_(q in cal(Q)) integral q(theta) log p(X,theta)/q(theta) d theta\
$
其中 $cal(L)(q(theta))$ 中分布都是已知可以计算的，进一步我们可以得到：
$
cal(L)(q(theta)) &= integral q(theta) log p(X,theta)/q(theta) d theta = integral q(theta) log (p(X|theta)p(theta))/q(theta) d theta\
& = integral q(theta) log p(X|theta) d theta + integral q(theta) log p(theta)/q(theta) d theta\
& = bb(E)_(q(theta)) log p(X|theta) - K L (q(theta)||p(theta))
$

对于第一项 $bb(E)_(q(theta)) log p(X|theta)$，我们需要其最大化。即使取 $log p(X|theta)$ 的加权平均值，其收敛点仍然与MLE一致：将 $q(theta)$ 设置在集中于MLE点估计的位置，有 $hat(theta) = arg max_theta p(X|theta)$，此时参数对应的似然函数 $p(X|theta)$ 取最大值，有：
$
bb(E)_(q(theta)) log p(X|theta) arrow log p(X|theta)
$
这告诉我们最大化第一项会导致参数 $theta$ 逐渐收敛到 $theta_(M L E)$. 同时这一项被叫作 data term，当模型的对数似然性很高时，意味着模型在给定的参数下生成观察数据 $X$ 的概率大于在其他参数下生成数据的概率。因此，可以说模型与观察数据的拟合程度较好

第二项被称为 regularizer，可以防止模型的过拟合。

#hd4("Mean field approximation")

对于 $q(theta)$ 的选择，我们可以使用均场近似（mean field approximation）来简化问题。均场近似假设 $q(theta)$ 可以分解为一系列独立的分布，即：
$
q(theta) = product_(i=1)^(m) q_i (theta_i)
$
均场近似的思想是在每一步中，固定参数 ${q_i (theta_i)}_(i eq.not j)$，只对单一参数 $q_j (theta_j)$ 做优化，其中参数 $theta_i$ 仍然可以为向量，因此我们的目标转化为：
$
q_j (theta_j) = arg max_(q_j (theta_j)) cal(L)(q(theta))
$
#pagebreak()
此时将固定参数视为常量，对 $q_j (theta_j)$ 做解析解，可以得到：
#set math.equation(number-align: top)
$
cal(L)(q(theta)) &= integral q(theta) log p(X,theta)/q(theta) d theta = integral q(theta) log  p(X, theta) d theta - integral q(theta) log q (theta) d theta\
&=integral product_i q_i (theta_i) log p(X, theta) product_i d theta_i - 
integral product_i q_i (theta_i) log product_i q_i (theta_i) product_i d theta_i\
&=integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) log p(X, theta) product_i d theta_i  - integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) log product_i q_i (theta_i) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) (log q_j (theta_j)+log product_(i eq.not j) q_i (theta_i)) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) log q_j (theta_j)  underbrace(integral product_(i eq.not j) q_i (theta_i) product_i d theta_i, "=1") - underbrace(integral q_j (theta_j),"=1") integral product_(i eq.not j) q_i (theta_i) log product_(i eq.not j) q_i (theta_i) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) log q_j (theta_j) d theta_j - integral product_(i eq.not j) q_i (theta_i) sum_(i eq.not j) log q_i (theta_i) product_(i eq.not j) d theta_i\
&=bb(E)_(q_j (theta_j))[bb(E)_(q_(i eq.not j)) log p(X, theta) - log q_j (theta_j)] - text("constant")
$
引入Lagrangian 函数计算其最小值：
$
partial_(q_j (theta_j)) cal(L) (q(theta)) &+ sum_i lambda_i (integral p_i (theta_i) d theta_i - 1) = 0 \
0&= bb(E)_(q_(i eq.not j)) log p(X, theta) - partial_(q_j (theta_j)) bb(E)_(q_j (theta_j)) [log q_j (theta_j)] + partial_(q_j (theta_j)) lambda_j integral p_j (theta_j) d theta_j
$ <BaysianVariation2>
#set math.equation(number-align: horizon)
利用变分法解决求导：
$
partial_(q_j (theta_j)) bb(E)_(q_j (theta_j)) [log q_j (theta_j)] &= partial_(q_j (theta_j)) integral q_j (theta_j) log q_j (theta_j) d theta_j\
&= partial_(q_j (theta_j)) q_j (theta_j) log q_j (theta_j) \
&= log q_j (theta_j) + 1\
partial_(q_j (theta_j)) lambda_j integral p_j (theta_j) d theta_j &= lambda_j
$
因此， @BaysianVariation2 改写为：
$
0&=bb(E)_(q_(i eq.not j)) log p(X, theta)- log q_j (theta_j) - 1 + lambda_j\
&=bb(E)_(q_(i eq.not j)) log p(X, theta) - log q_j (theta_j) + text("constant")\
$
得到：
$
q_j (theta_j) = 1/(Z_j) exp(bb(E)_(q_(i eq.not j)) log p(X, theta))
$
其中 $Z_j$ 为归一化常数，使得 $q_j (theta_j)$ 满足概率分布的性质。
#pagebreak()
因此， Mean field approximation 的算法为：
1. 初始化：
$
q(theta) = product_i q_i (theta_i)
$
2. 重复，直到ELBO收敛：
  - 对于每一个 $q_i (theta_i)$，做如下计算更新：
  $
  q_j (theta_j) = 1/(Z_j) exp(bb(E)_(q_(i eq.not j)) log p(X, theta))
  $
  或
  $
    log q_j(theta_j) = bb(E)_(q_(i eq.not j)) log p(X, theta) + text("constant")
  $
  - 重新计算 ELBO:
  $
  cal(L)(q(theta))
  $
问题在于如何计算 $Z_j$ 与期望 $bb(E)_(q_(i eq.not j)) log p(X, theta))$ 是否能够被解析解计算出来。如果要确保其能够被计算出来，需要假设 $theta arrow [theta_1,dots, theta_m]$ 的过程具有共轭性，即：
$
  forall theta_j,space.quad p(theta_j|theta_(i eq.not j)) in cal(A)(alpha_j), &space.quad p(x| theta_j, theta_(i eq.not j)) in cal(B)(beta_j) \
  &arrow p(theta_j|X, theta_(i eq.not j)) in cal(A)(alpha^prime)
$
在实际操作中，可以这样对共轭性进行检验：

对于每一个 $theta_j$:
- 固定 ${theta_i}_(i eq.not j)$ （将其视为常数）
- 检查 $p(X|theta)$ 与 $p(theta)$ 是否对于 $theta_j$ 是共轭的

#pagebreak()
#hd3("Dirichlet process mixture model")

#hd4("Dirichlet process")

Dirichlet process 是一种用于非参数贝叶斯统计的随机过程，可以创建无限个连续分布 $H$ 的离散副本，即 $H arrow.bar G$. 记作 $G tilde D P(alpha, H)$，其中 $alpha$ 为浓度参数，决定了聚类的程度。浓度参数越大，生成的簇的数量通常越多。

Dirichlet process 可以由 stick-breaking process 来描述。假设我们有一个长度为1的棍子，我们从棍子的一端开始，每次从棍子的长度中折断一部分，折断的长度服从beta分布，折断的位置服从beta分布。这样我们可以得到一个无限个分布的序列：

1. 生成一个无限长序列 $V_k tilde text("Beta")(1, alpha) in (0,1)$
2. 生成一个无限长序列权重 $pi_k$:
$
pi_k = V_k product_(j=1)^(k-1) (1-V_j), sum_(k=1)^infinity pi_k = 1
$
3. 从连续分布 $H$ 中抽取无限多个样本 $theta_k$，构成新的分布 $G$:
$
G = sum_(k=1)^infinity pi_k delta_(theta_k), delta_(theta_k) = delta(theta-theta_k)
$

Dirichlet process 有以下性质：

1. 期望不变：
$
bb(E)_(D P (alpha, H)) [x] = bb(E)_(H) [x]
$
2.
$
alpha arrow infinity arrow.double D P(alpha, H) = H
$
3. 序列为无限长，无法完全在计算机中表达

#hd4("Mixture model")

混合模型（Mixture Model）是一种统计模型，它假设数据来自多个不同的分布，每个分布被称为一个“成分”（component）。混合模型的主旨在于通过这些成分的组合来更好地描述数据的总体分布。每个成分可以用不同的概率分布函数来定义，如高斯分布、伯努利分布等。

对于给定的观察值 $x$ ，混合模型的概率密度函数可以表示为:
$
p(x) = sum_(k=1)^K pi_k p(x|theta_k), sum_(k=1)^K pi_k = 1
$

混合模型广泛应用于许多领域，包括但不限于：

- 聚类：通过对数据进行分组，帮助识别数据中的模式和结构。例如，K均值聚类可以看作是高斯混合模型的一个特例。
- 密度估计：能够捕捉到复杂的分布形状，比单一的概率分布（如正态分布）更具灵活性。
- 信号处理：在音频和图像处理等领域中，用于建模混叠信号。
- 生物信息学：在基因表达数据分析中识别不同的生物状态。

考虑利用核混合模型（Kernel Mixture Model）估计pdf，我们可以初步写为：
$
f(y|P) = integral cal(K)(y|theta) d P(theta)
$ <DPMM1>
其中，$cal(K)(dot|theta)$ 为核函数，以 $theta$ 作为其参数，可能包括每个核的中心位置、宽度等；$P$ 是混合测度。

#showybox[
  测度（measure）：测度是一个函数，它将集合映射到实数。常见的测度包括：长度、面积、概率测度等。假设样本空间 $Omega$，一个概率测度 $P$ 满足以下条件：
  - $P(A) >= 0, forall A in Omega$
  - $P(Omega) = 1$
  - 对于不相交事件 $A_1, A_2, dots$，有 $P(union.big_i A_i) = sum_i P(A_i) $

  混合测度（mixture measure）：混合测度是指在混合模型中，表示由多种不同的分布组合而成的分布特征。混合测度可以写作：
  $
  P = integral P_(theta) d H(theta)
  $ <DPMM2>
  其中：$P_theta$ 是给定参数的成分分布，例如可以是正态分布、指数分布等；$H$ 是先验测度，描述参数 $theta$ 的分布。

]

因此，@DPMM1 计算了在参数空间中，对于每个 $theta$ 的核函数的加权求和。

#hd4("DP for mixutre model")

DP适合用作为未知混合分布的先验。例如在 @DPMM1 中，考虑其为无限核混合模型，混合测度 $P$ 视为未知。此时，我们可以令 $P tilde pi_(cal(P))$，其中 $cal(P)$ 代表在样本空间中所有可能的概率测度，$pi_(cal(P))$ 则代表样本空间中的先验分布。考虑将 $pi_(cal(P))$ 选做 DP 的先验，这样我们可以获得一个离散的 DP 混合模型:
$
f(y) = sum_(k=1)^infinity pi_k cal(K)(y|theta_k)
$
其中 $pi_i$ 来自 DP 的权重（参数为 $alpha$），且：
$
y_i tilde cal(K)(theta_i), space.quad theta_i tilde P, space.quad P tilde D P(alpha, P_0)
$
采用DP作为 $P$ 的先验，会导致后验计算变得复杂。根据 @DPMM2 我们可知，以 DP 作为先验的混合模型，其混合测度拥有无限个参数，因此在拥有样本 $y^n=(y_1,y_2,dots,y_n)$ 的条件下无法直接获得 $P$ 的后验分布。解决方法是通过边缘化 $P$ 来获得参数 $theta^n=(theta_1, theta_2,dots,theta_n)$ 的先验分布。具体的，我们可以用Polya urn 预测规则来描述这种情形。即：
$
p(theta_i|theta_1,dots,theta_(i-1)) tilde (alpha / (alpha+i-1)) P_0(theta_i) + sum_(j=1)^(i-1) delta_(theta_j)
$ <DPMM3>

从 @DPMM3 开始进行聚类，假设有 $n$ 个样本，$k$ 个簇，$n_k$ 代表第 $k$ 个簇的样本数量，$theta_k$ 代表第 $k$ 个簇的参数，我们有：

$
p(theta_i|theta_(-i)) = 
underbrace(
  (alpha / (alpha+n-1)) P_0(theta_i),
  "新建聚类"
) + 
underbrace(
  sum_(h=1)^(k^((-i))) 
  overbrace(
    ( n_h^((-i))/(alpha+n-1)),
    "DP中聚类h的权重/占比")
    delta_(theta_h^(*(-i))),
  "选择已有聚类"
)
$ <DPMM4>
其中 $theta_h^*, h=1,dots,k^((-i))$ 是 $theta_(-i)$ 中的唯一值，代表了在移除第 $i$ 个样本后剩下的唯一聚类参数；$n_h^((-i)) = sum_(j eq.not i) 1_(theta_j = theta_h^*)$ 代表除去第 $i$ 个样本后，第 $h$ 个簇的样本数量。

#hd4("DPMM with Gibbs sampling")
Gibbs sampling 允许我们从多维分布中抽样，通过迭代更新每个维度的样本。对于 DPMM，我们可以通过 Gibbs sampling 来优化 @DPMM4:

令 $theta^* = (theta_1^*, dots, theta_k^*)$ 为参数 $theta$ 的唯一值，且令 $S_i$ 为第 $i$ 个样本的聚类分配，即若 $theta_i = theta_c^*$ 则 $S_i = c$. Gibbs sampler 的步骤如下：
#set math.cases(gap: 1em)
1. 通过从多项式条件后验中抽样来更新分配 $S$:
$
P(S_i = c|S_(i-1),theta^*,alpha,P_0) prop cases(
  &n_c^((-i)) cal(K)(y_i|theta_c^*)\, &c=1\,dots\,k^((-i)),

  &alpha integral cal(K)(y_i|theta) d P_0(theta)\, space.quad &c=k^((-i))+1
)
$
2. 通过从条件后验中抽样来更新参数 $theta^*$:
$
p(theta_c^*|-) prop P_0(theta_c^*) product_(i:S_i=c) cal(K)(y_i|theta_c^*)
$
其中 $product_(i:S_i=c) cal(K)(y_i|theta_c^*)$ 是聚类 $c$ 中所有样本在参数 $theta_c^*$ 下的似然函数的乘积，反应了在当前聚类分配下，样本数据对于参数的支持程度。

聚类行为的控制因素：

1. 浓度参数 $alpha$：$alpha$ 的大小直接影响聚类的数量。当 $alpha$ 接近于0时，获取的聚类会趋向于集中，从而展现出一种共同参数 $y_i tilde cal(K)(theta)$ 的行为。相反，增大 $alpha$ 则有更高的可能性去接受新聚类。
2. 先验 $P_0$ 的方差：高方差的先验 $P_0$ 表示对聚类位置的不确定性，阻碍新聚类的形成。

#pagebreak()
#hd2("Latent variable model")

#hd3("Laten variable model")

潜变量模型（Latent variable model）是一种统计模型，其中包含了一些未观测的变量，这些变量通常被称为潜变量（latent variable）。潜变量模型通常用于描述数据背后的潜在结构，以及数据生成的机制。潜变量模型可以用于多种任务，如聚类、降维、异常检测等。

对于观测数据 $X$ 与其模型参数 $theta$，要估计模型参数 $theta$，通常采用MLE方法，即
$
theta_("MLE") = arg max_theta log p(X|theta)
$

我们假设存在某种潜在变量 $Z$，其与观测数据 $X$ 之间存在关系，此时我们可以对 $log p(X|theta)$ 进行分解，类似 @BaysianVariation1：
$
  log p(X|theta) &= integral q(Z) (p(X,Z|theta))/q(Z) d Z  + integral q(Z) log q(Z)/p(Z|X,theta) d Z\
  &= cal(L)(q, theta) + K L(q||p) >= cal(L)(q, theta)
$
因此，我们可以通过最大化 ELBO $cal(L)(q, theta)$ 来使 $log p(X|theta)$ 尽可能大。对于ELOB，给出其广泛定义 variational lower bound:

#showybox()[
  函数 $g(xi, x)$ 是另一函数 $f(x)$ 的 variational lower bound，当且仅当：

  - $forall xi, f(x) >= g(xi, x)$
  - #margin-note($forall x_0, exists xi(x_0) arrow.double f(x_0) = g(xi(x_0), x_0)$)[例如过二次函数最低点的切线]
  这样，对于：
  $
    x = arg max_x f(x)
  $
  我们可以通过对 $g(xi, x)$ 区块坐标更新（Block-coordinate updates）来获得 $x$ 的近似解，即：
  $
    x_n &= arg max_x g(xi_(n-1), x)\
    xi_n &= xi(x_n) = arg max_xi g(xi, x_n)
  $
]

#hd3("EM Algorithm")
#hd4("EM Algorithm")

在最大化 $log p(X|theta)$ 时，我们不仅要最大化参数 $theta$，还要最大化潜变量 $Z$ 的分布，即
$
  cal(L)(q, theta) = integral q(Z) (p(X,Z|theta))/q(Z) d Z arrow max_(q,theta)
$
E-step，设置一个初始点 $theta_0$，类似 @BaysianVariation1：
$
  q(Z) &= arg max_q cal(L)(q, theta_0) = arg min_q  K L(q||p)\
  &= p(Z|X, theta_0) = (p(X,Z|theta_0))/p(X|theta_0) =  (p(X,Z|theta_0))/ (integral_i p(X,z_i|theta_0) d z_i)\
$ <EM-E-step>

当 $p(Z|X, theta_0)$ 无法获得解析解时，可以采取variational inference的方法。

M-step，考虑到 $Z$ 的具体值不明确但知道其分布 $q(Z)$，我们采用其期望：
$
  theta_* = arg max_theta cal(L)(q, theta) = arg max_theta bb(E)_Z log p(X,Z|theta)
$
将新的到的 $theta_*$ 传入 E-step，重复直到收敛。在更新过程中 variational lower bound 是单调递增的，因此可以保证收敛

#figure(
  image(
    "asset/EM-algorithm.png",
    width: 100%
    )
)

#hd4("Categorical latent variables")

对于离散型潜变量，假设 $z_i in {1,dots,K}$，则
$
  p(x_i|theta) = sum_k p(x_i, z_i = k|theta) = sum_k p(x_i|z_i = k, theta) p(z_i = k|theta)
$
进行EM，E-step：
$
  q(z_i = k) &= p(z_i = k|x_i, theta) \
  &= (p(x_i|z_i = k, theta) p(z_i = k|theta)) / p(x_i|theta)\
  &= (p(x_i|z_i = k, theta) p(z_i = k|theta)) / (sum_l p(x_i|z_i = l, theta) p(z_i = l|theta))
$
M-step：
$
  theta_* &= arg max_theta bb(E)_Z log p(X,Z|theta)\
  &= arg max_theta sum_i sum_k q(z_i = k) log p(x_i, z_i = k|theta)
$
而对于连续型潜变量：
$
  p(x_i|theta) = integral p(x_i, z_i|theta) d z_i = integral p(x_i|z_i, theta) p(z_i|theta) d z_i
$
E-step：
$
  q(z_i) &= p(z_i|x_i, theta) = p(z_i|x_i, theta) / p(x_i|theta)\
  & = (p(x_i|z_i, theta) p(z_i|theta)) / (integral p(x_i|z_i, theta) p(z_i|theta) d z_i)
$
根据 @conjugate，只有当 $p(X|Z,theta)$ 与 $p(Z|theta)$ 为共轭分布时，E-step才能获得解析解；否则，需要使用stochastic variational inference的方法

对于连续型潜变量，其重要应用之一在于representation learning：
1. 表示学习的目标 ：Representation learning 的核心目标是生成有效的数据表示，而连续潜变量提供了一个强大的工具来建模数据的内在结构。
2. 潜变量在表示学习中的作用 ：通过引入连续潜变量，模型能够更灵活地捕捉数据的连续变化和模式，形成有效的表示。这些潜变量通常在隐藏层中起作用，影响最终输出的生成。
3. 生成模型中的应用 ：许多现代的生成模型，如GAN（生成对抗网络）和VAE，都利用连续潜变量来生成新样本，通过学习数据的潜在结构来提高生成能力。
4. 优化和推断 ：在representation learning的上下文中，涉及到从观测数据中推断潜变量的分布，并优化这些潜变量以获得更好的数据表示。连续潜变量可以利用梯度下降等优化方法进行推断。

#hd3("VAE")

#hd4("Mixture PCA")
在线性代数下视角的PCA一般涉及特征值分解与主成分投影，对于 $n$ 个具有 $p$ 维特征的数据 $bold(X)in bb(R)^(n times p)$，将其中心化后计算协方差矩阵：
$
  bold(Sigma) = 1/n bold(X)^tack.b bold(X)
$
然后对协方差矩阵进行特征值分解：
$
  bold(Sigma) bold(v)_j = lambda_j bold(v)_j, space.quad j = 1,2,dots, p
$
其中 $lambda_1>=lambda_2>=dots>=lambda_p>=0$ 为特征值，$bold(v)_j$ 是对应的特征向量。选择前 $k$ 个特征，将中心化后的数据投影到前 $k$ 个特征向量上，得到降维的表示：
$
  bold(Z) = bold(X) bold(V)_k in bb(R)^(n times k)\
  bold(V)_k = [bold(v)_1, bold(v)_2, dots, bold(v)_k] in bb(R)^(p times k)
$
在latent variable model视角下，PCA可以被视为一个潜变量模型，其中潜变量为降维后的数据。假设 $x in bb(R)^D, z in bb(R)^d, D >> d$，则：
$
  p(X,Z|theta) &= product_i p(x_i|z_i, theta) p(z_i|theta)\
  &= product_i cal(N)(x_i|bold(V)z_i+mu, sigma^2 I) cal(N)(z_i|0, I)
$ <PCA-latent1>
$theta$作为参数，包含 $bold(V) in bb(R)^(D times d), mu, sigma in bb(R)^D$
我们可以利用 EM 算法求解 latent variable model 视角下的PCA。以 @PCA-latent1 中假设为gaussian分布为例，考虑到gaussian和gaussian互为共轭，因此可以获得解析解。使用EM而不是直接求解PAC的好处在于：
- EM算法每一个迭代的的复杂度为 $O(n D d)$，而直接求解PCA的复杂度为 $O(n D^2)$；因此当 $D>>d$ 时EM算法更加高效
- 可以处理缺失的 $x_i$ 或者多观察的 $z_i$
- 可以通过确定 $p(theta)$ 自动确定 $d$ 的值，而不需要想PCA一样提前确定
- 可扩展至混合PCA

现在考虑混合PCA(Mixture PCA)，即降维后的数据存在于多个子空间中，假设 $x in bb(R)^D, t in {1, dots, K}, z in bb(R)^d$，其中 $t$ 是每个子空间的索引，则有：
$
  p(X,Z,T|theta) &= product_i p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta)\
  & = product_i cal(N)(x_i|bold(V)_(t_i)z_i+mu_(t_i), sigma^2 I) cal(N)(z_i|0, I) pi_(t_i)
$
其中参数 $theta$ 包含 $bold(V)_k in bb(R)^(D times d), mu_k, sigma_k in bb(R)^D, pi_k in bb(R)^K$，并有 $p(t_i = k) = pi_k$

E-step:
$
  q(Z,T) &= p(Z,T|X, theta) = product_i p(z_i, t_i|x_i, theta)\
  &= product_i (p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta)) / (sum_i integral p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta) d z_i)
$
M-step:
#set math.equation(number-align: top)
$
  theta_* &= arg max_theta bb(E)_(Z,T) log p(X,Z,T|theta)\
  &= arg max_theta sum_i bb(E)_(Z,T) [log p(x_i|z_i, t_i, theta) + log p(z_i|theta) + log p(t_i|theta)]
$
#set math.equation(number-align: horizon)

通常来说，PCA仅构造线性子空间，即只能捕捉数据中的线性关系。然而，在许多实际应用中，数据常常分布在非线性流形上。例如，在图像处理、自然语言处理等领域，数据往往具有复杂的模式和结构，这些模式和结构不能用平面或超平面来描述。

为了更好地处理非线性数据分布，可以使用一些其他的降维技术，例如：
- t-SNE：一种有效的降维方法，特别适合于处理高维数据的非线性结构，能够保留局部邻域的相似性。
- UMAP：类似于t-SNE，但更注重全局结构的同时保持局部结构。
- 自编码器 ：基于神经网络的方法，可以学习复杂的非线性映射，例如VAE。

#hd4("VAE")

假设 $X in bb(R)^(n times D), Z in bb(R)^(n times d)$，则 latent variable model 告诉我们：
$
  p(X,Z|theta) &= product_i p(x_i|z_i, theta) p(z_i|theta)\
  &= product_(i=1)^n (product_(j=1)^D cal(N)(x_(i j)|mu_j (z_i), sigma^2_j (z_i)))cal(N)(z_i|0, I)
$
在这里，我们不需要局限 $cal(N)(dot)$ 的 $mu$ 为 $z_j$ 的线性函数，而是可以使用神经网络来表示非线性的 $mu_j (z_i)$ 和 $sigma^2_j (z_i)$，这样我们可以学习到更复杂的非线性关系

但使用非线性的 $mu_j (z_i)$ 和 $sigma^2_j (z_i)$ 会导致 $p(x_i|z_i, theta)$ 与 $p(x_i|theta)$ 不再共轭，因此无法获得解析解。同样，在EM算法的E-step中，我们也无法获得后验的解析解。即，无法计算：
$
  q(Z) = p(Z|X, theta) = (p(X,Z|theta))/p(X|theta)
$
从 @EM-E-step 可知，我们从 $K L(q||p)$ 的定义直接推导出可以通过求解 $p(Z|X, theta)$ 来替代求解 $q(Z)$. 而在不能求解 $p(Z|X, theta)$ 的情况下，我们则利用 variational inference 的方法求解 $q(Z)$. 例如我们可以利用mean field approximation 的方法，即：
$
  q(Z) = product_i q_i (z_i)
$
但是在神经网络的视角下，我们完全可以另外#margin-note([训练一个神经网络用于拟合$q(Z)$])[回归神经网络的本质：拟合]，用参数 $phi$ 表示：
$
  q(z_i|x_i, phi) &approx p(z_i|x_i, theta)\
  q(z_i|x_i, phi) &= product_(j=1)^d cal(N)(z_(i j)|mu_j (x_i), sigma^2_j (x_i))
$
因此：
$
  &"encoder:" phi: x arrow.bar q(z|x, theta), bb(R)^D arrow bb(R)^(2d)\
  &"decoder:" theta: z arrow.bar p(x|z, theta), bb(R)^d arrow bb(R)^(2D)
$
其中 $2d$ 与 $2D$ 都是包括了 $mu$ 和 $sigma^2$ 的两个参数

优化神经网络 $phi$ 等价于：
$
  q(Z|X, phi) = arg min_phi K L(q(Z|X, phi)||p(Z|X, theta))
$ <VAE-opt>
根据 @BaysianVariation1，@VAE-opt 等价于最大化 ELBO：
$
  q(Z|X, phi) &= arg max_(phi, theta) cal(L)(phi, theta)\
  &= arg max_(phi, theta) integral q(Z|X,phi) log p(X,Z|theta)/q(Z|X,phi) d Z\
$
鉴于在非共轭情况下我们无法使用EM对 $q(Z|X, phi)$ 进行求解，我们因此使用 stochastic gradient 的方法

#hd4("Stochastic gradient")

Stochastic gradient 通常使用mini-batch与Monte-Carlo estimation 来优化 ELBO，对于
$
  cal(L)(phi, theta) &= integral q(Z|X,phi) log p(X,Z|theta)/q(Z|X,phi) d Z\
  &=integral q(Z|X,phi) log (p(X|Z,theta)P(Z))/q(Z|X,phi) d Z
$
对 $theta$ 求导有：
#set math.equation(number-align: top)
$
  nabla_theta cal(L)(phi, theta) &= nabla_theta integral q(Z|X,phi) log (p(X|Z,theta)P(Z))/q(Z|X,phi) d Z\
  &= sum_(i=1)^n integral q(z_i|x_i, phi) nabla_theta log (p(x_i|z_i, theta)P(z_i))/(q(z_i|x_i, phi))d z_i\
  &= sum_(i=1)^n integral q(z_i|x_i,phi) nabla_theta log p(x_i|z_i, theta) d z_i\
  &approx n integral q(z_i|x_i, phi) nabla_theta log p(x_i|z_i,theta) d z_i, i tilde cal(U){1,dots, n} space.quad "(mini-batch)"\
  &approx n/abs(cal(U)) sum_(i in cal(U)) nabla_theta log p(x_i|z^*_i,theta), z^*_i tilde q(z_i|x_i, phi) space.quad "(Monte-Carlo estimation)"\
  &= n nabla_theta log p(x_i|z^*_i,theta), space.quad i tilde cal(U){1,dots, n}, abs(cal(U))=1, z^*_i tilde q(z_i|x_i, phi)
$
#set math.equation(number-align: horizon)

其中 $cal(U)$ 为数据集的子集，其大小为 $abs(cal(U))$，包含随机选择的$({x}_i, {z}_i)^abs(cal(U))$ ；$z^*_i$ 为从 mini-batch 中的 $q(z_i|x_i, phi)$ 采样的样本。通常来说，先对数据集进行 mini-batch 选择，然后在选定的 mini-batch 中进行 Monte-Carlo 采样，这样可以提高计算效率

对 $phi$ 求导有：
$
  nabla_phi cal(L)(phi, theta) = nabla_phi [&integral q(Z|X,phi) log p(X|Z,theta) d Z \
  - &integral q(Z|X,phi) (log q(Z|X,phi))/P(Z) d Z]\
$ <VAE-phi-dri>
第一项：
#set math.equation(number-align: top)
$
  nabla_phi integral q(Z|X,phi) log p(X,Z|theta) &= integral log p(X|Z,theta) nabla_phi q(Z|X,phi) d Z\
  &= integral q(Z|X,phi) log p(X|Z,theta) nabla_phi log q(Z|X,phi) d Z space.quad "(log-derivative trick)"\
  &= sum_(i=1)^n integral q(z_i|x_i,phi) log p(x_i|z_i,theta) nabla_phi log q(z_i|x_i,phi) d z_i\
  &= n integral q(z_i|x_i,phi) log p(x_i|z_i,theta) nabla_phi log q(z_i|x_i,phi) d z_i, i tilde cal(U){1,dots, n} space.quad "(mini-batch)"\
  &= n/abs(cal(U)) sum_(i in cal(U)) log p(x_i|z^*_i,theta) nabla_phi log q(z_i|x_i,phi), z^*_i tilde q(z^*_i|x_i, phi) space.quad "(Monte-Carlo estimation)"\
  &= n log p(x_i|z^*_i,theta) nabla_phi log q(z^*_i|x_i,phi), space.quad i tilde cal(U){1,dots, n}, abs(cal(U))=1, z^*_i tilde q(z_i|x_i, phi)
$
#set math.equation(number-align: horizon)

注意到 $nabla_phi log q(z^*_i|x_i,phi)$ 为分数函数 (score function)，具有以下性质：
- $
    bb(E)[nabla_theta log p(x|theta)] &= integral p(x|theta) nabla_theta log p(x|theta) d x\
    &= integral p(x|theta) 1/p(x|theta) nabla_theta p(x|theta) d x\
    &= nabla_theta integral p(x|theta) d x = 0
  $
  考虑到 $z^*_i$ 为我们抽样与 $q(z_i|x_i,phi)$ 的样本，因此可以认为 $nabla_phi q(z^*_i|x_i,phi)$ 在 $0$ 附近震荡，且其质量严格与抽样情况相关。除非 $n$ 具有较大值，否则几乎为0，因此可以认为梯度下降时速度较慢
- $
    "Var"(nabla_theta log p(x|theta)) &= bb(E)[nabla_theta log p(x|theta)^2]\
    &= I(theta)
  $
  可知 score function 的方差为 fisher information. Fisher information 反应了估计参数 $theta$ 的方差下界，即 Cramer-Rao Lower Bound，它越大，代表估计的 $theta$ 约精确。因此可以认为在梯度下降过程中，随着参数 $theta$ 的逼近精确值，方差逐渐增加，导致收敛速度变慢

综上所述 $nabla_phi log q(z^*_i|x_i,phi)$ 的存在会导致梯度下降效率降低，因此一般不这么做，而使用 reparameterization trick 来避免这个问题。

#hd4("Reparameterization trick")

考虑复杂期望的求导：
$
  partial / partial_x integral p(y|x) h(x, y) d y
$
假设 $y$ 可以被表达为一个随机变量 $epsilon$ 与 $x$ 的函数，即 $y = g(x, epsilon)$，利用Monte-Carlo estimation，我们可以将上式改写为：
$
  integral p(y|x) h(x, y) d y &= integral r(epsilon) h(x, g(x, epsilon)) d epsilon\
  &approx d/(d x) h(x, g(x, epsilon^*)), space.quad epsilon^* tilde r(epsilon)\
  &= partial/(partial x) h(x, g(x, epsilon^*)) + partial/(partial y) h(x, g(x, epsilon^*)) partial/(partial x) g(x, epsilon^*)\
$
常见的 reparameterization trick 有：
#figure(
  table(
    columns: (auto, auto, auto),
    rows: (2em, 3em, 3em, 3em),
    [$p(y|x)$], [$r(epsilon)$], [$g(epsilon,x)$],
    [$cal(N)(y|mu, sigma^2)$], [$cal(N)(epsilon|0, 1)$], [$x = mu + sigma epsilon$],
    [$cal(G)(y|1,beta)$], [$cal(G)(epsilon|1, 1)$], [$x = beta epsilon$],
    [$epsilon(y|lambda)$], [$cal(U)(epsilon|0,1)$], [$x = -log(epsilon)/lambda$],
    [$cal(N)(y|mu, Sigma)$], [$cal(N)(epsilon|0,I)$],[$x=A epsilon + mu$, where  $A A^tack.b = Sigma$]
  )
)

reparameterization trick 并不适用于所有连续分布，且不适用于离散分布。对于离散分布，我们可以使用 Gumbel-Softmax trick 来进行 reparameterization

对于ELBO @VAE-phi-dri 第一项，我们首先进行mini-batch，然后对其使用 reparameterization trick，最后使用Monte-Carlo estimation：
$
  nabla_phi integral q(Z|X,phi) log p(X|Z,theta) &approx n nabla_phi integral q(z_i|x_i, phi) log p(x_i|z_i,theta) d z_i\
  &=n nabla_phi integral r(epsilon) log p(x_i|g(epsilon,x_i,phi)z_i, theta) d epsilon\
  &approx n nabla_phi log p(x_i|g(epsilon^*,x_i,phi)z_i, theta)\
  &i tilde cal(U){1,dots, n}, abs(cal(U))=1, z_i=g(epsilon,x_i,phi), epsilon^* tilde r(epsilon)
$

#hd4("VAE Algorithm")
我们的目标函数则为：
$
  cal(L)(phi,theta) = bb(E)_(q(Z|X,phi)) log p(X|Z, theta) - K L(q(Z|X,phi)||p(Z))
$
其中， $q(Z|X,phi)$ 为 encoder 网络，$p(X|Z,theta)$ 为decoder

接着更新 $phi, theta$，迭代直至收敛。因为两个都是无偏估计且存在ELBO的下界保证，因此可以保证收敛

对于 $X in bb(R)^(n times d)$，随机选取 mini-batch $cal(U) = ({x}_i, {z}_i)^abs(cal(U))$，计算：
$
  "stoch.grad"_theta cal(L)(phi, theta) = n/abs(cal(U)) sum_(i in cal(U)) nabla_theta log p(x_i|z^*_i,theta)
$
其中 $z_i^* tilde q(z_i|x_i,phi)$
$
  "stoch.grad"_phi cal(L)(phi, theta) = n/abs(cal(U)) sum_(i in cal(U)) nabla_phi log p(x_i|g(epsilon^*,x_i,phi), theta)&\
  - nabla_phi K L(q(z_i|x_i,phi)||p(z_i))&
$
其中 $z_i=g(epsilon,x_i,phi), epsilon^* tilde r(epsilon)$

#hd4("VAE Code")

对于目标函数：
$
  cal(L)(phi,theta) = bb(E)_(q(Z|X,phi)) log p(X|Z, theta) - K L(q(Z|X,phi)||p(Z))
$
对于第一项期望，可以使用 Monte-Carlo estimation，这样只剩下 $log p(X|Z,theta)$，根据数据类型一般考虑 $p(X|Z,theta)$ 为 Bernoulli 或者 Gaussian 分布，在 Bernoulli 分布下，我们可以使用交叉熵损失函数:
$
  "loss" = -sum_i sum_j [x_(i j) log y_(i j) + (1-x_(i j)) log (1-y_(i j))]
$
考虑到 $x_(i j)$ 与 $y_(i j)$ 范围取值为 $[0,1]$，因此通常来说都可以使用交叉熵损失函数

对于KL-divergence，考虑到 $q(Z|X,phi)$ 为高斯分布，$p(Z)$ 为标准高斯分布，因此 KL-divergence 可以直接计算：
$
  K L(q(Z|X,phi)||p(Z)) = 1/2 sum_(i=1)^d [1 + log(sigma_i^2) - mu_i^2 - sigma_i^2]
$
#showybox(breakable: true)[
  proof：

  对于两高斯分布的 KL-divergence 的：
  #set math.equation(number-align: top)
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(mu', Sigma')) =\
    &1/2 [tr(Sigma'^(-1) Sigma) + (mu'-mu)^tack.b Sigma'^(-1) (mu'-mu) - k + log(det(Sigma'^(-1) Sigma))]
  $
  其中 $Sigma$ 为协方差矩阵，$k$ 为维度

  特殊的，当一个高斯分布为标准高斯分布时，其 KL-divergence 为：
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(0, I)) = 1/2 [tr(Sigma) + mu^tack.b mu - k - log(det(Sigma))]
  $
  更特殊的，对于对角协方差矩阵$Sigma = "diag"(sigma_1^2, sigma_2^2, dots, sigma_k^2)$
  有：
  $
    "tr"(Sigma) = sum_(i=1)^k sigma_i^2, space.quad "det"(Sigma) = product_(i=1)^k sigma_i^2
  $
  因此：
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(0, I)) = 1/2 [sum_(i=1)^k (1 + log(sigma_i^2) - mu_i^2 - sigma_i^2)]
  $
]
因此，目标函数可以改写为：
$
  cal(L)(phi, theta) tilde.eq 1/2 sum_(i=1)^d [1 + log(sigma_i^2) - mu_i^2 - sigma_i^2] + 1/L sum_(l=1)^L log p(x|z_l, theta)
$
其中 $z_l = mu + sigma dot.circle epsilon, epsilon tilde cal(N)(0,I)$

```python
def loss_function(recon_x, x, mu, logvar):
    """
    计算VAE的损失函数，包括重构损失和KL散度
    """
    # 重构损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL散度计算
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```
对于 encoder $P(Z|X,phi)$，要计算 $mu$ 和 $log sigma^2$，这样我们可以根据 $mu$ 和 $log sigma^2$ 采样 $z$，这样我们可以保证 $z$ 为高斯分布
```python
def encode(self, x):
    """
    编码器前向传播，输出潜在变量的均值和对数方差
    """
    h1 = F.relu(self.fc1(x))
    mu = self.fc_mu(h1)
    logvar = self.fc_logvar(h1)
    return mu, logvar
def reparameterize(self, mu, logvar):
    """
    重参数化技巧，从 Q(Z|X) 中采样潜在变量 Z
    """
    std = torch.exp(0.5 * logvar)  # 标准差 σ
    eps = torch.randn_like(std)    # 从标准正态分布中采样 ε
    return mu + eps * std          # 潜在变量 Z
```
对于 decoder $P(X|Z,theta)$，我们可以直接计算重构的 $x$：
```python
def decode(self, z):
    """
    解码器前向传播，重构输入
    """
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))  # 输出范围 [0,1]
```

#hd3("Discrete Latent Variables")

#hd4("Reinforce estimator")

在连续潜变量模型中，我们利用 reparameterization trick 对 @VAE-phi-dri 的第一项进行了求导。而对于离散潜变量模型，我们无法使用 reparameterization trick，因此我们需要使用 Reinforce estimator. 考虑：
$
  cal(L)(phi) = sum_Z q(Z;phi) f(Z) = bb(E)_(q(Z;phi)) f(Z)
$ <reinforce-estimator-eg>
对其求导:
$
  nabla_phi cal(L)(phi) &= sum_Z nabla_phi q(Z;phi) f(Z)\
  &= sum_Z q(Z;phi) f(Z) nabla_phi log q(Z;phi), space.quad "(log-derivative trick)"\
  &= 1/M sum_(m=1)^M f(z_m) nabla_phi log q(z_m;phi), space.quad "(Monte-Carlo estimation)"
$
因此 reinforce estimator 为：
$
  g(z_(1:M), phi) = 1/M sum_(m=1)^M f(z_m) nabla_phi log q(z_m;phi), space.quad z_m tilde q(z_m;phi)
$ <reinforce-estimator>
注意到，reinforce estimator 不仅允许离散潜变量，甚至允许不可微函数 $f(dot)$ 的存在。但同时存在以下缺点：
1. 方差较大，通过增大 $M$ 只会使std以 $1\/sqrt(M)$ 的速率减小
2. 其梯度方向由向量 $nabla_phi log q(z_m;phi)$ 决定，步长由标量 $f(z_m)$ 决定，因此可以认为梯度方向指向 $Z$ 概率增加的方向。
3. 不像 reparameterization trick, reinforce estimator 缺少 $nabla_phi f(Z)$ 的信息（例如VAE中decoder的梯度信息），而只使用了值；因此还对函数 $f(Z)$ 的移动 (e.g. $f(x)+c$) 敏感

#hd4("Gumbel-SoftMax trick")
考虑到 @reinforce-estimator 这么多的缺点，一个合理的想法是将离散潜变量转化为连续潜变量，然后使用 reparameterization trick，即：
$
  bb(E)_(q(Z;phi)) f(Z) = bb(E)_(q(tilde(Z)|phi)) f(tilde(Z)) = bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))
$
其中 $gamma$ 为噪声，$tilde(Z)(gamma, phi)$ 为 $Z$ 的连续估计，此时要确保 $f(dot)$ 连续可微

一种方式是使用 Gumbel-Max trick，假设 $z$ 为 $K$ 类离散变量，各自概率为 ${pi_i}_(i=1)^K, sum_i pi_i = 1$，则
$
  z &= arg min_i zeta_i/pi_i, space.quad zeta_i tilde "Exp"(1)\
  &= arg max_i [log pi_i - log zeta_i]\
  &= arg max_i [log pi_i + gamma_i], space.quad gamma_i tilde "Gumbel"(0,1)
$
唯一的问题在于 $arg max(dot)$ 不可导
#showybox()[
  proof:

  对于：
  $
    Y_i = zeta_i/pi_i = g(zeta_i), space.quad zeta_i tilde "Exp"(1)
  $
  计算其pdf：
  $
    f_(Y_i)(y) &= f_(zeta_i)(g^(-1)(zeta_i))abs(d/(d y) g^(-1)(zeta_i))\
    &= f_(zeta_i)(pi_i y) pi_i\
    &= pi_i e^(-pi_i y)
  $
  因此：
  $
    Y_i  tilde "Exp"(pi_i)
  $
  且：
  $
    P(Y_i = min {Y_j}_(j=1)^K) &= integral_0^(+infinity) P(Y_i=y) product_(i eq.not j) P(Y_j >= y) d y\
    &= integral_0^(+infinity) pi_i e^(-pi_i y) product_(i eq.not j) e^(- pi_j y) d y\
    &= integral_0^(+infinity) pi_i exp(-pi_i y + sum_(i eq.not j) -pi_j y)d y\
    &= integral_0^(+infinity) pi_i exp(-y) d y\
    &= pi_i
  $
]
因此使用 Gumbel-SoftMax trick，即使用带温度控制的 SoftMax 替代 argmax:
$
  "softmax"(x;tau)_j = (exp(x_j\/tau))/(sum_i exp(x_i\/tau))
$
其中温度 $tau$ 控制与 argmax 的相似性：
- 当 $tau=0$ 时，$"softmax"="argmax"$
- 当 $tau=infinity$ 时，$"softmax"="Uniform"$
因此：
$
  tilde(z)(gamma,pi) = "softmax"(log pi_i+gamma_i;tau), space.quad i = 1,dots,K
$
其中 $gamma_i tilde "Gumbel"(0,1)$，等价于
$
  gamma_i = -log(log u_i), space.quad u_i tilde "Uniform"(0,1)
$
此时，@reinforce-estimator-eg 可改写为：
$
  cal(L)(phi) = bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))
$
此时
$
  nabla_phi cal(L)(phi) &= nabla_phi bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))\
  &= nabla_phi f(tilde(Z)(gamma^*, phi)), space.quad gamma^* tilde "Gumbel"(0,1)
$
引入噪声 $gamma$ 的好处有：
- 提升泛化能力
- 正确的噪声类型（例如 $"Gumbel"(0,1)$）可以使 $tilde(z)$ 类似于 one-hot vector，提升训练集与测试集的相似度

对于温度 $tau$，通常使 $tau <= 1\/(K-1)$，并且使用grid search搜索。小 $tau$ 会导致高方差，但更能表示离散值；大 $tau$ 反之。

#hd4("Control Variates")

对于 @reinforce-estimator，另一个合理的想法是控制 reinforce estimator的方差，利用 variant control，通过减去已知量的估计误差来降低未知量的估计误差. 假设对于未知随机变量 $V(X)$，以及已知随机变量 $W(X)$ 与 其计算出的期望 $mu = bb(E)[W(X)]$，我们可以构造一个新的随机变量，与 $V(X)$ 具有相同期望，但方差更小：
$
  Z = V(X) - alpha(W(X) - mu)
$
此时方差为：
$
  "Var"(Z) = "Var"(V(X)) - 2 alpha "Cov"(V(X), W(X)) + alpha^2 "Var"(W(X))
$
注意到上式为关于 $alpha$ 的二次方程，最小值为：
$
  "Var"_min (Z) &= "Var"(V(X)) - ("Cov"^2(V(X), W(X)))/("Var"^2(W(X)))\
  alpha^* &=  "Cov"(V(X), W(X)) / "Var"(W(X))
$
除此以外，我们还可以利用多个已知变量 $W_i (X)$ 来降低方差（查看 @goodman2005montecarlo）

因此，假设 $mu = bb(E)_(q(Z;phi))[b(Z)]$，则 @reinforce-estimator 可以改写为：
$
  cal(L) &= bb(E)_(q(Z;phi))[f(Z) - b(Z) + mu]\
  &= bb(E)_(q(Z;phi))[f(Z) - b(Z)] + mu(phi)
$
利用 Monte-Carlo estimation，我们可以得到：
$
  cal(L) &= bb(E)_(q(z_(1:M)|phi))[1/M sum_(m=1)^M f(z_i)-b(z_i)] + mu(phi)
$
求导得到 reinforce estimator：
$
  g(z_(1:M), phi) = 1/M sum_(m=1)^M [f(z_m) - b(z_m)] nabla_phi log q(z_m;phi) +nabla_phi mu(phi)
$
其中 $z_i tilde q(Z;phi)$，$b(Z)$ 被称为 baseline, $b(Z) nabla_phi log q(Z;phi)$ 被称为 control variate. 接下来讨论 baseline 的选择：

1. 选择 baseline 为常数 $b(Z) = c$，有 $nabla_phi mu(phi)=0$，因此：
  $
    "Var"(g) &= "Var" (1/M sum_(m=1)^M [f(z_m) - b(z_m)] nabla_phi log q(z_m;phi))\
    &= 1/M^2 sum_(m=1)^M "Var"([f(z_m) - c] nabla_phi log q(z_m;phi)), space.quad "i.i.d. "z_m\
    &= 1/M^2 sum_(m=1)^M ["Var"(f dot nabla_phi)+c^2 "Var"(nabla_phi)-2 c "Cov" (f dot nabla_phi,nabla_phi)]
  $
  因此，$c$ 的最佳选择为：
  $
    c^* = "Cov"(f(Z) nabla_phi log q(Z;phi), nabla_phi log q(Z;phi))/"Var"(nabla_phi log q(Z;phi))
  $
  但如果有额外的观察项 (例如VAE中的 $log q(Z|X,phi)$)，则最佳 baseline 的选择应该与 $x$ 有关，因此不再适用于上式
2. NVIL: 针对上面的问题，@mnih2014neural 提出了使用 MSE 来估计 baseline：
  $
    b(X) = arg min_b bb(E)_p(X) bb(E)_(q(Z|X,phi))[f(Z) - b(X)]^2
  $
3. MuProp: @gu2015muprop 提出了将 $f(z)$ 的一阶泰勒展开作为 baseline：
  $
    b(Z) = f(mu) + nabla_Z f(mu)^tack.b dot (Z-mu)
  $
  对于 $mu$ 取 $mu = mu(phi)=bb(E)_(q(Z;phi))Z$，有
  $
    g(Z,phi) = (f(Z)-b(Z)) nabla_phi log q(Z;phi) + nabla_phi f(mu(phi))
  $
除此以外，还有其他不同的方法，如 @maddison2016concrete, @tucker2017rebar, etc.

#hd3("GAN")

#hd4("GAN")
GAN 的思想在于，使神经网络生成的分布 $q(x)$ 与真实分布 $p(x)$ 尽可能接近。接近程度由判别器决定：
$
  f(x;phi) = p(y=1|x, phi)
$ <gan1>
或者计算两个分布之间的举例：
$
  D_f (p||q)
$ <gan2>
通过判别器的指示，我们让 $q(x)$ 逐渐逼近 $p(x)$. 在这里我们需要对 $q(x)$ 这个分布进行采样，但$q(x)$ 的具体分布不知道，因此我们可以训练一个生成器：
$
  hat(x) = G(z;theta), space.quad z tilde p(z)
$
使得：
1. $z$ 容易从 $p(z)$ 中采样，例如 $cal(N)(0,I)$，
2. 采样的结果可以映射到 $q(x)$ 中，即 $hat(x) tilde q(x)$
如果使用 @gan1, 考虑到这是一个二分类问题，且判别器需要尽可能分开 $p(x)$ 与 $q(x)$，因此其 loss 为：
$
  cal(L)(phi, theta) = bb(E)_(p(x))[log f(x;phi)] + bb(E)_(p(z))[log(1-f(G(z;theta);phi))]
$ <gan-binary>
上式等价于使用 @gan2, 其loss为：
$
  cal(L)(phi, theta) = D_f (p(x)||q(x;theta))
$ <gan-df>
我们只需要
1. 更新判别器：
$
  phi = arg max_phi cal(L)(phi, theta)
$
2. 更新生成器（最小化 $p,q$ 距离）：
$
  theta^(t+1) = theta^t + nabla_theta cal(L)(phi, theta^t)
$
3. 重复1,2直至收敛
其中，$D_f$ 为 f-divergence，可以是任意的 divergence measure
$
  D_f (P||Q) = integral_cal(X) f (p(x)/q(x)) q(x) d x
$
其中 $f$ 指明了 divergence 的形式：
$
  f(t) = cases(
    t log t \, & space.quad "KL-divergence",
    - log t \, & space.quad "Reverse KL-divergence",
    1/2 abs(t - 1) \, & space.quad "Total variation",
  )
$

#hd4("Optimal transport")
根据 @gan-binary 和 @gan-df 我们知道：最小化generator的loss等价于最小化generator生成的分布与target分布的JS divergence。

但是用JS divergence来作为度量有个致命缺陷，就是在两个分布互不相交的情况下，两个分布的JS divergence永远都是常数 $log 2$，并且由于generator生成的分布和target分布的支撑集是在嵌在高维空间中的低维流形，所以他们重叠的部分的测度几乎为0。这样完全无法进行度量两个分布在不相交情况下的距离。计算梯度的时候也会出现0梯度的情况。@ymhuang2024

因此我们需要采用新的度量方式，这就是optimal transport. Optimal transport是一个线性规划问题，它的目标是找到两个分布之间的最小运输成本。假设从 $x$ 运输到 $y$ 具有一定的成本 $c$，一般来说，定义为：
$
  c(x,y) = ||x-y||_k^k
$
原始定义为：
$
  L = arg min_(Gamma) sum_(i,j)^(M,N) Gamma_(i,j) c(x_i,y_j)
$
optimal transport 具有概率版本，其optimal transport divergence 为
$
  T(P,Q) = inf_(Gamma in P(x tilde P, y tilde Q)) bb(E)_((x,y) tilde Gamma) c(x,y)
$
其中，$Gamma$ 为 $P$ 到 $Q$ 的一个联合分布，表示从 $P$ 到 $Q$ 的运输方案，满足联合分布的性质。转换为对偶问题 (dual problem) 为：
$
  T(P,Q) = sup_(phi,psi in L_1) (bb(E)_(p(x))phi(x) + bb(E)_(q(y))psi(y))
$
其中 $L_1:{phi(x)+psi(x)<=c(x,y)}$

因此，需要保证神经网络的函数是光滑的 Lipschitz 函数，即：
$
  ||f(x) - f(y)||_K <= L ||x-y||_K
$

对于 FFN，由仿射变换和逐点非线性组成的函数，这些非线性是光滑的 Lipschitz 函数（例如 sigmoid, tanh, elu, softplus 等）@arjovsky2017wasserstein. 对于线性矩阵计算函数，通过以下方式判断：
$
  ||A x_1 - A x_2||_2 <= L||x_1 - x_2||_2\
  sigma_max = sup_x (||A x||_2)/(||x||_2) <= L
$
其中 $sup_x (||A x||_2)/(||x||_2)$ 刚好等价于矩阵的 spectral norm, 即矩阵的最大奇异值。因此，为了使矩阵 $A$ 为满足 Lipschitz 函数，只需要使
$
  A := A / sigma_max
$
可以使用 power iteration 来计算 $sigma_max$ (#link("https://en.wikipedia.org/wiki/Power_iteration")[power iteration wiki])

#hd4("Gan Algorithm")
$
  &min_theta min_phi cal(L)(theta, phi) =\
  &min_theta min_phi bb(E)_(p(x)) log D(x;phi) + bb(E)_p(z) log(1-D(G(z;theta);phi))
$
1. $
     phi^(t+1) = phi^t + alpha nabla_phi cal(L)(theta^t, phi^t)
   $
2. $
      theta^(t+1) = theta^t - alpha nabla_theta cal(L)(theta^t, phi^(t+1))
   $
3. 重复1,2直至收敛

#hd3("Normalizing Flows")

与VAE类似，normalizing flows 假设潜变量 $z$，并可以根据数据 $x$ 得到潜变量，即：
$
  z = f_theta (x)
$
与VAE不同的是，normalizing flows 不使用 decoder 从 $z$ 获得 $x$，而是期望找到 $f^(-1)_theta (dot)$ 使得：
$
  x = f^(-1)_theta (z)
$
根据变量转换公式，我们有：
$
  p(x) &= p(z) abs(det (d f^(-1)_theta (z))/(d z))\
  &= p(f(x)) abs(det (d f_theta (x))/(d x)) 
$ <nf-change>
此时需要保证 $z$ 和 $x$ 的维度相同。同时，还需要保证 $f_theta (dot)$ 是可逆的，因此设计如下灵活且可解决的双射函数作为 coupling layer. 假设 $x in bb(R)^D$ 且 $d < D$，有：
$
  y_(1:d) &= x_(1:d)\
  y_(d+1:D) &= x_(d+1:D) dot.circle exp(s(x_(1:d))) + t(x_(1:d))
$ <nf-forward>
#figure(
  image("asset/NormalizingFlowsCP.png", width: 80%)
)
可以很容易得到其逆变换：
$
  x_(1:d) &= y_(1:d)\
  x_(d+1:D) &= (y_(d+1:D) - t(y_(1:d))) dot.circle exp(-s(y_(1:d)))
$
其中 $s,t: bb(R)^d arrow.bar bb(R)^(D-d)$. 这样，Jacobian 矩阵为：
$
  (partial y)/(partial x^tack.b) = mat(
    I_d, 0;
    (partial y_(d+1:D))/(partial x^tack.b_(1:d)), "diag"(exp[s(x_(1:d))])\
  )
$
代入 @nf-change，我们有：
$
  abs(det (d f_theta (x))/(d x)) = exp(sum_j s(x_(1:d))_j)
$
注意到，在 @nf-forward 中，$y_(1:d) = x_(1:d)$ 并没有经过变换，我们可以结合多个不同的 coupling layer 来解决这个问题，对于在一个 coupling layer 上未经变换的部分，我们让其在下一个 coupling layer 进行变换。即：
$
  f_theta (x) = f^N circle.small dots.h.c circle.small f^1 (x)
$
因此，根据MLE：
$
  log p_theta (x) &= log p(f_theta (x)) + log abs(det (partial f_theta (x))/ (partial x^tack.b))\
  &= log p(f_theta (x)) + sum_i^N log abs(det (partial f^i)/ (partial f^(i-1)))
$

#pagebreak()
#hd2("优化算法")

#hd3("Conjugate gradient algorithm")

#text(blue)[
  link: #link("https://en.wikipedia.org/wiki/Conjugate_gradient_method")[Conjugate gradient method-Wikipedia]
]

from: Rasmussen, C. (2006). Conjugate gradient algorithm, version 2006-09-08. available online.

共轭梯度算法（Conjugate Gradient Algorithm）是一种用于求解大规模线性系统 
$bold(A) bold(x)=bold(b)$ 的迭代方法，其中 $bold(A)$ 是一个对称正定矩阵。这个算法尤其适用于稀疏矩阵，因为它可以避免直接求解矩阵的逆，降低了计算复杂度。

共轭梯度算法的基本思想是通过迭代的方法逐步逼近线性方程的解，利用前一步的解信息来加速收敛。其步骤通常如下：

1. 初始化：选择初始点 $bold(x)_0$，计算残差 $bold(r)_0 = b - bold(A) bold(x)_0$, 设置初始搜索方向 $bold(p)_0 = bold(r)_0$
2. 迭代：
  - 计算步长
  $ 
  alpha_k = (bold(r)_k^tack.b bold(r)_k)/(bold(p)_k^tack.b bold(A) bold(p)_k)
  $
  - 更新解
  $
  bold(x)_(k+1) = bold(x)_k + alpha_k bold(p)_k
  $
  - 更新残差
  $
  bold(r)_(k+1) = bold(r)_k - alpha_k bold(A) bold(p)_k
  $
  - 检查收敛条件，若满足则停止迭代
  $
  bold(r)_(k+1) = bold(c)
  $
  - 否则，计算新的搜索方向
  $
  bold(p)_(k+1) = bold(r)_(k+1) + beta_k bold(p)_k, beta_k = (bold(r)_(k+1)^tack.b bold(r)_(k+1))/(bold(r)_k^tack.b bold(r)_k)
  $
3. 重复步骤2，直到满足收敛条件

#hd3("Natural Gradient")

#pagebreak()
#hd1("机器学习论文")

#pagebreak()
#hd2("聚类")

#hd3("Prototypical Contrastive Learning")

#link("https://arxiv.org/abs/2005.04966")

#link("https://github.com/salesforce/PCL")

无监督特征学习方法，将对比学习与聚类结合。

不仅学习用于实例区分的低级特征，更重要的是将聚类所发现的语义结构编码到学习到的嵌入空间中

#pagebreak()
#hd2("Latent Variable Model")

#hd3("Variational RNN")

#hd4("Motivation")

对于通常RNN来说，其训练为：
$
  h_t = f_theta (h_(t-1), x_t)
$
而推理过程则使用训练好的神经网络 $theta$，这导致RNN网络的变异性较低，即：输出的任何变化或波动都仅仅取决于RNN经过训练所学习到的 $theta$。

这意味着，RNN生成的输出受学习到的模式和规律影响，而不是受隐藏状态之间的直接转移过程的影响。换句话说，内部状态改变（即隐藏状态的变化）并不会直接引入新的变异性，所有变化都通过输出层的概率机制来实现。

因此，作者提出：在每一个时间步加入一个VAE机制，隐变量 $z_t$ 从 $cal(N)(theta_t)$ 采样，但 $theta_t$ 又由 $h_(t-1)$ 决定。这样，$z_t$ 的变化将直接影响到输出的变异性。

#hd4("数学公式")

在生成过程中：

对于每个时间步 $t$ 的VAE，其隐变量 $z_t$ 采样于：
$
  z_t tilde cal(N)(mu_(0,t), "diag"(sigma_(0,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $h_(t-1)$ 生成：
$
  mu_(0,t), sigma_(0,t) = phi_tau^("prior") (h_(t-1))
$
$x_t$ 的生成则为：
$
  x_t|z_t tilde cal(N)(mu_(x,t), "diag"(sigma_(x,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $z_t$ 和 $h_(t-1)$ 生成：
$
  mu_(x,t), sigma_(x,t) = phi_tau^("dec") (phi_tau^z (z_t), h_(t-1))
$
对于RNN：
$
  h_t = f_theta (phi_tau^x (x_t), phi_tau^z (z_t), h_(t-1))\
  p(x_(<=T), z_(<=T)) = product_(t=1)^T p(x_t|z_(<=t),x_(<t))p(z_t|x_(<t),z_(<t))
$
在推理过程中：

隐变量的后验采样于：
$
  z_t|x_t tilde cal(N)(mu_(z,t), "diag"(sigma_(z,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $x_t$ 和 $h_(t-1)$ 生成：
$
  mu_(z,t), sigma_(z,t) = phi_tau^("enc") (phi_tau^x (x_t), h_(t-1))
$
进而：
$
  q(z_(<=T)|x_(<=T)) = product_(t=1)^T q(z_t|x_(<=t),z_(<t))
$

我们的目标函数为：
$
  bb(E)_(q(z<=T|x<=T))[sum_(t=1)^T (-"KL"(q(z_t|x_(<=t),z_(<t))||p(z_t|x_(<t),z_(<t)))& \
  + log p(x_t|z_(<=t),x_(<t)))]&
$

#pagebreak()
#hd2("长序列算法")

#hd3("记忆力")
一种记忆力的评估方法是评估模型，在第 $n$ 步时可以利用多远的信息计算输出@poli2023hyena. 对于输入序列 $u(t)$，输出序列 $y(t)$，统计以下输出中不为零的数量：
$
(partial y(t)) / (partial u(t-n)), space.quad n=0,1,dots,t
          $
以 SSM 为例，有：
$
  (partial y(t)) / (partial u(t-n)) = C A^n B
$
除此以外，@chen2024hope 提出了以下方式：
1. 生成序列：序列第一个位置为`[bos]`，后续的每一个位置随机来自于字典中的token
2. 计算 pre-softmax logits：在计算注意力时，会计算query和所有key的相似性(点积)，因此这里相当于计算了位置 $i$ 和所有位置的相似性
$
  text("logits")_(i,j) = q_i^tack.b k_j
$
3. 对 pre-softmax logit归一化：在所有注意力头上平均

#hd3("Hyena Hierarchy")

论文：@poli2023hyena

#hd4("结合SSM的卷积")

以普通的卷积为例，假设 $S in bb(R)^(L times L)$ 为filter，$U  in bb(R)^(L times C)$ 为输入，$y$ 为输出，则有：
$
  y_t = (h * u)_t = sum_(n=0)^(L-1) h_(t-n) u_n
$
这里 $h in bb(R)^(L)$，为了便于SSM的加入，令filter $S$ 为 Toeplitz 矩阵，即每一个对角线上元素相同：
$
  S_(i,j) = S_(i+1, j+1)\
  S = mat(
    h_0 , h_(-1) , dots, h_(-L+1) ;
    h_1 , h_0 , dots, h_(-L+2) ;
    dots , dots , dots , dots ;
    h_(L-1) , h_(L-2) , dots , h_0;
  )
$
根据 SSM @SSM-soultion，我们可以得到filter:
$
  h_t = cases(
    0 space.quad & t < 0,
    C A^t B + D delta_t space.quad & t >= 0
  )
$

#hd4("FFT Conv")
卷积操作可以利用FFT优化运行速度。卷积操作的空间复杂度为$O(L^2)$，而利用FFT可以将其降低到$O(L log L)$。具体运用了傅里叶变换的卷积性质 @FFT-conv.
考虑filter为Toeplitz 矩阵的特殊情况，循环矩阵：
$
  S_h = mat(
    h_0 , h_1 , dots , h_(L-1) ;
    h_(L-1) , h_0 , dots , h_(L-2) ;
    dots , dots , dots , dots ;
    h_1 , h_2 , dots , h_0;
  )
$
利用FFT，我们可以将循环矩阵对角化：
$
  S_h = W^(-1) D_H W
$
其中 $D_H$ 为对角矩阵，$W$ 为DFT矩阵。因此，卷积操作可以写为：
$
  y &= S_h u\
  &= W^(-1) D_H W u\
  &= text("iFFT")(D_h text("FFT")(u))
$
其中，对角矩阵 $D_H$ 的对角元素为循环矩阵的特征值，可以通过以下方式计算：
$
  p(lambda) = det(S_h - lambda I) = 0
$

#hd4("Order-N hyena operator")

假设 $(v, x^1, dots, x^N)_t$ 为输入 $u$ 的投影，同时filters $(h^1, dots, h^N)_t$为可学习的，hyena operator 执行以下循环操作：
$
  z_t^1 &= v\
  z_t^(n+1) &= x_t^n (h^n * z^n)_t space.quad n=1\,dots\,N,\
  y_t &= z_t^(N+1)
$
这一循环操作的卷积由FFT完成，空间复杂度是 $O(N L log L)$.同时注意到，每一步循环操作包括：
1. 对时域进行卷积 ($h^n_t * z^n_t$)，
2. 对频域进行卷积 (时域element-wise product, $x_t^n (h^n * z^n)_t$). 
作者认为：#margin-note("时域上的卷积被认为提高了记忆的长度，而频域上的卷积被认为提高了频率的精细度。")[卷积本质可以理解为对信号的加权和操作，在时域中反映为对历史信息（过往信号）的累积，而在频域中则反映为对信号的频率成分的加权调整] 

#hd4("self-attention operatior")
通常来说，self-attention 只包括3个部分：query, key, value:
$
  y &= text("self-attetnion")(u)\
  &= text("softmax")(1/sqrt(D) u M_q M_k^tack.b u^tack.b) u M_v\
  &= A(q, k) v
$
其中，$M_q, M_k, M_v in bb(R)^(D times D)$ 为输入 $u in bb(R)^(L times D)$ 可学习的投影。

在hyena的attention操作中，我们可以将其拓展为更多的部分。

首先，对于注意力矩阵，使用替代的注意力矩阵 $A(q, k)$，其计算方式为：
$
  A(q,k) = D_q S_(epsilon) D_k S_(phi)
$
其中，$D_q, D_k in bb(R)^(L times L)$ 分别为 $q,k$ 的对角矩阵。$S_epsilon, S_phi$ 为 Toeplitz 矩阵，其参数由 SSM 决定。

因此，3个部分的self-attention操作可以写为：
$
  H_3(q,k,v) = A(q, k) v = D_q S_(epsilon) D_k S_(phi) v
$
拓展到多个部分，我们令 $D_x^n = text("diag")(x^n) in bb(R)^(L times L)$，$S_h^n$ 为 Toeplitz 矩阵，来源于 filter $h^n$，则有：
$
  y = H(u)v = D_x^N S_h^N D_x^(N-1) S_h^(N-1) dots D_x^1 S_h^1 v
$

#hd4("Hyena filter")
Hyena filter采用FFN更新：
$
  h_t = text("Window")(t) dot (text("FFN") circle.tiny text("Positional Encoding"))(t)
$

#hd4("算法")
#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Projection])[
    + *Require* $u in bb(R)^(L times D)$
    + 在dim $L$ 上：$z = text("Linear"(u)), space.quad text("Linear") :bb(R)^D arrow bb(R)^((N+1)D)$
    + 在dim $D$ 上：$z = text("DepthwiseConv1d")(h,z)$
    + reshape：将 $z$ 拆分为 $v, x^1, dots, x^N in bb(R)^(D times L)$
    + *Return* $(v, x^1, dots, x^N)$
  ]
)

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Hyena Filter])[
    + *Require* 序列长度 $L$，Positional Embedding Dim $D_e$\
    + $t = text("PositionalEncoding"(L)), space.quad t in bb(R)^(L times D_e)$
    + 在 dim $L$ 上：$h = text("FFN"(t)), space.quad text("FFN"):bb(R)^(D_e) arrow bb(R)^(N D_e)$
    + reshape: $h in bb(R)^(N times D times L)$
    + $h = h dot text("Window")(t), space.quad h in bb(R)^(N times D times L)$
    + split: $h = (h^1, dots, h^N)$
    + *Return* $(h^1, dots, h^N)$
  ]
)

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [Forward])[
    + *Require* $u in bb(R)^(L times D)$，order N operator，PED $D_e$
    + $x^1, dots, x^N, v = text("Projection"(u))$
    + $h^1, dots, h^N = text("HyenaFilter"(L, D_e))$
    + *for* $n = 1, dots, N$ *do*
      + 在dim$D$上：$v_t arrow.l x_t^n dot text("FFTConv")(h^n, v)_t$
    + *end for*
    + *Return* $v$
  ]
)


#pagebreak()
#hd2("其他算法")

#hd3("Noise-contrastive estimation")

#link("https://proceedings.mlr.press/v9/gutmann10a")

#hd4("数学推导")

#h(2em) 假设观察到数据的分布（概率密度函数pdf）为 $p_d (dot)$，考虑pdf由参数 $bold(alpha)$ 决定，因此可以认为pdf属于参数家族 ${p_m (dot;bold(alpha))}_bold(alpha)$.其中 $bold(alpha)$ 为参数向量，且存在某个 $bold(alpha)^*$ 使得 $p_d (dot) = p_m (dot;bold(alpha)^*)$。

这里的问题是，如何在观察数据的基础上，通过最大化目标函数去估计参数 $bold(alpha)^*$. 我们知道的是，不管参数估计结果如何，都一定满足：

$
integral p_m (bold(u);hat(bold(alpha))) d bold(u) = 1
$ <NCE1>

为了绕过这个限制，并且保证为一，我们可以考虑#margin-note[使用归一化避免积分限制]使用归一化的方法，即：

$
p_m (dot;bold(alpha)) = (p_m^0 (dot;bold(alpha)))/(Z(alpha)) d bold(u), Z(bold(alpha)) = integral p_m^0 (bold(u);bold(alpha)) d bold(u)
$

这里，$p_m^0 (dot;bold(alpha))$ 不一定要满足 @NCE1 的限制，可以是一个形式类似pdf的函数。但是，#margin-note[要尽量避免积分，积分计算困难]归一化系数 $Z(bold(alpha))$ 包括积分，通常是难以计算的。为了避免直接计算 $Z(bold(alpha))$ 的积分，我们可以将其考虑为一个新参数，通过优化的方法估计 $Z(bold(alpha))$.即：

$
p_m (dot;bold(theta)) = (p_m^0 (dot;bold(alpha)))/(C),theta = {bold(alpha), c}, c = Z(bold(alpha))
$

问题出现在使用最大似然估计MLE时：

$
hat(bold(theta))&= arg max_(bold(theta)) sum_(bold(u)) log p_m (bold(u);bold(theta))\
&= arg max_(bold(theta)) sum_(bold(u)) log (p_m^0 (bold(u);bold(alpha)))/(c)\
&arrow.double c arrow 0
$

会导致 $c$ 趋近于无穷小，这样的结果显然无效。

为了解决这样的问题，作者提出了一种新的估计方法，即Noise-contrastive estimation。其基本思想是，引入噪声分布 $p_n (dot)$，通过比较 $p_m (dot;bold(alpha))$ 和 $p_n (dot)$ 的相似性来估计参数 $bold(alpha)$.

从本质上来说，区分噪声与观测数据属于二分类问题。而二分类问题我们可以使用logistic回归解决，即：Logistic回归通过使用sigmoid函数将线性组合的输入映射到0到1之间的输出，以估计某个事件发生的概率。其模型形式为：

$
p(C=1|bold(u)) &= sigma(bold(u)^tack.b bold(theta)) = 1/(1+exp(-bold(u)^tack.b bold(theta)))\
p(C=0|bold(u)) &= 1 - p(C=1|bold(u)) = 1 - sigma(bold(u)^tack.b bold(theta))
$

其中，$C=1$ 代表为观测数据，$C=0$ 代表为噪声数据。

当进行回归时，我们的目标函数为：

$
L(bold(theta)) &= sum_(t=1)^T log p(C_t|bold(u)_t;bold(theta))\
&= sum_(t=1)^T C_t log p(C=1|bold(u)_t) +(1-C_t) log p(C=0|bold(u)_t)
$ <NCE2>

对于观测的数据：$X={bold(x)_1,bold(x)_2,dots,bold(x)_T}$，我们引入相同大小的噪声数据 $Y={bold(y)_1,bold(y)_2,dots,bold(y)_T}$. 根据logistic回归，我们令 $U=X union Y = {bold(u)_1, bold(u)_2, dots, bold(u)_(2T)}$.对于每一个 $bold(u)_t arrow.bar C_t = {0,1}$，当 $bold(u)_t$ 为观测数据时，$C_t=1$；当 $bold(u)_t$ 为噪声数据时，$C_t=0$.

#margin-note("直接类比logistic回归的结果即可") $p_m (bold(u);bold(theta))$，即通过优化获得参数 $bold(theta)$.因此，现在需要做的是获得目标函数 $L(bold(theta))$，且根据logistic回归的 @NCE2，我们需要求出 $p(C=1|bold(u);bold(theta))$

我们知道，观测数据的pdf为等价于 $C_t=1$ 时对应的条件分布；噪声数据的pdf为等价于 $C_t=0$ 时对应的条件分布：
$
p_m (bold(u);bold(theta)) = P(bold(u)|C=1;bold(theta)), p_n (bold(u);bold(theta)) = P(bold(u)|C=0;bold(theta))
$

我们同时知道，$C$ 的分布为bernoulli分布，有 $P(C=1)=P(C=0)=1\/2$.因此，我们可以使用贝叶斯来计算：
$
P(C=1|bold(u);bold(theta)) &= (P(C=1 union bold(u);bold(theta)))/(P(bold(u);bold(theta)))\
&= (P(bold(u)|C=1;bold(theta)) dot P(C=1;bold(theta))) / (P(bold(u)|C=1;bold(theta)) dot P(C=1;bold(theta)) + P(bold(u)|C=0;bold(theta)) dot P(C=0;bold(theta)) )\
&= (p_m (bold(u);bold(theta)))/(p_m (bold(u);bold(theta)) + p_n (bold(u);bold(theta)))\
&=h(bold(u);bold(theta))\
P(C=0|bold(u);bold(theta)) &= 1 - h(bold(u);bold(theta))
$
代入 @NCE2，我们可以得到
$
L(bold(theta)) = sum_(t) C_t log h(bold(u)_t;bold(theta)) +(1-C_t) log(1-h(bold(u)_t;bold(theta)))
$
考虑到 $bold(u)_t arrow.bar C_t = {0,1}$，我们可以化简上述公式为：
$
L(bold(theta)) = sum_t log h(bold(x)_t;bold(theta)) + log(1-h(bold(y)_t;bold(theta)))
$
我们有 $2T$ 个样本，#margin-note[
  解决：
  1. 损失值不可比\
  2. 梯度下降快\
  3. 数值不稳定
  ]因此对目标函数平均化：
$
L(bold(theta)) = 1/(2T)sum_t log h(bold(x)_t;bold(theta)) + log(1-h(bold(y)_t;bold(theta)))
$

#pagebreak()
#bibliography("ref.bib", style: "ieee", title: "参考文献")