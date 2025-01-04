---
title: Zero or One Inflated beta distribution for sleep staging probability ensemble calibration
date: 2024-09-01 00:00:00 +0000
categories: [projects]
tags: [ensemble modeling,sleep staging porbability,zero or one inflated beta distribution,calibration,Multi-Layer perceptron]
image: 
    path: '/assets/project_sleep_stage_calibration/figures/MLP_CS.png'
    alt: Figure (1) - Multi-Layer Perceptron head architecture performing a Zero or One Inflated beta calibration
---
---
## Information
<div style="text-align: justify">
Presentation of the research project on probability ensemble calibration using a Zero or One Inflated beta distribution. Full research paper in progress. This research is supported by the Sleep Revolution project at the University of Reykjavik. Python code implementation is available.
</div>
---


---
## Abstract
<div style="text-align: justify">


</div>

---


---
## Calibration problem
<script async src="/assets/article_ensemble_forecast/js/mathjax.js"></script>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Let's define each ensemble set as $X = \{X_1=(X_{1_1},\cdots,X_{1_M}),\cdots,X_n = (X_{n_1},\cdots,X_{n_M})\}$, where $X_{i_m}$ represents the $m^{th}$ ensemble member, which is the sleep stage probability of the $i^{th}$ stage epoch for $i\in\{1,\cdots,n\}$ and $m\in\{1,\cdots,M\}$. $n$ is the number of staging epochs and $M$ is the size of the ensemble. The observed sleep staging probabilities provided by the sleep technologist are set to $y = \{y_1,\cdots,y_n\}$. The main idea of this section is then to propose a calibration model with a specific objective function. This function is designed for a chosen distribution that models the sleep technologist probability that the ensemble prediction needs to map. Finally, this distribution is integrated in the CRPS to derive an expression to infer the distribution parameters based on the ensemble $X$ and the observation $y$.


The CRPS definition's for an ensemble of random variable $X=\{X_1,\dots,X_M\}$ for any cumulative distribution function (CDF) $\mathbf{F}$ and deterministic observation $y$ is:
\[
\begin{equation}
    CRPS(\mathbf{F},y) = \int^{+\infty}_{-\infty}|\mathbf{F}(x)-1_{[y,+\infty)}(x)|^2dx
\label{eq:CRPS}
\end{equation}
\]
</div>

### Distribution choice: Zero or One Inflated beta (bic)

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Let define the random variable (RV) $X$ following a Zero or One inflated distribution with probability density function (PDF) and CDF with parameter $\alpha$, $a$ and $b$ and a chosen $c\in\{0,1\}$ :
\[
\begin{align}
bi_c(x;\alpha,a,b) &= 
\begin{cases}
\alpha & if \ x=c\\
(1-\alpha)f_{Beta}(x;a,b) & if \ x \in (0,1)
\end{cases}\\
\mathbf{BI}_c(x;\alpha,a,b)  &= \alpha\textbf{1}_{[c,1]}(x)+(1-\alpha)\mathbf{F}_{Beta}(x;a,b)
\label{eq:ZOI}
\end{align}
\]
</div>

```python
from scipy.stats import beta
import scipy.integrate as integrate
from scipy import special
import numpy as np

# Define the probability density function of the zero or one inflated beta distribution
def bic(x,c,alpha,a,b):
    if c==x:
        return alpha
    else:
        if x>=1:
            x=0.99
        if x<=0:
            x=0.01
        return (1-alpha)*beta.pdf(x, a, b)

# Define the cumulative distribution function of the zero or one inflated beta distribution
def BIc( x,c,alpha,a,b):
    if x>=1:
        x=0.99
    if x<=0:
        x=0.01
    return (1 - alpha) * beta.cdf(x, a, b) + alpha *((x >= c) &(x <= 1))

# Define the quantile function of the zero or one inflated beta distribution
def BIcq(p,c,alpha,a,b):
    if p <= alpha:
        return c
    else:
        return beta.rvs(a, b)

# Define the random variable generator function of the zero or one inflated beta distribution
def rvs(c,alpha,a,b, size=1,seed=0):
    np.random.seed(seed)
    p = np.random.rand(size)
    return np.array([BIcq(pp,c,alpha,a,b) for pp in p])
```

### CRPS of bic

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

\[
\begin{equation}
CRPS(\mathbf{BI}_c,y) = \int^{+\infty}_{-\infty}|\mathbf{BI}_c(z;\alpha,a,b)-1_{\{y \le z\}}|^2dz
\label{eq:CRPSTH}
\end{equation}
\]

\[
\begin{equation}
CRPS(\alpha,a,b;y) = \begin{cases}
y(2\mathbf{BI}_c(y;\alpha,a,b)-1)+(1-\alpha)g(y;a,b,c) & if \ c=0\\
\alpha^2+y(2(1-\alpha)F_{Beta}(y;a,b)-1)+(1-\alpha)g(y;a,b,c) & if \ c=1
\end{cases}
\label{eq:CRPSBeta}
\end{equation}
\]

\[
\begin{equation}
\begin{split}
g(a,b,c;y) &=\frac{a}{a+b}(1-2F_{Beta}(y;a+1,b)+h_{\alpha}(c))-2(1-\alpha)\frac{B(2a+1,2b)}{aB(a,b)^2}\\
h_{\alpha}(c) &= \begin{cases} -\alpha & if \ c=0\\
\alpha & if \ c=1
\end{cases}\\
\end{split}
\label{eq:utils}
\end{equation}
\]

</div>

```python
def crps_operation(y,c,alpha,a,b,intg=False):
    if intg:
        return integrate.quad(lambda x: (BIc(x,c,alpha,a,b)-1)**2 if x>=y else (BIc(x,c,alpha,a,b))**2, 0, 1)[0]
    else:
        if y>=1:
            y=0.9999
        if y<=0:
            y=0.0001
        if c==0:
            part1 = y*(2*BIc(y,c,alpha,a,b)-1)
            part21= a/(a+b)*(1-2*beta.cdf(y,a+1,b)-alpha)
            part22=-(1-alpha)*(2*special.beta(2*a+1,2*b)/(a*special.beta(a,b)**2))
            part2 = (1-alpha)*(part21+part22)
        if c==1:
            part1 = y*(2*(1-alpha)*beta.cdf(y,a,b)-1)+alpha**2
            part21= a/(a+b)*(1-2*beta.cdf(y,a+1,b)+alpha)
            part22=-(1-alpha)*(2*special.beta(2*a+1,2*b)/(a*special.beta(a,b)**2))
            part2 = (1-alpha)*(part21+part22)
        return part1+part2

def crps(y,c,alpha,a,b,intg=False):
    if isinstance(y, (int, float)):
        return crps_operation(y,c,alpha,a,b,intg)
    else:
        crpsv = np.vectorize(crps_operation,excluded=["c","alpha","a","b","intg"])
        return crpsv(y=y,c=c,alpha=alpha,a=a,b=b,intg=intg)
```


---

---
## Neural Network architectures
<div style="text-align: justify">

</div>

### Local MLP

<div style="text-align: justify">

<figure>
  <img
  src="/assets/project_sleep_stage_calibration/figures/MLP_CSS.png"
  >
  <em>Figure (2) - Double CNN architecture.</em>
</figure>

</div>

### CRPS keras ops implementation

```python
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

def crps_loss(y_true, y_pred):
    y = tf.keras.ops.reshape(y_true, [-1])
    c = y_pred[:, 0]
    alpha = y_pred[:, 1]
    a = y_pred[:, 2]
    b = y_pred[:, 3]
    crps_loss = 0
    if tf.test.gpu_device_name() == '/device:GPU:0':
        try:
            alpha = tf.keras.ops.clip(alpha, 0.001, 0.9999)
            c = tf.keras.ops.round(c)
            y = tf.keras.ops.clip(y, 0.001, 0.9999)
            beta_dist = tfp.distributions.Beta(a + 1, b)
            params_2a12b = tf.concat([tf.expand_dims(2 * a + 1, axis=-1), tf.expand_dims(2 * b, axis=-1)], axis=-1)
            params_ab = tf.concat([tf.expand_dims(a, axis=-1), tf.expand_dims(b, axis=-1)], axis=-1)
            params_2a12b = tf.keras.ops.map(lambda x: tf.math.lbeta([x[0], x[1]]), params_2a12b)
            params_ab = tf.keras.ops.map(lambda x: tf.math.lbeta([x[0], x[1]]), params_ab)
            part221 = (2 * tf.exp(params_2a12b))
            part222 = (a * tf.exp(params_ab) ** 2)
            part22 = -(1 - alpha) * (part221 / part222)
            part1 = tf.zeros_like(y)
            part21 = tf.zeros_like(y)
            mask_c0 = tf.equal(c, 0)
            mask_c1 = tf.equal(c, 1)
            part10 = y * (2 * self.BIc_ops(y, c, alpha, a, b) - 1)
            part11 = y * (2 * (1 - alpha) * tfp.distributions.Beta(a, b).cdf(y) - 1) + alpha ** 2
            part1 = tf.where(mask_c0, part10, part1)
            part1 = tf.where(mask_c1, part11, part1)
            part210 = a / (a + b) * (1 - 2 * beta_dist.cdf(y) - alpha)
            part211 = a / (a + b) * (1 - 2 * beta_dist.cdf(y) + alpha)
            part21 = tf.where(mask_c0, part210, part21)
            part21 = tf.where(mask_c1, part211, part21)
            part2 = (1 - alpha) * (part21 + part22)
            crps_loss = part1 + part2
        except Exception as e:
            print(y, c, alpha, a, b, crps_loss, tf.reduce_mean(crps_loss))
            raise e

        crps_loss = tf.reduce_mean(crps_loss)
        return crps_loss
    else:
        c = tf.round(c)
        alpha = tf.clip_by_value(alpha, 0.001, 0.9999)
        y = tf.clip_by_value(y, 0.001, 0.9999)
    return tf.reduce_mean(crps_loss)
```

### Sleep staging probability estimation from Zero or One inflated beta
<div style="text-align: justify">

</div>

### Global MLP

<div style="text-align: justify">

<figure>
  <img
  src="/assets/project_sleep_stage_calibration/figures/fMLP_CC.png"
  >
  <em>Figure (3) - Double CNN architecture.</em>
</figure>

</div>
---

---
## Results

<div style="text-align: justify">
<!-- 
<figure>
  <img
  src="/assets/project_sleep_stage_calibration/figures/fMLP_CC.png"
  >
  <em>Figure (3) - Double CNN architecture.</em>
</figure> -->

</div>
---