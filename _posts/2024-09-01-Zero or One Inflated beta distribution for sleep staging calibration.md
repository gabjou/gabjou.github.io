---
title: Zero or One Inflated beta distribution for sleep staging probability ensemble calibration using Multi-Layer Perceptron model
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
Research project on probability ensemble calibration using a Zero or One Inflated beta distribution. Full research paper in progress. This research is supported by the Sleep Revolution project at the University of Reykjavik. Python code implementation is available. This research was inspired by <a href="https://journals.ametsoc.org/view/journals/mwre/146/11/mwr-d-18-0187.1.xml?tab_body=fulltext-display">Neural Networks for Postprocessing Ensemble Weather Forecasts</a>, <a href="https://d1wqtxts1xzle7.cloudfront.net/99043773/2207.07753-libre.pdf?1677178065=&response-content-disposition=inline%3B+filename%3DDo_Not_Sleep_on_Linear_Models_Simple_and.pdf&Expires=1736069818&Signature=GTJaHfqjMvyrBoQ96BPF5-34hCrVwILROh9Fm5up5TwnEdV~oqbddJc7AqZ-U~Rrn3W3qt1KImWcxvPiTBvQFciISGbYxvdHCnZbBMj3f0op~3pMeg98f8MaKrjQ50j8x5py04ytQcAdRoaxccY1qpXzkdTOCkRt~pAvszELH5AvKzbK9hvXebtmnJvRzMej12pmkGC0UEqtpI9~SoqD7AvHlM~EEmYw4os~o1DJ6c2YZKmHAgXg2trSX9w90mgSW3Z8TERevkHoBV9AJsFvVi8GFMjWLy71z7J6sP2uD-5ongT85450wFnsm1AE0WVdbipDaWVmFFy1Uz0N9LfD2w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA">Do not sleep on traditional machine learning</a> and <a href="https://www.sciencedirect.com/science/article/abs/pii/S0167947311003628">A general class of zero-or-one inflated beta regression models</a>. This work would be followed by a publication in the next months.


</div>
---


---
## Abstract
<div style="text-align: justify">

For the diagnosis of most sleep disorders, an objective sleep study called polysomnography (PSG) is required. This study includes electroencephalography (EEG), electrooculography (EOG), etc. for the assessment of sleep stages, which are scored manually. The expert provides sets of sleep stage annotation for each 30-second period of the night (called epoch) based on the American Academy of Sleep Medicine (AASM) standards and EEG, EOG and chin EMG signals recorded during the PSG. According to the AASM standards, sleep is divided into 5 stages, the wake stage (wake), rapid eye movement (rem), and three non-REM stages (n1, n2, and n3) that describe the depth of the sleep. Despite the experience and expertise of scorers, it is well known that the scoring task has a not fully understood amount of uncertainty. Datasets consisting of an ensemble of independent scorers providing multiple sets of sleep staging are needed to estimate inter-scorer agreement as well as the uncertainty associated with manual scoring. From the ensemble of scorers, uncertainty is highlighted by the sleep stages where scorers disagree for a certain epoch.
<br>
<br>
Automatic sleep staging prediction is a problem studied in the field of sleep research. Most traditional classification models used for this task have been inferred using inputs from a single EEG signal. However, new cohorts provide access to records that easily contain 6 EEG signals and 2 EOG signals. The ensemble of these signals is used to train independent classification models based on a validated state-of-the-art architecture in the sleep staging field. The resulting ensemble of predicted sleep staging probabilities, which differ from the observed probabilities provided by sleep technologists, presents a calibration problem. With an accurate choice of distribution, a calibration model is proposed using a specific objective function based on the selected distribution.
<br>
<br>
To address this, we introduce an ensemble of sleep staging probabilities and employ a calibration model as a post-processing step to deliver better sleep staging probabilities. The calibration of sleep staging probabilities is performed using the zero or one inflated beta distribution to deduce an objective function for the model inference. This distribution offers a superior model for incorporating samples with true zero or one probabilities provided by human experts, avoiding issues at the extremes of the beta distribution range. The ensemble modeling and calibration postprocessing method aim to improve the reliability and interpretability of sleep staging outcomes.  We present a novel postprocessing method with a specific objective function designed for such task. 
<br>
<br>
Effective results were obtained on a real case of uncertainty analysis in the sleep staging domain. A series of actual sleep stages scored by an ensemble of 10 independent scorers for a dataset of 50 participants was used. Sleep stage probabilities derived from calibrated ensemble has shown accuracy improvement compared to direct classification method.

</div>

---

---
## Ensemble of sleep staging probabilities
<div style="text-align: justify">
Three different kind of ensemble sleep staging probabilities have been studied by extracting features from EEG and EOG combinaitions. The first ensemble is generated by a classification model independently trained on features extracted from different EEG and EOG signals. Each member of this ensemble is predicted from a dedicated signal, and the resulting ensemble is named 1S. The second approach involves training a model on combined features from 1 EEG and 1 EOG signal, with this ensemble named 1EEG_1EOG. Finally, the last ensemble, named 2EEG_2EOG, consists of predictions from a model trained on combined features from 2 EEG and 2 EOG signals. The gradient boosting method XGBoost was chosen as the classification model to provide sleep stage probabilities for each ensemble features extracted using the same signals preprocessing pipeline from <a href="https://d1wqtxts1xzle7.cloudfront.net/99043773/2207.07753-libre.pdf?1677178065=&response-content-disposition=inline%3B+filename%3DDo_Not_Sleep_on_Linear_Models_Simple_and.pdf&Expires=1736069818&Signature=GTJaHfqjMvyrBoQ96BPF5-34hCrVwILROh9Fm5up5TwnEdV~oqbddJc7AqZ-U~Rrn3W3qt1KImWcxvPiTBvQFciISGbYxvdHCnZbBMj3f0op~3pMeg98f8MaKrjQ50j8x5py04ytQcAdRoaxccY1qpXzkdTOCkRt~pAvszELH5AvKzbK9hvXebtmnJvRzMej12pmkGC0UEqtpI9~SoqD7AvHlM~EEmYw4os~o1DJ6c2YZKmHAgXg2trSX9w90mgSW3Z8TERevkHoBV9AJsFvVi8GFMjWLy71z7J6sP2uD-5ongT85450wFnsm1AE0WVdbipDaWVmFFy1Uz0N9LfD2w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA">Do not sleep on traditional machine learning</a>. The XGBoost classifier offers the best trade-off between training time efficiency and model prediction performance, with robustness against bias and overfitting.

Each of the three ensembles is exchangeable, meaning that the joint distribution of the ensemble is invariant to any permutation order. Each ensemble member represents a prediction of sleep stage probabilities derived from different combinations of signals that are not independent of each other and follow the same distribution.

In this blog, we directly consider that we have access to each trained XGBoost model predicting the ensemble of sleep staging probabilities for the 10 scorers dataset.
</div>
---

---
## Calibration problem
<script async src="/assets/article_ensemble_forecast/js/mathjax.js"></script>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
The calibration problem arises when the deterministic observation of an unknown system diverges from the distribution of the ensemble of forecasts produced by the model learning from that system. Here, the system is considered to be the sleep staging rules. The true observation is the sleep staging probabilities provided by the ensemble of sleep technologist and set to $y = \{y_1,\cdots,y_n\}$. $n$ is the number of staging epochs. The ensemble of predicted sleep staging probabilities is defined as $X = \{X_1=(X_{1_1},\cdots,X_{1_M}),\cdots,X_n = (X_{n_1},\cdots,X_{n_M})\}$. $X_{i_m}$ represents the $m^{th}$ ensemble member, which is the sleep stage probability of the $i^{th}$ stage epoch for $i\in\{1,\cdots,n\}$ and $m\in\{1,\cdots,M\}$. $M$ is the size of the ensemble.
<br>
<br>
The main idea of this section is to propose a calibration model with a specific objective function. This function is tailored to a chosen distribution that models the sleep technologist's probabilities, which the ensemble predictions need to align with. Finally, this distribution is integrated into the Continuous Ranked Probability Score (CRPS) to derive an expression for inferring the distribution parameters based on the ensemble $X$  and the observation $y$. The CRPS definition's for an ensemble of random variable $X=\{X_1,\dots,X_M\}$ for any cumulative distribution function (CDF) $\mathbf{F}$ and deterministic observation $y$ is:
\[
\begin{equation}
    CRPS(\mathbf{F},y) = \int^{+\infty}_{-\infty}|\mathbf{F}(x)-1_{[y,+\infty)}(x)|^2dx
\label{eq:CRPS}
\end{equation}
\]

To effectively calibrate the ensemble of sleep staging probabilities, it is crucial to select an appropriate distribution that accurately models the probabilities provided by sleep technologists. At each epoch, the sleep staging probability is a vector of five values ranging between 0 and 1, with their sum equal to one. This makes the calibration problem multi-dimensional and challenging to resolve. Following a process similar to multinomial regression, the calibration can be performed independently for each dimension of the probability vector. In this case, each element of the probability vector can be viewed as the probability of a binary event—either having a specific sleep stage or not. The beta distribution would be a reasonable choice to model probability values ranging between 0 and 1 using only two parameters, $a$ and $b$. However, true observations of sleep staging often include many discrete values of 0 or 1, reflecting the convergence of sleep technologist annotations. The chosen distribution must handle this unique characteristic of the data, including the presence of true zero or one probabilities. This leads us to the Zero or One Inflated Beta distribution from <a href="https://www.sciencedirect.com/science/article/abs/pii/S0167947311003628">A general class of zero-or-one inflated beta regression models</a>, which is particularly suited for this task. By integrating this distribution into the Continuous Ranked Probability Score (CRPS), we can derive a robust objective function for model inference, ensuring that the calibrated probabilities align closely with the observed probabilities from human experts.
</div>

### Distribution choice: Zero or One Inflated beta (bic)

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Let define the random variable (RV) $z$ following a Zero or One inflated distribution with probability density function (PDF) and CDF with parameter $\alpha$, $a$ and $b$ and a chosen $c\in\{0,1\}$ :
\[
\begin{align}
bi_c(z;\alpha,a,b) &= 
\begin{cases}
\alpha & if \ z=c\\
(1-\alpha)f_{Beta}(z;a,b) & if \ z \in (0,1)
\end{cases}\\
\mathbf{BI}_c(z;\alpha,a,b)  &= \alpha\textbf{1}_{[c,1]}(z)+(1-\alpha)\mathbf{F}_{Beta}(z;a,b)
\label{eq:ZOI}
\end{align}
\]
</div>

The Zero or One Inflated Beta distribution is defined by the following parameters:
- **$\alpha$**: This parameter represents the probability mass at the point $c$. It determines the weight of the probability assigned to the discrete value $c$ (either 0 or 1).
- **$a$**: This is the shape parameter of the Beta distribution. It influences the skewness of the distribution.
- **$b$**: This is another shape parameter of the Beta distribution. Along with $a$, it defines the shape of the Beta distribution.
- **$c$**: This is the discrete value (either 0 or 1) where the distribution is inflated. It indicates the point at which the probability mass $\alpha$ is concentrated.
- **$f_{Beta}(z; a, b)$**: This is the probability density function (PDF) of the Beta distribution. It describes the likelihood of a random variable $z$ taking on a particular value within the interval (0, 1). The PDF of the Beta distribution is given by:
    $$
    f_{Beta}(z; a, b) = \frac{z^{a-1} (1-z)^{b-1}}{B(a, b)}
    $$
    where $B(a, b)$ is the Beta function, which acts as a normalization constant to ensure that the total probability integrates to 1.
- **$\mathbf{F}_{Beta}(z; a, b)$**: This is the cumulative distribution function (CDF) of the Beta distribution. It represents the probability that a random variable $z$ is less than or equal to a certain value. The CDF of the Beta distribution is given by:
    $$
    \mathbf{F}_{Beta}(z; a, b) = \int_0^z f_{Beta}(t; a, b) \, dt
    $$
    It accumulates the probabilities from the PDF up to the value $z$, providing the cumulative probability up to that point.

The beta distribution is well known to the community and there are many implementations of it, in this work the beta implementation of the Python scipy modules has been chosen.

```python
from scipy.stats import beta
import scipy.integrate as integrate
from scipy import special
import numpy as np

# Define the probability density function of the zero or one inflated beta distribution
def bic(z,c,alpha,a,b,tol=0.01):
    if c==z:
        return alpha
    else:
        if z>=1:
            z=1-tol
        if z<=0:
            z=tol
        return (1-alpha)*beta.pdf(z, a, b)

# Define the cumulative distribution function of the zero or one inflated beta distribution
def BIc( z,c,alpha,a,b,tol=0.01):
    if z>=1:
        z=1-tol
    if z<=0:
        z=tol
    return (1 - alpha) * beta.cdf(z, a, b) + alpha *((z >= c) &(z <= 1))

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
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
A tolerance $tol$ hyperparameters has been added to prevent any issues with $z$ and the beta distribution extremes (e.g. case where c=1, but z=0 cans lead to beta.pdf diverging to the infinite).
</div>


### CRPS of bic

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Now, lets supposed that the ensemble dataset $X$ are samples distributed from the Zero or One Inflated Beta distribution. The definition of the CRPS $(\ref{eq:CRPS})$ is then used to infer the dataset ($X$,$y$). By integrating this expression with the CDF $\mathbf{BI}_c$ of the Zero or One Inflated Beta distribution, an expression of the $CRPS$ only relying on the distrbution parameters ($c$,$\alpha$,$a$,$b$) and the true observation $y$ is derived:

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
This approach allows us to effectively calibrate the ensemble of sleep staging probabilities, ensuring that the calibrated probabilities align closely with the observed probabilities from human experts. Following the previous implemention og $bi_c$, here is the $CRPS$ python implementation:
</div>

```python
def crps_operation(y,c,alpha,a,b,intg=False, tol=0.01):
    if intg:
        return integrate.quad(lambda x: (BIc(x,c,alpha,a,b)-1)**2 if x>=y else (BIc(x,c,alpha,a,b))**2, 0, 1)[0]
    else:
        if y>=1:
            y=1-tol
        if y<=0:
            y=tol
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
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
A tolerance $tol$ hyperparameters has been added to prevent any issues with $y$ and the beta distribution extremes. Numerical integration of the crps is available to compare the derived expression. With same parameters, both methods converge to the same results.
</div>
---

---
## Neural Network architectures

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
In this section, the application of neural networks for calibrating sleep stage probabilities is explored. Two main neural network framework to address the calibration problem are proposed: a local and a global. The local focuses on calibrating each sleep stage probability individually using the CRPS as the objective function. This approach ensures that the calibrated probabilities align closely with the observed probabilities provided by sleep technologists. On the other hand, the global architecture aims to solve both the classification and calibration problems simultaneously. By leveraging a comprehensive neural network design, this model integrates the classification of sleep stages with the calibration of the resulting probabilities.
<br>
Both architectures are designed including Multip-Layer Perceptron (MLP) heads to infer different kinds of inputs, such as ensemble statistics derived from predicted sleep staging probabilities ensemble, and auxiliaries features extracted from signals. These models aim to deliver robust and interpretable sleep staging outcomes.
</div>

### Local

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
The proposed local architectures are designed to calibrate the ensemble of sleep staging probabilities and the observed probabilities. The first architecture is based on a simple MLP head depicted in Figure (1) and named $\textbf{MLP CS}$. Following the approach of <a href="https://journals.ametsoc.org/view/journals/mwre/146/11/mwr-d-18-0187.1.xml?tab_body=fulltext-display">Neural Networks for Postprocessing Ensemble Weather Forecasts</a>, it is designed to infer ensemble statistics such as mean, variance, and the estimated parameters \(c\), \(\alpha\), \(a\), and \(b\).  \(c\) is estimated by isolating the 0 or 1 elements of the ensemble and taking the majority value. If there are no 0 or 1 elements available in the ensemble,  \(c\) is determined by comparing the ensemble mean to a threshold of 0.5. If the mean is strictly greater than 0.5, 
 \(c\) is set to 1; otherwise, it is set to 0. \(\alpha\) is estimated by counting the ensemble member equal to the estimated \(c\). \(a\) is derived from the estimation formula $\hat{a} = \frac{\mu(\mu(1-\mu)}{\sigma^2}-1$, and \(b\) from $\hat{b} = \frac{\hat{a}(1-\mu)}{\mu}$, where $\mu$ is the ensemble empirical mean
 and $\sigma^2$ the empirical
 variance.
</div>

```python

def compute_c_alpha_a_b(row):
    if sum((row==0)|(row==1))>0:
        c = int(np.mean(row[(row==0)|(row==1)]).round(0))
    else:
        c = int(np.mean(row).round(0)>0.5)
    alpha = np.mean((row==c))
    mu = np.mean(row[row!=c])

    
    if sum(row!=c)>1:
        s2 = np.var(row[row!=c])
        a = mu*(mu*(1-mu)/s2-1)
        b = a*(1-mu)/mu
    else:
        a = 0
        b = 0
    return c,alpha,a,b

```

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
By incorporating these ensemble statistics, the MLP model aims to learn the underlying patterns and relationships between the predicted and observed probabilities, ultimately improving the calibration accuracy. The model's ability to handle complex, non-linear relationships makes it well-suited for this calibration task, ensuring that the calibrated probabilities align closely with the true observations provided by sleep technologists. In Figure (1), the MLP head consists of three hidden layers composed of $D=64$, $D=62$, and $D=16$ neurons with ReLU as the activation function. A dropout with the parameter set to 0.3 is set between each hidden layer. The output layer is composed of 4 neurons outputting respectively \(c\), \(\alpha\), \(a\), and \(b\). Sigmoid activation functions are set to \(c\) and \(\alpha\), and Softplus activation functions are set for \(a\) and \(b\). To prevent any issues of beta distribution divergences in the $CRPS$ function, outputs \(a\) and \(b\) are clipped between 0.05 and 20.
<br>
<br>
The 
<figure>
  <img
  src="/assets/project_sleep_stage_calibration/figures/MLP_CSS.png"
  alt="Double CNN architecture."
  >
  <em>Figure (2) -A multi-model MLP architecture that performs a zero or a one-inflated beta calibration with ensemble statistics and auxiliary features as inputs.</em>
</figure>

</div>

### CRPS keras ops implementation
#### TO DO
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
#### TO DO
<div style="text-align: justify">

</div>

### Global
#### TO DO
<div style="text-align: justify">

<figure>
  <img
  src="/assets/project_sleep_stage_calibration/figures/fMLP_CC.png"
  alt="Double CNN architecture."
  >
  <em>Figure (3) - Double CNN architecture.</em>
</figure>

</div>
---

---
## Results
### TO DO
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