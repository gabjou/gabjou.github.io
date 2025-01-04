---
title: Gaussian mixture models for Exchangeable ensemble forecast with jax implementation
date: 2021-11-09 00:00:00 +0000
categories: [articles]
tags: [ensemble forecast, Gaussian mixture model, exchangeability,python,jax,flax]
image: 
    path: '/assets/article_ensemble_forecast/figures/example.png'
    alt: Figure (1) - Clustering example of a bivariate exchangeable ensemble dataset clustered by an Exchangeable gaussian mixture model with three distributions.
---

---
## Information

Presentation of a deep learning extension proposed for the exchangeable Gaussian model from [Gaussian mixture models for clustering and calibration of ensemble weather forecasts](https://www.researchgate.net/profile/Goulven-Monnier/publication/358989436_Gaussian_mixture_models_for_clustering_and_calibration_of_ensemble_weather_forecasts/links/638789c0bbdef30dc9877e90/Gaussian-mixture-models-for-clustering-and-calibration-of-ensemble-weather-forecasts.pdf).
(Update December-2024) Full code implementation is given using python modules jax and flax and can be found at [DEGMM github repo](https://github.com/gabjou/DEGMM).
---

---
## Abstract
<div style="text-align: justify">
Nowadays, most weather forecast centres produce ensemble forecasts.  Ensemble forecasts provide information on the probability distribution of weather variables. Since the weather forecasting system is sensitive to the initial state of the atmosphere, the ensemble forecasts are subject to large temporal deviations from the true state of the atmosphere, leading to different distribution errors. One way to isolate these distribution errors is to apply a clustering algorithm. The Gaussian mixture model represents the target standard clustering algorithm with a Gaussian assumption on the data. Nevertheless, the ensemble prediction presents the interesting aspect of interchangeability, so that its members are not affected by their rank. Each member gives the same statistical description of the general ensemble distribution. The main GMM framework is not adapted to infer the ensemble, but only member by member. In this blog we will present the general framework of the Gaussian Mixture Model and introduce the novel Exchangeable Gaussian Mixture Model developed to fit an ensemble of exchangeable Gaussian random vectors.
</div>
---

---
## Gaussian mixture model
<script async src="/assets/article_ensemble_forecast/js/mathjax.js"></script>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
First of all, lets present the Gaussian Mixture Model (GMM). The model describes the distribution of a couple of variables composed of a latent class variable and an observed continuous variable. The class variable $Z$ is distributed according to a multinomial distribution with probability vector $(\pi_1,\cdots,\pi_K)$ leading the observed continuous data assignement to one of the k-th clusters. The continuous variable $X \in \mathbb{R}^d$ has a Gaussian distribution with parameters $\mu_k$ and $\Sigma_k$ given a cluster assignement $Z=k$. The probability density function (pdf) of $X$ is given by
\[
\begin{equation}
f_X(x;\Psi) = \sum^K_{k=1} \pi_k \varphi(x;\mu_k,\Sigma_k)
\label{eq3.1.1.1}
\end{equation}
\]
where $\varphi(x;\mu,\Sigma)$ denotes the Gaussian pdf with mean $\mu$ and variance $\Sigma$.<br>
Given a sample $\{x_1, \cdots, x_n\}$ of $n$ independent realizations of the random variable $X$, the unknown parameters $\Psi = \left(\pi_1,\cdots,\pi_K,\mu_1,\cdots, \mu_K, \Sigma_1,\cdots,\Sigma_K \right)$ are estimated by maximizing the log-likelihood 
\[
\begin{equation}
 \log\mathcal{L}(x_1,\cdots,x_n;\Psi)
 = \sum^n_{i=1}\log \sum_{k=1}^K \pi_k \varphi(x_i;\mu_k,\Sigma_k).
\label{eq:loglik}
\end{equation}
\]
Most programming languages have an implementation of the Gaussian distribution pdf $\varphi$ which takes $\mu$ and $\Sigma$ as parameters. In this blog, we have chosen the jax version of the Gaussian pdf to implement the equations $\ref{eq3.1.1.1}$ and $\ref{eq:loglik}$:

</div>

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.random import PRNGKey, split, normal
from jax import vmap

# Define the Gaussian density function with x of dimension D, mu of dimension D, and sigma of dimension D,D
def gaussian_density(x, mu, Sigma):
    return multivariate_normal.pdf(x, mean=mu, cov=Sigma)

# Define the PDF for the Gaussian mixture with x of dimension D, mu of dimension K,D, and sigma of dimension K,D,D
def pdf_gmm(x, pi, mu, Sigma):
    K = len(pi)
    pdf_value = 0.0
    for k in range(K):
        pdf_value += pi[k] * gaussian_density(x, mu[k], Sigma[k])
    return pdf_value

# Define the log-likelihood function of the Gaussian mixture model for a data set X of dimension n,D, mu of dimension K,D, and sigma of dimension K,D,D
def log_likelihood_gmm(X, pi, mu, Sigma):
    log_pdf_gmm = vmap(lambda x: jnp.log(pdf_gmm(x, pi, mu, Sigma)))
    return jnp.sum(log_pdf_gmm(X))
```
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
GMM parameter estimation cannot be derived from solving the loglikelihood $\ref{eq:loglik}$ maximisation problem. Instead, the iterative Expectation-Maximisation (EM) algorithm is traditionally used. The EM algorithm alternates between two steps, the E-step and the M-step. At iteration $[h]$ of the EM algorithm, the E-step computes the posterior probabilities of cluster membership $\gamma^{[h]}_{ik}$ for all individuals $i= 1,...,n$, given the current parameter values $\Psi^{[h-1]}$. The posterior probabilities are computed using
\[
\begin{equation}
\gamma^{[h]}_{ik}  =  \mathbb{P}(Z=k|x_i,\Psi^{[h-1]}) 
 =  \frac{\pi^{[h-1]}_k \varphi(x_{i};\mu^{[h-1]}_k,\Sigma^{[h-1]}_k)}{\sum_{\ell=1}^K \pi^{[h-1]}_\ell \varphi(x_i;\mu^{[h-1]}_\ell,\Sigma^{[h-1]}_\ell)}.
\label{eq:gamma_ik}
\end{equation}
\]
Here is the python implementation of the GMM class embedding the E-step:
</div>

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the class GMM with parameter K: number of mixture components, max_iter: maximum EM iteration, tol: stop criterion.
class GMM:
    def __init__(self, K, max_iter=100, tol=1e-6,logger=logger):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None
        self.mu = None
        self.Sigma = None
        self.logger = logger
        self.epoch = -1
        self.loss_value = jnp.inf
        self.parameters =[{
            "epoch": self.epoch,
            'pi': self.pi,
            'mu': self.mu,
            'Sigma': self.Sigma,
            "loglikelihood": self.loss_value
        }]


    @property
    def model_parameters(self):
        return self.parameters
    
    @model_parameters.setter
    def model_parameters(self, value):
        self.parameters.append(value)

    # Define the Expectation of the EM for the step h taking X of dimension n,D and pi_h_1 of dimension K, mu_h_1 of dimension K,D, sigma_h_1 of dimension K,D,D; and then producing gamma_h of dimension n,K
    def e_step(self, X):
        K = self.K
        n = X.shape[0]
        gamma = jnp.zeros((n, K))

        denom = vmap(lambda i:jnp.sum(vmap(lambda k: self.pi[k] * gaussian_density(X[i,:], self.mu[k,:].T, self.Sigma[k,:,:]))(jnp.arange(K)))
                                    )(jnp.arange(n))

        for k in range(K):
            gamma_k = vmap(lambda i: self.pi[k] * gaussian_density(X[i,:], self.mu[k,:].T, self.Sigma[k,:,:]))(jnp.arange(n))
            gamma = gamma.at[:, k].set(gamma_k)
        gamma = gamma / denom[:, jnp.newaxis]
        return gamma
```
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Then, in the M-step, the conditional expectation of the  log-likelihood given the current parameters value is maximized. It leads to the following updates of  parameters at iteration $[h]$, for all $k= 1,...,K$:
\[
\begin{eqnarray}
\label{eq:pik}
&&{\pi}^{[h]}_k   = \frac{\sum^n_{i=1}\gamma^{[h]}_{ik}}{n}\\
\label{eq:muk}
&&{\mu}_k^{[h]}  = \frac{\sum_{i=1}^n\gamma^{[h]}_{ik}x_{i}}{\sum_{i=1}^n\gamma^{[h]}_{ik}}\\
\label{eq:Sigmak}
&&{\Sigma}_k^{[h]}   = \frac{\sum_{i=1}^n\gamma^{[h]}_{ik}(x_{i}-{\mu}_k^{[h]})(x_{i}-{\mu}_k^{[h]})^\intercal}
{\sum_{i=1}^n\gamma^{[h]}_{ik}}.
\end{eqnarray}
\]
The python implementation of the M-step for the GMM class is given as follow:
</div>

```python
# Define the Maximisation of the EM for the step h taking X of dimension n,D and gamma_h of dimension n,K, and then updating pi_h of dimension K, mu_h of dimension K,D, sigma_h of dimension K,D,D
def m_step(self, X, gamma):
    n, K = gamma.shape
    D = X.shape[1]
    N_k = jnp.sum(gamma, axis=0)
    self.pi = N_k / n
    self.mu = jnp.dot(gamma.T, X) / N_k[:, jnp.newaxis]
    self.Sigma = jnp.zeros((K, D, D))
    for k in range(K):
        weighted_X = jnp.sqrt(gamma[:, k][:, jnp.newaxis]) * (X - self.mu[k])
        self.Sigma = self.Sigma.at[k].set(jnp.dot(weighted_X.T, weighted_X) / N_k[k])
```
---

---
## Exchangeable Gaussian mixture model
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
In meteorological applications, numerical weather prediction models provide ensembles instead of regular samples. More precisely, each observation $i$ is a set of $M$ realisations ${x_{i1},\cdots, x_{iM}}$ of the variable $X$. These realisations are called {\it-members}. The members are considered as realisations of a vector of exchangeable variables $\{X_1, \cdots,X_M\}$. The exchangeability of the random vector induces that the joint distribution of $\{X_1, \cdots,X_M\}$ is invariant under variable permutations. This can be translated as each member $X_m$ carries the same statistical information describing the joint distribution. The GMM is initially designed to infer only one member of $\{X_1, \cdots,X_M\}$ missing information carried by the remaining $M-1$ members. We propose to exploit this problem with a new pdf constructed for the $\{X_1, \cdots,X_M\}$. However, this new pdf is designed with each member $X_m$ assumed to be independent of $X_\ell$ if $m \neq \ell$, and interchangeable. The GMM is then applied to $(Z,X_1,\cdots,X_M)$, where $Z$ is a discrete latent variable. The joint pdf of $(X_1,\cdots,X_M)$ is given by 
\begin{equation}
f_{X_1,...,X_M}(x_1,...,x_M;\Psi) =  \sum^K_{k=1}\pi_k\prod^M_{m=1}\varphi(x_m;\mu_k,\Sigma_k)
\label{eq:jpdf_ens}
\end{equation}
</div>

```python
# Define the joint PDF for the Gaussian mixture with an exchangeable random vector x of dimension M,D 
def joint_pdf_egmm(x, pi, mu, Sigma):
    K = len(pi)
    M = x.shape[0]
    pdf_value = 0.0
    for k in range(K):
        product = jnp.prod(vmap(lambda m: gaussian_density(x[m,:], mu[k,:].T, Sigma[k,:,:]))(jnp.arange(M)))
        
        pdf_value += pi[k] * product
    return pdf_value
```




<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
The posterior probabilities for observation $i$ are written for this joint pdf as follows:

\begin{equation}
\gamma^{[h]}_{ik}  =  \mathbb{P}(Z=k|x_{i1},\cdots,x_{iM},\Psi^{[h-1]}) 
=  \frac{\pi^{[h-1]}_k \prod_{m=1}^M\varphi(x_{im};\mu^{[h-1]}_k,\Sigma^{[h-1]}_k)}{\sum_{\ell=1}^K \pi^{[h-1]}_\ell \prod_{m=1}^M\varphi(x_{im};\mu^{[h-1]}_\ell,\Sigma^{[h-1]}_\ell)}.
\label{eq:gamma_ik_mv}
\end{equation}

The product of the marginal pdf in (\ref{eq:gamma_ik_mv})  affects the shape of posterior probabilities. More precisely, the between clusters boundaries hypersurfaces are steeper than for classical GMM. As a consequence, the weight given to the ensemble close to the centers of the classes is increased  in the E-step. It helps to improve the convergence rate of the EM algorithm. 

For all $k= 1,...,K$, the M-step is derived as

\begin{eqnarray}
\label{eq:pik_mv}
&&{\pi}^{[h]}_k   = \frac{\sum^n_{i=1}\gamma^{[h]}_{ik}}{n}\\
\label{eq:mu_mv}
&&{\mu}_k^{[h]}  = \frac{\sum_{i=1}^n\gamma^{[h]}_{ik} \sum_{m=1}^Mx_{im}}{M\sum_{i=1}^n\gamma^{[h]}_{ik}}\\
\label{eq:Sigma_mv}
&&{\Sigma}_k^{[h]}   = \frac{\sum_{i=1}^n\gamma^{[h]}_{ik}\sum_{m=1}^M(x_{im}-{\mu}_k^{[h]})(x_{im}-{\mu}_k^{[h]})^\intercal}
{M\sum_{i=1}^n\gamma^{[h]}_{ik}}
\end{eqnarray}

</div>

```python
# Define the Maximisation of the EM for the step h taking X of dimension n,M,D and gamma_h of dimension n,K, and then updating pi_h of dimension K, mu_h of dimension K,D, sigma_h of dimension K,D,D
def m_step_egmm(self, X, gamma):
    n, K = gamma.shape
    n,M,D = X.shape
    N_k = jnp.sum(gamma, axis=0)
    self.pi = N_k / n
    self.mu = vmap(lambda k: jnp.sum(vmap(lambda d: gamma[:,k]*(jnp.sum(X, axis=1)[:,d]))(jnp.arange(self.D)).T,axis=0) / (self.M * N_k[k]))(jnp.arange(self.K))
    self.Sigma = jnp.zeros((self.K, self.D, self.D))
    for k in range(self.K):
        weighted_X = jnp.sum(vmap(lambda i:gamma[i,k]*(X[i, :, :] - self.mu[k,:]).T@(X[i, :, :] - self.mu[k,:]))(jnp.arange(n)),axis=0)
        self.Sigma = self.Sigma.at[k].set(weighted_X/ (self.M * N_k[k]) + 1e-6 * jnp.eye(self.D))
```
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
In the model (\ref{eq:jpdf_ens}), the members are assumed to be independent of each other. Remember that in practice the ensembles are obtained by independent perturbations of the initial conditions and parameters when running the numerical weather prediction model. However, the physical model clearly induces some dependence between members. The independence properties could be relaxed in the model (\ref{eq:jpdf_ens}). For each $k$, the product $\prod^M_{m=1}\varphi(x_m;\mu_k,\Sigma_k)$ would be replaced by a multivariate Gaussian pdf with mean $\mu_k \mathbf{e}_M$, where $\mathbf{e}_M$ is a vector of $1$s. The covariance could be a block matrix with $\Sigma_k$ matrices on the diagonal and an extra diagonal matrix repeated on the other blocks. This improvement is beyond the scope of this blog. 
</div>
---

---
# Deep Exchangeable Gaussian mixture model
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
The work of <a href="https://openreview.net/forum?id=BJJLHbb0-">Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection</a>, proposes to explore a clustering algorithm in which a GMM includes a novel multi-layer perceptron (MLP) inference instead of the EM algorithm. In this blog, we propose a similar exploration to infer a GMM (DGMM) and a replaceable GMM (DEGMM). The DEGMM and DGMM follow the same iterative approach where an MLP is trained to assign each sample to a cluster by predicting $\gamma^{[h]}$ at each epoch. Using the predicted $\gamma^{[h]}$, the M-step updates the model parameter to compute the current value of the loglikelihood taken as the objective function. The MLP step replaces the E step of the EM algorithm.
</div>
Here is an example of the loss and fit functions of the DEGMM python class:

```python
    def loss_function(self, X,pi, mu, Sigma):
        # Calculate the log-likelihood
        log_lik = self.lambda1 * log_likelihood_egmm(X, pi, mu, Sigma)
        # Penalisation of singular matrices
        penalisation = self.lambda2 * jnp.sum(vmap(lambda k: jnp.sum(jnp.linalg.diagonal(Sigma[k,:,:])))(jnp.arange(self.K)))

        return -log_lik+penalisation  # Negative log-likelihood for minimization


    def fit(self, X, num_epochs=100, epsilon=1e-6, patience=3):
        self.epsilon = epsilon
        self.patience = patience
        patience_iter = 0
        n = X.shape[0]

        @jit
        def train_step(state, X):
            def loss_fn(params):
                # E-step
                output = self.mlp.apply(params, X.reshape(n, self.D * self.M))
                gamma = output
                
                # M-step
                N_k = jnp.sum(gamma, axis=0)
                pi = N_k / n
                mu = vmap(lambda k: jnp.sum(vmap(lambda d: gamma[:,k]*(jnp.sum(X, axis=1)[:,d]))(jnp.arange(self.D)).T,axis=0) / (self.M * N_k[k]))(jnp.arange(self.K))


                Sigma = jnp.zeros((self.K, self.D, self.D))
                for k in range(self.K):
                    weighted_X = jnp.sum(vmap(lambda i:gamma[i,k]*(X[i, :, :] - mu[k,:]).T@(X[i, :, :] - mu[k,:]))(jnp.arange(n)),axis=0)
                    Sigma = Sigma.at[k].set(weighted_X/ (self.M * N_k[k]) + 1e-6 * jnp.eye(self.D))
                loss_value = self.loss_function(X, pi, mu, Sigma)
                return loss_value, (pi, mu, Sigma, gamma)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_value, (pi, mu, Sigma, gamma)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value, pi, mu, Sigma, gamma
```

---

---
## Results
<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
A data set $X$ of n=1000 samples, M=10 interchangeable members, D=2 bivariate, was generated using $K=3$ Gaussian components. This dataset is used to determine the difference between the inference of the GMM, the DGMM trained on a randomly selected member, and the DEGMM inferring the entire ensemble. Figure (2) provides a comparison of the performance of each model. The accuracy, which measures the correctness of cluster assignment between the true and predicted samples, is estimated per model and shown in the title of the figure. The accuracy of DEGMM shows the best performance with 98% correctness with the true samples, in contrast to the performance of DGMM and GMM. 

<figure>
  <img
  src="/assets/article_ensemble_forecast/figures/results.png"
  >
  <em>Figure (2) - Comparison of generated clusters from the exchangeable ensemble and fitted clusters from GMM, DGMM and DEGMM. Squares represent cluster centres. Accuracy between the predicted cluster samples and the actual generated clusters is given in the title per model.</em>
</figure>

This results was expected because DGMM and GMM are inferred on one member using only 1/10 information from the dataset. In the paper <a href="https://www.researchgate.net/profile/Goulven-Monnier/publication/358989436_Gaussian_mixture_models_for_clustering_and_calibration_of_ensemble_weather_forecasts/links/638789c0bbdef30dc9877e90/Gaussian-mixture-models-for-clustering-and-calibration-of-ensemble-weather-forecasts.pdf">Gaussian mixture models for clustering and calibration of ensemble weather forecasts</a>, other extensions of the GMM are explored to fit correctly datasets composed of exchangeable datasets. However, those models are only implemented with R language <a href="https://gitlab.com/gabrijou/gaussianmixturemodels.git"></a>.
</div>

<figure>
  <img
  src="/assets/article_ensemble_forecast/figures/inference_degmm.gif"
  >
  <em>Figure (3) - Representation of the DEGMM assignement during the iterative inference process.</em>
</figure>


---