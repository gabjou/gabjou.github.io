---
title: Ensemble forecast uncertainty quantification for coarse-grid computational fluid dynamics
date: 2020-05-28 02:30:00 +0000
categories: [articles]
tags: [collaboration, ensemble forecast, fluid dynamics]
image: 
    path: '/assets/article_ensemble_forecast_uq/figures/rescale_es.png'
    alt: Figure (1) - Energy score (ES) of the temporal structure of the SQG turbulence flow ensemble forecast.
---

---
## Information

Short Presentation of a contribution in [New Trends in Ensemble Forecast Strategy: Uncertainty Quantification for Coarse-Grid Computational Fluid Dynamics](https://link.springer.com/article/10.1007/s11831-020-09437-x)

---

---
## Abstract
<div style="text-align: justify">
Numerical simulations of industrial and geophysical fuid flows cannot usually solve the exact Navier-Stokes equations. Accordingly, they encompass strong local errors. For some applications-like coupling models and measurements-these errors need to be accurately quantified, and ensemble forecast is a way to achieve this goal. The ensemble forecast field unlocks new aspects for the uncertainty quantification. It allows to estimate and compare the underlying system uncertainty distribution to the true observation. In this blog, we will introduce the statistical estimator of the continous ranked probability score from the ensemble forecast calibration field, usefull to measure the calibration performance of an ensemble model.
</div>

---

---
## Main Contribution
<script async src="/assets/article_ensemble_forecast_uq/js/mathjax.js"></script>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Lets consider we have a high-resolution deterministic surface quasi-geostrophic
(SQG) model generating a high dimensional flow in space-time of in squared grid (128x128) representing an area of 1000x1000 km. Such high-resolution model are time consuming to fully execute, new method are investigated to challenge it. In the presented paper, a stochastic version of the SQG model is presented and named $\textbf{mu}_{spec}$. This model is able to reproduce result from the high-resolution model for a lower execution cost and simulation resolution. Moreover, it presents the ability to generate an ensemble of forecast for small perturbation of the initial input.
</div><br>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
As a reference system, a high-resolution SQG model is considered to generate multiphysical (D - set of physical dimension) observation in space (S - set of space point) and time (T - set of time) $q^{(0)}_{dst}$ of our fluid. The stochastic SQG model is our forecasting model delivering a space-time ensemble of $n_e$ forecasts $q_{dst} = \{q_{dst}^{(1)},...,q_{dst}^{(n_e)}\}$ realization depending on a set of initial perturbed inputs. 

By definition the continous ranked probability score (CRPS) is a pointwise score measuring the distance between the observation $q^{(0)}$ and the cumulative distribution function of the ensemble forecast. The CRPS cannot summarize the whole uncertainty of the ensemble in one value in a multivariate case and it needs an apriori on the distribution of the ensemble forecast. The statistical estimator of the continous ranked probability score $\widehat{\text{CRPS}}$ does not break the rule and can only support observation and ensemble for a chosen d-physical dimension and fixed s-spatial, t-time points. However, the ($\widehat{\text{CRPS}}$) is an empirical estimator of the CRPS only based on samples from the ensemble forecast and the observation, so it supports unknown distribution of the ensemble.

Therefore, lets define $\widehat{\text{CRPS}}$ for a given univariate ensemble distribution $q$ and obervation $q^{(0)}$: 
</div><br>

$$
  \widehat{\text{CRPS}}(q,q^o)
= \frac{1}{n_e}\sum^{n_e}_{i=1}|q^{(i)}-q^o|-\frac{1}{2n_e^2}\sum^{n_e}_{i=1}\sum^{n_e}_{j=1}|q^{(i)}-q^{(j)}|
\label{crps}
$$

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
From the equation $\ref{crps}$, the $\widehat{\text{CRPS}}$ is a difference between two main bais. The first part estimates the bias of the ensemble members $q^{(i)}$ and the observation $q^o$ and the second part estimates the bias of the ensemble members with themselves. More the observation is far from the ensemble members and more the bias explodes making the $\widehat{\text{CRPS}}$ growing positively. In most of the time, this situation is associated to uncalibrated ensemble. In contrast, calibrated ensemble should provide distribution where the observation become an indistinguishable member. In this case, both bias would anihilate each other making the $\widehat{\text{CRPS}}$ close to zero. However, it is relevent to remmber that the $\widehat{\text{CRPS}}$ is a pointwise and can not be use alone to conclude about if an ensemble is calibrate or not. 
</div><br>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
Before concluding this blog, Figure (2) shows the resulting high-resolution SQG simulation as the $q^{o}$ observation at t=0 and t=9 days. The initial state of the model is set to a homogeneous Gaussian field with small vortices (red and blue point clouds). At t=9 days, some of this initial energy cascades to larger scales, creating larger vorticesby merging small vortices. The right panel shows the resulting $\widehat{\text{CRPS}}$, which measures the distance between the high resolution and the ensemble forecast from $mu_{spec}$, the stochastic SQG model. The highest values of $\widehat{\text{CRPS}}$ are in the centre of the vortices. This indicates that the ensemble forecast has spatial inconsistencies where its vorticesare not aligned with the high-resolution simulation. The ensemble model may underestimate the intensity of the eddies due to its lower representation of the physics.
</div>


<figure>
  <img
  src="/assets/article_ensemble_forecast_uq/figures/es_vs_spec.png"
  >
  <em>Figure (2) - Initial state and high-resolution SQG observation with the spatial analysis of the estimated CRPS for the mu_spec ensemble forecast. Left figure represent the initial state of the system; middle figure is high-resolution SQG observation at t=9 days; and right figure is the CRPS estimated on the ensemble and observation at t=20 days for each spatial point and a chosen physical dimension.</em>
</figure>

<div style="text-align: justify">
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
In conclusion, $\widehat{\text{CRPS}}$ is an easy to use statistical metric to asses distance between ensemble and observation without any assumption on the ensemble model distribution. It helps to identify error pattern from the ensemble model affecting the quality of the forecast. Moreover, the theoritical definition and the estimator of the CRPS can be extended to multivariate forecast with the Energy score using euclidian distance to estimate the bias between ensemble members and the observation. The introduction Figure (1) shows the obtained Energy score (ES) estimated on the temporal structure of the ensemble forecast. An other multivariate metric named the Variogram is also presented in the paper. It relies on the assesement of the multivariate correlation structure between the ensemble and the observation.
</div>
---
