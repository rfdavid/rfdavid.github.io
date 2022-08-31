---
layout: post
author: Rui F. David
title:  "Influence Functions in Machine Learning"
date:   2022-08-31 09:00:00 -0400
usemathjax: true
toc: true
categories: paper
---

## Introduction

With the increasing complexity of machine learning models, the generated
predictions are not easily interpretable by humans and are usually treated as black-box
models. To address this issue, a rising field of explainability try to understand 
why those models make certain predictions. In recent years, the work by {% cite
pmlr-v70-koh17a %} has attracted a lot of attention in many fields,
using the idea of influence functions {% cite 10.2307/2285666 %} to identify
the most responsible training points for a given prediction.

## Robust Statistics 

Statistical methods rely explicitly or implicitly on assumptions based on
the data analysis and the problem stated. The assumption usually concerns the
probability distribution of the dataset. The most widely used framework makes
the assumption that the observed data have a normal (Gaussian) distribution, 
and this *classical* statistical method has been used for regression, analysis of
variance and multivariate analysis.  However, real-life data is noisy and contain 
atypical observations, called outliers. Those observations deviate from the
general pattern of data, and classical estimates such as sample mean and sample
variance can be highly adversely influenced. This can result in a bad fit of data.
Robust statistics provide measures of robustness to provide a good fit for data 
containing outliers {% cite maronna2006robust %}.


### Influence Functions

The Influence Functions (IF) was first introduced in "The Influence Curve and Its Role in
Robust Estimation" {% cite 10.2307/2285666 %}, and measures the impact of an infinitesimal perturbation on
an estimator. The very interesting work by {% cite pmlr-v70-koh17a %} brought
this methodology into machine learning. 

### Influence Functions in Machine Learning

Consider an image classification task where the goal is to predict the label for
a given image. We want to measure the impact of a particular training image on
a testing image. A naive approach is to remove the image and retrain the model.
However, this approach is prohibitively expensive. To overcome this problem, influence
function upweight that particular point by an infinitesimal amount and measure
the impact in the loss function without having to train the model.


![medium](/assets/images/upweight-a-training-point.jpg "Upweighting a training point")
_Figure 1: The fish image is upweighted by an infinitesimal amount so the model
try harder to fit that particular sample. Image by the author._

### Change in Parameters 

The empirical risk minimizer to solve an optimization problem can be defined as
the following:

$$
\begin{equation}
  \hat\theta = arg \; \underset{\theta}{min} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(z_i, \theta)
\end{equation}
$$

Where $$z_i$$ is each training point from a training sample.  First, we need to understand how 
the parameters $$\hat\theta$$ change after perturbing a particular training point $$z$$ by an infinitesimal 
amount $$\epsilon$$, defined by $$\theta - \hat\theta$$ where $$\theta$$ is the original parameters
for the full training data and $$\hat\theta$$ is the new set of parameters after upweighting:   

$$
\begin{equation}
  \hat\theta_{\epsilon,z} = arg \; \underset{\theta}{min} \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(z_i,\theta) + \epsilon \mathcal{L}(z,\theta)
\end{equation}
$$

As we want to measure the rate of change of the parameters after perturbing the
point, the derivation made by {% cite cook1982influence %} yields the following:

$$ 
\begin{equation}
  I(z) = \frac{d\hat\theta_{\epsilon,z}}{d\epsilon} \bigg|_{\epsilon=0} = -H_{\hat\theta}^{-1}\nabla_{\theta} \mathcal{L}(z,\hat\theta)
\end{equation}
$$

Where $$H_{\hat\theta}$$ is the Hessian matrix and assumed to be positive
definite (symmetric with all positive eigenvalues), which can be calculated by
$$ \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}^2 \mathcal{L}(z_i,\hat\theta) $$.  

**The equation $$ 3 $$ gives the influence of a single training
point z on the parameters $$\theta$$.** When multiplying $$-\frac{1}{n} I(z)$$ 
the result is similar as removing $$z$$ and re-training the model.

### Change in the Loss Function

As we want to measure the change in the loss function for a particular testing
point, applying chain rule gives the following equation:

$$ 
\begin{equation}
  I(z, z_{test}) =  \frac{d L(z_{test},\hat\theta_{\epsilon, z})}{d\epsilon} \bigg|_{\epsilon=0} = -\nabla_\theta \mathcal{L}(z_{test},\hat\theta)^T H_{\hat\theta}^{-1} \nabla_\theta \mathcal{L}(z,\hat\theta)
\end{equation}
$$

$$ \frac{1}{n} I(z, z_{test}) $$ approximately measures **the impact of $$z$$ on $$z_{test}$$**.
This is based on the assumption that the underlying loss function is strictly {% include def.html term="convex" %} in 
the parameters $$\theta$$. Some loss functions are not differentiable 
({% include def.html term="hinge loss" %}), so in this case, one of the contributions of 
Koh's work is to approximate to a differentiable region right at the margin.

## Influence Functions on Groups

As previously seen, the influence functions measure the impact of a training point 
in a single testing point.  They are based on first-order 
{% include def.html term="Taylor approximation" %}, which is fairly accurate
for small changes. In order to study the effect of a large group of training
points, {% cite NEURIPS2019_a78482ce %} analyze this phenomenon where
influence functions can be used for some particular cases. It can be written as
the sum of the influences of individual points in a group:

$$ \sum_{i=1}^n I(z_i, z_{test}) = -\nabla_\theta \mathcal{L}(z_{test},\hat\theta)^T H_{\hat\theta}^{-1} \sum_{i=1}^n \nabla_\theta \mathcal{L}(z,\hat\theta)$$

Given a group $$\mathcal{U}$$ and $$ I(\mathcal{U})^{(1)} $$ the first-order group
influence, {% cite pmlr-v119-basu20b %} proposes second-order group influence
function to capture informative cross-dependencies among samples:

$$ I(\mathcal{U})^{2} =  I(\mathcal{U})^{(1)} + I(\mathcal{U})^{'} $$

Hence, first-order group influence function $$I(\mathcal{U})^{(1)}$$ can be
defined as:

$$ I(\mathcal{U})^{(1)} = \frac{\partial \theta_{\mathcal{U}}^{\epsilon}}{\partial \epsilon} \bigg|_{\epsilon=0} $$

And the second-order group influence $$I(\mathcal{U})^{'}$$ as:

$$ I(\mathcal{U})^{(1)} = \frac{\partial^2 \theta_{\mathcal{U}}^{\epsilon}}{\partial \epsilon^2} \bigg|_{\epsilon=0} $$

This technique was empirically proven that can be used to improve the selection
of the most influential group for a test sample across different group sizes
and types. The idea is to capture more information when the changes to the
underlying model are relatively large.

## The Calculation Bottleneck

Computing the inverse hessian is quite expensive and infeasible for a network with 
lots of parameters. In numpy, it can be calculated using  `numpy.linalg.inv`.
As a side note, numpy is mostly written in c and the high-level functions are
python bindings. Nevertheless, it is still an expensive function. In the
PyTorch framework, you can compute the Hessians using `torch.autograd.functional.hessian` 
and then inversing it with `torch.linalg.inv`.  In spite of that, second-order optimization
techniques can efficiently approximate the calculation.

### Conjugate Gradients

Conjugate gradient {% cite Shewchuk94 %} is an iterative method for solving large systems of linear
equations, and it is effective to solve systems in the form of $$ Ax = b $$.
In {% cite 10.5555/3104322.3104416 %}, the hessian is calculated by
approximation using second-order optimization technique. This method does not
invert the hessian directly but calculate the inverse hessian product:

$$ H^{-1} v = arg min_{t}(t^T Ht - v^Tt) $$

### Linear Time Stochastic Second-Order Algorithm (LiSSA)

The main idea of LiSSA {% cite JMLR:v18:16-491 %} is to use Taylor expansion ([Neumann series](https://en.wikipedia.org/wiki/Neumann_series)) to 
construct a natural estimator of the inverse Hessian:

$$ H^{-1} = \sum^{\infty}_{i=0} (I - H)^i $$

Rewriting this equation recursively, as $$ \lim_{j \to \infty} H_{j}^{-1} = H^{-1} $$, we have the following:

$$ H_{j}^{-1} = \sum^{j}_{i=0} (I - H)^i = I + (I - H) H^{-1}_{j-1}  $$



### FastIF

In order to improve the scalability and computational cost, FastIF {% cite
guo-etal-2021-fastif %} present a set of modifications to improve the runtime. 
The work uses k-neareast neighbours to narrow the search space down, 
which can be inexpensive for this context since i k-nn is a {% include def.html term="lazy learner" %}) algorithm.

## The Problem with Influence Functions

Influence functions are an approximation and do not always produce correct
values. In some particular settings, influence functions can have a significant loss in
information quality. It is known to work with convex loss functions, but for
non-convex setups, the estimations can not work as expected. The work
'Influence Functions in Deep Learning are Fragile' {% cite basu2021influence
%} examines the conditions where influence estimation can be applied to deep
networks through vast experimentation. In short, there are a few obstacles:

* The estimation in deeper architectures is erroneous, possibly due to poor
  inverse hessian estimation. Weight-decay regularization can help.
* Wide networks perform poorly. When increasing the width of a network, the
  correlation between the true difference in the loss and the influence
  function decreases substantially.
* Scale influence functions is challenging. ImageNet contains 1.2 million
  images in the training set, being difficult to evaluate if influence
  functions are effective since it is computationally prohibitive to re-train the 
  model multiple times, leaving each training point out of the training.
  

## Libraries

There are several implementations available in Python with PyTorch and
TensorFlow. A few others are built on R and Matlab.

[Influence Functions](https://github.com/kohpangwei/influence-release)  
The official version of {% cite pmlr-v70-koh17a %} built on TensorFlow.

[Influence Functions for PyTorch](https://github.com/nimarb/pytorch_influence_functions)  
PyTorch implementation. It uses stochastic estimation to calculate the
influence.

[Torch Influence](https://github.com/alstonlo/torch-influence)  
A recent implementation (Jul/2022) of influence functions on PyTorch, providing
three different ways to calculate the inverse hessian: direct computation and
inversion with torch.autograd, truncated conjugate gradients and LiSSA.

[Fast Influence Functions](https://github.com/salesforce/fast-influence-functions)  
A modified influence function computation using k-Nearest Neighbors (kNN),
implemented in PyTorch.

### Other implementations

[Influence Function with LiSSA](https://github.com/nayopu/influence_function_with_lissa)  
A simple implementation with LiSSA on TensorFlow.

[Influence Pytorch](https://github.com/jrepifano/influence-pytorch)
One-file code with the implementation for a random classification problem.

[IF notebook](https://github.com/zedyang/46927-Project)  
Python notebook with IF applied to other algorithms (Trees, {% include def.html term="Ridge Regression" %}).

[Influence Functions Pytorch](https://github.com/Benqalu/influence-functions-pytorch)  
Another implementation of influence functions.

## Applications

- **Explainability:** This is the most common use we explored so far, measuring
  the impact of a training point to explain the impact in a given testing point.
- **Adversarial Attacks:** Real-world data is noisy, and it can be problematic for machine learning.
Adversarial machine learning methods are methods used to feed a model with
deceptive input, changing the predictions of a classifier. Influence functions
can help by identifying how to modify a training point to increase the
loss in a target point.
- **Label mismatch:** Toy datasets are pretty good for experimentation, but
  real data might contain many mislabeled examples. The idea is to calculate
  the influence of a particular training point $$ I(z_{i}, z_{i}) $$ if that point was removed. 
  Email spam is a good example since it usually uses the user's input in
  classifying whether an email is spam or not.

## Conclusion

The very interesting work from {% cite pmlr-v70-koh17a %} brought influence
functions to the context of machine learning. In principle, this technique was
introduced more than 40 years ago by {% cite 10.2307/2285666 %}. 
One of the main contributions is how to apply to non-differentiable loss functions (i.e.
hinge loss). In addition to that, the paper uses other existing ideas to
overcome the computation issue, such as conjugate gradients and LiSSA
algorithm. Subsequent work studied influence functions on groups ({% cite NEURIPS2019_a78482ce %} 
and {% cite pmlr-v119-basu20b %}). The last used second-order influence
functions to capture hidden information when the group size is relatively large.
I believe this is a powerful technique that will continue to derive new ideas in
many different areas. One example is in pruning, where a single-shot pruning
technique was based on sensitivity connections {% cite lee2018snip %}, exploring
the idea of perturbing weights in a network. Another idea is in the area of
graphs, a popular framework JK Networks {% cite JKNets %} uses perturbation
analysis to measure what is the impact of a change in one node embedding in
another node embedding.

## References

{% bibliography --cited %}
