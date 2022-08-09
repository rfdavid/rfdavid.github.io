---
layout: post
author: Rui F. David
title:  "Influence Functions in Machine Learning"
date:   2022-07-27 09:00:00 -0400
usemathjax: true
toc: true
categories: paper
---

## Introduction

With the increasing complexity of the machine learning models, the generated
predictions are not easily interpretable by humans and are usually treated as black-box
models. To address this issue, a rising field of explainability try to understand 
why those models make certain predictions. In recent years, the work by {% cite
pmlr-v70-koh17a %} has attracted a lot of attention into many fields,
using the idea of influence functions {% cite 10.2307/2285666 %} to identify
the most responsible training points for a given prediction.

## Robust Statistics 

Statistical methods relies explicitely or implicitely on assumptions based on
the data analysis and the problem stated. The assumption usually concern the
probabiliy distribution of the dataset. The most widely framework used makes
the assumption that the observed data have a normal (Gaussian) distribution, 
and this *classical* statistical method has been used for regression, analysis of
variance and multivariate analysis.  However, real-life data is noisy and contain 
atypical observations, called outliers. Those observations deviate from the
general pattern of data, and classical estimates such as sample mean and sample
varianve can be highly adversely influenced. This can result in a bad fit of data.
Robust statistics provides measures of robustness to provide a good fit for data 
containing outliers {% cite maronna2006robust %}.


### Influence Functions

The Influence Functions (IF) was first introduced in "The Influence Curve and Its Role in
Robust Estimation" {% cite 10.2307/2285666 %}, and measures the impact of an infinitesimal perturbation on
an estimator. The very interesting work by {% cite pmlr-v70-koh17a %} brought
this methodology into machine learning. 

### Influence Functions in Machine Learning

Consider a image classification task where the goal is to predict the label for
a given image. We want to measure the impact of a particular training image on
a testing image. A naive approach is to remove the image and retrain the model.
However, this approach is prohibitively expensive. To overcome this problem, influence
function upweight that particular point by an infinitesimal amount and measure
the impact in the loss function, without having to train the model.


![medium](/assets/images/upweight-a-training-point.jpg "Upweighting a training point")
_Figure 1: The fish image is upweighted by an infinitesimal amount so the model
try harder to fit that particular sample. Image by the author._

### Change in Parameters 

First we need to understand how the parameters changes after perturbing a
particular training point $$z$$ by an infinitesimal amount $$\epsilon$$,
defined by $$\theta - \hat\theta$$ where $$\theta$$ is the original parameters
for the full training data and $$\hat\theta$$ is the new set of parameters
after upweighting:   

$$\hat\theta_{\epsilon,z} \overset{\mathrm{def}}{=}
argmin_{\theta\in\Theta}\frac{1}{n}\sum_{i=1}^{n}L(z_i,\theta) + \epsilon L(z,\theta)$$

As we want to measure the rate of change of the parameters after perturbing the
point, the derivation made by {% cite cook1982influence %} yields to the following:

$$ I_{up,params}(z) \overset{\mathrm{def}}{=} \frac{d\hat\theta_{\epsilon,z}}{d\epsilon} \bigg|_{\epsilon=0} = -H_{\hat\theta}^{-1}\nabla_{\theta} L(z,\hat\theta)  $$

Where $$H_{\hat\theta}$$ is the Hessian matrix and assumed to be positive
definite (symmetric with all positive eigenvalues), which can be calculated by
$$ \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}^2 L(z_i,\hat\theta) $$.  

**The equation $$ I_{up,params}(z) $$ gives the influence of a single training
point z on the parameters $$\theta$$.** Therefore, when multiplying $$-\frac{1}{n} I_{up,params}(z)$$ 
the result is the same as removing $$z$$ and re-training the model.

### Change in the Loss Function

As we want to measure the change in the loss function with respect to
infinitesimal perturbation $$\epsilon$$, applying chain rule gives the
following equation that can be explained as **the impact of $$z$$ on $$z_{test}$$:**

$$ \frac{d L(z_{test},\hat\theta_{\epsilon, z})}{d\epsilon} \bigg|_{\epsilon=0} = -\nabla_\theta L(z_{test},\hat\theta)^T H_{\hat\theta}^{-1} \nabla_\theta L(z,\hat\theta) $$


## Influence Functions on Groups

 The technique was first introduced in "The Influence Curve and Its Role in Robust Estimation" {% cite 10.2307/2285666 %}.
{% cite pmlr-v70-koh17a %} brought this technique to machine learning to
measure the impact of a particular data point on a prediction (ie: the impact
of a specific image on a classifier).

Paper reference: {% cite JMLR:v18:16-491 %}

## How to Efficiently Calculate Influence Functions

### Conjugate Gradients

Related paper: {% cite 10.5555/3104322.3104416 %}

### Linear Time Stochastic Second-Order Algorithm (LiSSA)

### FastIF

## Applications

### Explainability

### Adversarial Attacks

### Debugging

### Dataset relabelling

## The Problem with Influence Functions

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
inversion with torch.autograd, truncated conjugate gradients and LiSSa.

[Fast Influence Functions](https://github.com/salesforce/fast-influence-functions)  
A modified influence function computation using k-Nearest Neighbors (kNN),
implemented in PyTorch.

### Others

[Influence Function with LiSSA](https://github.com/nayopu/influence_function_with_lissa)  
A simple implementation with LiSSA on TensorFlow.

[Influence Pytorch](https://github.com/jrepifano/influence-pytorch)
One-file code with the implementation for a random classification problem.

[IF notebook](https://github.com/zedyang/46927-Project)  
Python notebook with IF applied to other algorithms (Trees, {% include def.html term="Ridge Regression" %}).

[Influence Functions Pytorch](https://github.com/Benqalu/influence-functions-pytorch)  
Another implementation of influence functions.

## References

{% bibliography --cited %}
