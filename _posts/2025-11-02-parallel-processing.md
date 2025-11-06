---
layout: post
author: Rui F. David
title:  "Parallel Processing"
date:   2120-05-13 05:27:00 -0400
usemathjax: false
published: false
categories: software engineering
toc: true
---

# Introduction

## Many-threaded GPU vs Multicore CPU

[explain the difference between these]

## Hyperthreading and Many-thread

Hyperthreading is a technology developed by Intel that allows a single physical
processor to act as a multiple processor through logical units. The core idea
is that the CPU has various execution units (for arithmetic operations, memory
access, etc.) that may not be fully utilized at all times. Hyperthreading takes
advantage of this by allowing the core to work two separate threads simultaneously,
using whichever execution units are available at any given moment.

In contrast, many-thread processing refers to the ability of a system to handle
a large number of threads concurrently, often through multiple cores or processors.
For instance, NVIDIA Tesla Ampere 100 GPUs  XXX

TODO:

https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

[TABLE comparison with Hopper, Ampere, Blackwell]



## Data Parallelism

[good example of RGB]


## Task parallelism

