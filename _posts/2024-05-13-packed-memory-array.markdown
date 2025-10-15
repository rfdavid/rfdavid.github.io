---
layout: post
author: Rui F. David
title:  "Packed Memory Array"
date:   2120-05-13 05:27:00 -0400
usemathjax: true
categories: data structures
toc: true
---

## Introduction

Packed Memory Array (PMA) is one of the interesting data structures based on
cache-oblivious algorithms. The idea is to store data in a way you can minimize
the insertion/deletion/update cost and maximize the cache hit rate by leaving gaps between an array of elements. 
The paper "A sparse table implementation of priority queues" {% cite 10.5555/646235.682700 %} first
explored the idea of "PMA" by introducing gaps between elements to facilitate insertions. Further work explored 
cache-obvlious algorithms, which are designed to leverage the computer's cache
without requiring explicit knowledge of the memory system, differently from
cache-aware algorithms which are designed to exploit the cache mechanism.

## Cache-oblivious Algorithm

Many algorithms are cache-oblivious by its nature. Assume you have a simple
unsorted array and want to find a specific element. Having no previous
knowledge about the actual elements, the best is to perform a linear scan to
find that element. Assuming the elements are stored in a contiguous block of
memory, the CPU will load a block of memory into the cache, facilitating the
access of the next element, known as cache line filling. Anoter reason is
prefetching, where modern processors can predict the next memory access and
preemptively load it into the cache. 

## Packed Memory Array


## References

{% bibliography --cited %}
