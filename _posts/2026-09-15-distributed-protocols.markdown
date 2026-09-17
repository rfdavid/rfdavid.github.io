---
layout: post
author: Rui F. David
title:  "Distributed Protocols"
date:   2026-09-15 00:27:00 -0400
draft: true
usemathjax: true
published: true
categories: software engineering
toc: true
---

# Paxos

## Introduction

Paxos is a distributed consensus algorithm that allows several
computers to agree on a single value. There are some variations
of Paxos, as well as different implementations. We will see how
Cassandra implemented Cassandra and what was changed from the original proposal.

There is a great essay from Leslie Lamport named "Paxos Made Simple"
which explains how the algorithm operates in two phases. Don't worry
about understanding it now, we will have some background before and come back to these phases.

Phase 1. (a) A proposer selects a proposal number n and sends a prepare
request with number n to a majority of acceptors.

(b) If an acceptor receives a prepare request with number n greater than
that of any prepare request to which it has already responded, then it
responds to the request with a promise not to accept any more proposals
numbered less than n and with the highest-numbered proposal (if any) that it has accepted.

Phase 2. (a) If the proposer receives a response to its prepare requests
(numbered n) from a majority of acceptors, then it sends an accept request
to each of those acceptors for a proposal numbered n wit value v, where v
is the value of the highest-numbered proposal among the responses, or is
any value if the responses reported no proposals.

(b) If an acceptor receives an accept request for a proposal numbered n,
it accepts the proposal unless it has already responded to a prepare request
having a number greater than n.

## The three PALs: Proposers, Acceptors, and Learners

We start by describing the three roles in Paxos: Proposers, Acceptors, and Learners. A single process may have more than one role.
In simple words, we can describe in a very high level:

**Proposers**: suggest a value for the group to agree on Acceptors: vote on the proposed values
**Acceptors**: vote on the proposed values
**Learners**: apply the values that was chosen

### Proposers

### Acceptors

### Learners


https://lamport.azurewebsites.net/pubs/paxos-simple.pdf


## References

{% bibliography --cited %}
