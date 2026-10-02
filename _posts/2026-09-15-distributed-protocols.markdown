---
layout: post
author: Rui F. David
title:  "Distributed Protocols"
date:   2026-09-15 00:27:00 -0400
draft: true
usemathjax: true
published: false
categories: software engineering
toc: true
---

# Paxos

## Introduction

Paxos is a distributed consensus algorithm that allows several
computers to agree on a single value. There are some variations
of Paxos, as well as different implementations. We will see how
Cassandra implemented Cassandra and what was changed from the original
proposal {% cite lamport1998parliament %}.

There is a great essay from Leslie Lamport named "Paxos Made Simple"
{% cite lamport2001paxos %} which explains how the algorithm operates in two phases. Don't worry
about understanding it now, we will have some background before and come back to these phases.

**Phase 1.**
> (a) A proposer selects a proposal number n and sends a prepare
>request with number n to a majority of acceptors.

> (b) If an acceptor receives a prepare request with number n greater than
> that of any prepare request to which it has already responded, then it
> responds to the request with a promise not to accept any more proposals
> numbered less than n and with the highest-numbered proposal (if any) that it has accepted.

**Phase 2.** 
> (a) If the proposer receives a response to its prepare requests
> (numbered n) from a majority of acceptors, then it sends an accept request
> to each of those acceptors for a proposal numbered n wit value v, where v
> is the value of the highest-numbered proposal among the responses, or is
> any value if the responses reported no proposals.

> (b) If an acceptor receives an accept request for a proposal numbered n,
> it accepts the proposal unless it has already responded to a prepare request
> having a number greater than n.

## The three PALs: Proposers, Acceptors, and Learners

We start by describing the three roles in Paxos: Proposers, Acceptors, and Learners. A single process may have more than one role.
In simple words, we can describe in a very high level:

**Proposers**: suggest a value for the group to agree on  
**Acceptors**: vote on the proposed values  
**Learners**: apply the values that was chosen

## Phases

![large](/assets/images/paxos.png "Paxos Protocol")
_Figure 1: Paxos Protocol._

### Phase 1

#### Phase 1a. Prepare

Coming back to Lamport's explanation {% cite lamport2001paxos %}, in the first phase a proposer
picks proposal `n` and sends a `Prepare(n)` message to a majority of acceptors.
`n` must be unique across all proposers and greater than any number this
proposer has used before. Numbers are also totally ordered, typically as a pair
`(round, proposerId)` but varies from implementation.

#### Phase 1b. Promise

Once acceptors receive the message from the proposers, they either respond or
reject. Depending on the implementation, acceptors can return a
`NACK(currentHigherBallot)` or simply reject.

An acceptors responds to a `Prepare(n)` if `n` is higher than the highest proposal
number it has already seen. Then, it returns a `Promise(n, na, va)`
where `na` and `va` means the previously accepted proposal, or empty if it has
never accepted one. By replying, the acceptor also promises never to accept
a proposal numbered lower than `n`.

### Phase 2

#### Phase 2a. Accept

Once received the majority of promises for its request `n` from acceptors,
then it sends an accept request to each of those acceptors `accept(n,v)` where
`n` is the proposal number and `v` is the actual value. The proposer must take
the value from the promise carrying the highest `na` among the promises
received, which could also be from another proposer. Only if no acceptor
reported a previously accepted proposal may it use its own value.

#### Phase 2b. Accepted

Once an acceptor receives an accept request `accept(n,v)`, it accepts the
proposal unless it has already responded to a prepare request having a number
greater than `n`. In a real system, this can cause preemption when having
multiple proposal happening at the same time.

Once a majority of acceptors have accepted the same `(n,v)`, the value `v` is
chosen. Learners find out through the accepted messages.

## Implementation in Cassandra

Having the necessary background about Paxos, we analyze how paxos is actually
implemented in a real system. Cassandra uses paxos to linearize transactions. This
is called Lightweight Transactions (LWT).

When there is a conditional mutation, LWT transaction happens using paxos
coordination. Examples:

{% highlight sql %}
INSERT INTO lock (key, owner)
VALUES ('XXX', '550e8400-e29b-41d4-a716-446655440000')
IF NOT EXISTS;

DELETE FROM lock
WHERE key = 'XXX'
IF owner = '550e8400-e29b-41d4-a716-446655440000';
{% endhighlight %}

The `IF` clause makes the mutations LWT/paxos operations.
A Cassandra cluster can contain multiple datacenters (DCs). In Cassandra, this
is configurable via consistency `SERIAL` (global across all datacenters) or
`LOCAL_SERIAL` (within the same datacenter):

{% highlight sql %}
SERIAL CONSISTENCY SERIAL; -- Paxos will run across all DCs
SERIAL CONSISTENCY LOCAL_SERIAL; -- Paxos will run only on coordinator's DC
{% endhighlight %}

## Paxos v1

## Paxos v2

## Accord

The accord consensus protocol is also a leaderless protocol. Cassandra 6
implements Accord.

---

EPaxos revisited:
https://www.usenix.org/conference/nsdi21/presentation/tollman

Draft Whitepaper for CEP-15:
CEP-15: Fast General Purpose Transactions

---
What it takes from EPaxos:

Any replica can coordinate a transaction. There's no stable leader.
It has a fast path: one WAN round trip when there are no conflicts.
It tracks dependencies between conflicting transactions.

Where it departs (closer to Caesar and Tempo):

Timestamp ordering. The coordinator proposes a timestamp, and replicas either accept it or propose a later one. Execution follows timestamp order, so you avoid EPaxos's dependency-graph cycles and the strongly-connected-component resolution needed to execute them.
No livelock. EPaxos and Caesar can stall under contention. Accord's timestamp scheme guarantees progress.
Flexible fast-path electorates. It can shrink the fast-path quorum when nodes are slow or down, so it keeps the fast path under partial failure. EPaxos degrades to the slow path in that case.
Reorder buffer. It uses loosely synchronized clocks plus a small delay to keep the fast path likely across distant regions.
Multi-shard transactions. It's built for general cross-partition transactions, not just a replicated log for a single state machine.




## Multi-Paxos

## References

{% bibliography --cited %}
