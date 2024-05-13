---
layout: post
author: Rui F. David
title:  "Neural Graph Reasoning"
date:   2120-05-30 05:27:00 -0400
usemathjax: true
categories: paper
---

## Introduction



Traditional databases organizes the data in a tabular way, and make use of
JOINs to explore the relationship between tables. These operations are very
expensive and inneficient when trying to retrieve data in a distant
relationship. Graph Databases are optimized to handle high-order relationships
between distant entities. For instance, querying a multi-hop information such
aspects `person a->knows->person b->knows->person c` is a simple task in graph
databases but very costly in relational dbs.

Graph databases are essencially useful in many critical use cases, especially when
combined with graph neural networks:
- Fraud detection
- Customer relationship
- Recommender systems
- Drug discovery

This post is a review of the paper [Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases](https://arxiv.org/pdf/2303.14617.pdf) {% cite ren2023neural %}. 
The paper expands around the idea of Complex logical query answering (CLQA) by
providing a framework from different perspectives: graph types (modality,
reasoning domain, background semantics),  modeling aspects (encoder, processor, decoder),
supported queries (operators, patterns, projected variables), datasets, evaluation metrics, and applications.
Traditional link prediction in GNN tasks can infer the probability of a link
between two given nodes. The proposal of CLQA is to go beyond one-hop link
prediction and solve a more complex reasoning, identifying links in multi-hops
locations in a potentially incomplete graph.
It introduces the concept of Neural Graph Databases (NGDBs), consisting in two
main components: **Neural Graph Storage (an encoder to store embeddings)** and **Neural Graph Engine
(the engine to interact with the storage embeddings results)**. 
An interesting problem that the paper explores is the incompleteness of graphs,
although it remains unclear how effectively they address this task.

RDF-style graphs is well-known in AI community for the capability of reasoning
and inference on knowledge graphs. Recent advances in graph machine learning
enabled a more expressive reasoning over large graphs in a latent space. Graph
Query Embedding (GQE) {% cite NEURIPS2018_ef50c335 %} laid foundations on
going beyong simple edge prediction to answer complex logical queries.

### Graph Query Embedding (GQE)

QGE was introduced in Embedding Logical Queries on Knowledge Graphs paper {%
cite NEURIPS2018_ef50c335 %}, and the focus of the work was to explore
unobserved edges in a conjunctive query. Conjuctive queries allows the resoning
about the existence among subgraph relationships. In formal words, it 
can be expressed as follow:

$$
\begin{equation}
  q = V_{?}.\exists V_{1}, \dots, V_{m} : e_{1} \land  e_{2} \land \dots \land e_{n}, \\
  where\ e_{i} = \tau(v_{j} V_{k}), V_{k} \in \{V_?, V_{1}, \dots, V_{m}\}, v_{j} \in \mathcal{V}, \tau \in \mathcal{R}\\
  or\ e_{i} = \tau(V_{j},V_{k}), V_{j}, V_{k} \in \{V_{?}, V_{1}, \dots, V_{m}\}, j \neq k, \tau \in \mathcal{R}
\end{equation}
$$ 

Figure 1 demonstrates two examples of conjunctive queries. The first example
tries to answer the question "Predict communities C in which user u is likely
to upvote a post". The subgraph is represented by node u (user), post (p) and
community (c). The dashed lines are irrelevant links for the query and the 
solid lines are the path that satifies the query. Some solid lines might be
missing in the subgraph, therefore the task is to predict these missing links -
the likelihood of pair of nodes between `user-upvote-post`. A naive solution is
to calculate and rank all subgraphs that satifies the query. However, the
required computation time for this approach is not feasible. To address this
problem, GQE develops an embedding-based framework to make queries on
incomplete knowledge graph.

![](/assets/images/graph-query-embeddings.png "Graph Query Embeddings")
_Figure 1: Examples of conjunctive graph queries. Image extracted from {% cite NEURIPS2018_ef50c335 %}._



## References

{% bibliography --cited %}
