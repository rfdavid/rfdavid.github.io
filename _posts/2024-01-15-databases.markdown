---
layout: post
author: Rui F. David
title:  "Databases"
date:   2120-05-30 05:27:00 -0400
usemathjax: true
categories: software engineering
---

# Introduction

Database is a way to organize a collection of data.

- Store database in simple data structures (relations)
- Physical storage left up to the DBMS implementation
- Access data through high-level language, DBMS figures our best execution
  strategy

There are three aspects of relational model: structure, the definition of the
database's relations and their contents. Integrity, to ensure the database's
content meets contraints. Manipulation, the interface the users access and
modify content.

Two key concepts:
*Relation*: a relation is essentially a table. It's an unordered set (rows order
doesn't matter) and represents a relationship between two entities, for
example, customers and orders.
*Tuple*: a tuple is basically the row in a table, represented by a set of
attribute values.

A _primary key_ is a unique key that identifies a single tuple. A _foreign_ 
key specifies that an attribute from one relation maps to a tuple in
another relation.
Constraints are another important aspect where users define conditions that
must be hold by any instance of the database. DBMS enforce contraints.

## Data Manipulation Languages (DML)

There are mainly two methods two store and retrieve information from a
database:


*Procedural*: Uses relational algebra, where the query specifies in a
high-level the strategy to find the desired result. Mostly commonly used.

*Non-Procedural:* Relational calculus. The query specifies only what data is
wanted, and not how to find it.

## Relational Algebra

Defines the algebra on how to interact with the data. These are the fundamental
operations to retrieve and manipulate tuples in a relation.

_Selection (σ)_: 

