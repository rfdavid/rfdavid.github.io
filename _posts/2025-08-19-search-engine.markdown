---
layout: post
author: Rui F. David
title:  "Bulding a Search Engine From Scratch"
date:   2120-05-13 05:27:00 -0400
usemathjax: true
published: false
categories: software engineering
toc: true
---


## TODO

- CC Multithreaded multiplexed downloader
- HTML to markdown parser
- Sentencizer/Chunking
- Storage Database / KV DB to store the text content
- Embedding server
- Vector DB
- Crawler

## Architecture

- Crawler
  - Pull a list of urls from a queue
  - Perform a traversal
  - Push the urls to a queue
  - parse html to markdown            <----- same as CC worker
  - Chunking
  - Store

(same machine)
- Create a fast CC downloader & parser
  - Multithread/mutiplexed range downloader
  - Parse html to markdown
  - Chunking
  - Store
  - Generate embeddings somewhere and store back in a vector DB

- Create embedding server
  - Pull chunks from the store
  - Embed
  - Store in vector db

Research: how do I generate the embeddings?

## Introduction

Project: Building an Agentic Search Engine from scratch with X billions neural
embeddings

Heavily inspired by the amazing work from Wilson L:
https://blog.wilsonl.in/search-engine


Instead of a crawler/indexer that just retrieves and ranks pages, you embed a reasoning agent loop into the search process itself:

	1.	Comprehend
	•	Don’t just keyword-match or embed every page.
	•	Actually read/understand snippets (with a transformer or smaller LLM) and decide: is this relevant? trustworthy? spam?
	•	Example: skip pages that are AI-generated SEO slop, promote ones that have original reporting.

	2.	Filter
	•	Actively decide which branches of the crawl frontier are worth exploring.
	•	Instead of blindly following every link on cnn.com, the agent can say:

“This sports subsection isn’t relevant to my index focus; let’s allocate bandwidth to investigative journalism pages instead.”

	3.	Rank
	•	Beyond PageRank-style link graphs or hand-crafted features, you let the agent weigh signals: writing quality, originality, author credibility, citation network.
	•	Transformers can model this holistically (“does this text sound like authoritative reporting?”).

	4.	Beam search instead of simple retrieval
	•	Today: query → embed → nearest neighbors in vector DB.
	•	Agentic: query → generate hypotheses → explore multiple “beams” of reasoning (e.g., related entities, possible answers, related perspectives).
	•	Fetch supporting docs, check consistency, prune beams that don’t hold up.
	•	This is closer to how humans research: you don’t just type once and accept the top-10; you explore, discard, and refine.





### Crawling

https://www.firecrawl.dev
https://github.com/msgpack/website

How to distribute the crawlers?

- Cralwers are network bounded, not CPU bounded

Concrete C++ stack options
	•	HTTP/1.1 & HTTP/2: libcurl (multi API) + nghttp2; or Boost.Beast for HTTP/1.1 and nghttp2 for H2.
	•	HTTP/3/QUIC (optional): quiche or msquic if you need H3 to reduce head-of-line blocking.
	•	Async runtime: Boost.Asio (proactor) with coroutines (C++20 co_await) for clean “one-URL-per-task” semantics.
	•	DNS: c-ares for async lookups; enforce “public-IP only” to prevent SSRF (the author manually resolved and filtered to avoid private IPs).  ￼
	•	Compression: brotli + zlib; stream in fixed chunks; cap max page bytes/time.
	•	Scheduler: per-origin token buckets + jitter; exponential backoff on 429/5xx; exclude rate-limited origins and requeue later (author’s pattern).


Leveraging Common-crawl data: https://commoncrawl.org/blog/index-to-warc-files-and-urls-in-columnar-format

Query from Common Crawl Index table all the english websites using Amazon Athena:
https://github.com/commoncrawl/cc-index-table#query-the-table-in-amazon-athena
https://github.com/commoncrawl/cc-index-table/blob/main/src/sql/examples/cc-index/get-records-for-language.sql

Total pages (eng):

SELECT count(url)
FROM "ccindex"."ccindex"
WHERE crawl = 'CC-MAIN-2025-33'
  AND subset = 'warc'
  AND content_languages = 'eng';

951,641,302


SELECT url,
       warc_filename,
       warc_record_offset,
       warc_record_length
FROM "ccindex"."ccindex"
WHERE crawl = 'CC-MAIN-2025-33'
  AND subset = 'warc'
  AND content_languages = 'eng'
ORDER BY warc_filename, warc_record_offset;

I'm trying to leverage common crawl data. However, It is not easy to come up
with some cheap way to download everything. First, I filtered out to only fetch
english pages, which corresponds to 951 million pages. The index of all pages
are stored in a file of 300GB.

My second goal is to process this fast. So my idea is to order this index by
file name and record offset, and then extract the ranges I have to download. In
this way, I avoid to perform one request for each record.
I can either download from EC2 instance at the same region or external. The
advantage of doing external is the cost free. From EC2, although the data
transfer is free, there is a cost per request which is $0.0004 per 1000
requests. This seems low, but for 951 records / request, that would cost $380.

Another alternative is to download the whole file and process only what is
interesting, reading the files using io_uring. However, I usually need only
10%-20% of each file, so it will be waste of network and disk.

Yet, EC2 is not an option since I need to perform chunking and generate
embeddings outside EC2. Network cost would be very expensive as well.


Crawling is still a challenge and has lots of open questions.


### Parsing

Parsing is a very challenged task. I have thought about some solutions:
- Using a small LLM model to extract and convert to markdown. I tried Jina readerV2 but it outputs a lot of garbage.
- Using a deterministic library that parses the html

I first strived to correctness before thinking about performance. I tried the
following. As a reference, I used a SEC Form 8-K from a public traded company
which is very dense, containing multiple tables and information.

- Turndown.js and html-to-markdown (go): tables were broken
- htmd: rust built version inspired by turndown. Tables were rendered wrongly
- firecrawl and markdownify: rendered decently but tables still contain errors, not perfectly
  as html. For example, some '(' were offset to a different column
- Jina: very good human output

- Custom beautifulsoup: broken tables

  https://www.youtube.com/watch?v=QxHE4af5BQE


Headless chrome is used by google but it heavy in resources. Servo is a potential
project but it still in academic. I believe a good potential project is to
build a headless browser that consumes less resources.


### Queue

We need a queue to pull the urls from the global queue and process the URLs.

- We have a fixed number of consumers and producers (let's say 1000 nodes)
- However, the number of writes is different from the reads (calculate)
- Based on this information, check the best implementation: atomic queue vs
  locks

Reference:

https://github.com/max0x7ba/atomic_queue?tab=readme-ov-file
https://max0x7ba.github.io/atomic_queue/html/benchmarks.html

### Chunking

Late chunking:
https://jina.ai/news/late-chunking-in-long-context-embedding-models/

### Embeddings

DistilBERT
https://huggingface.co/distilbert/distilbert-base-uncased

Static Embeddings:
https://huggingface.co/blog/static-embeddings

### Querying & Search

### Inference



### DNS

https://github.com/coredns/coredns

### Databases

https://github.com/facebook/rocksdb/wiki/BlobDB

### Information Retrieval
https://github.com/gabriben/awesome-generative-information-retrieval?tab=readme-ov-file


### Services

Cheap GPUs
https://vast.ai

### Other

https://commoncrawl.org/use-cases
https://opensearch.org

### References

Search Engine Course
https://www.youtube.com/channel/UCZvIjWUXzBTr0Brm4qT4rBA/playlists

Sparse embedding or BM25?
https://medium.com/@infiniflowai/sparse-embedding-or-bm25-84c942b3eda7

Hybrid Search Database
https://infiniflow.org

Click House Hot Cache System
https://clickhouse.com/blog/building-a-distributed-cache-for-s3?trk=feed_main-feed-card_reshare_feed-article-content

Show and Tell:
- EXA discord
- Hackernews
- LinkedIn
- Product Hunt ?

