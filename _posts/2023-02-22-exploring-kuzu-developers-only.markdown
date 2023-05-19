---
layout: post
author: Rui F. David
title:  "Exploring Kùzu Graph Database Management System code"
date:   2023-02-22 12:51:31 -0400
usemathjax: false
categories: software engineering
---

## Introduction

[Kùzu](https://https://kuzudb.com) is a Graph Database Manaagement System born 
after extensive research conducted over several years at University of Waterloo. 
Kùzu is highly optimized to handle complex join-heavy
analytical workloads on very large databases. It is similar to what
[DuckDB](https://duckdb.org/) is doing for SQL. It is extremely useful when you
need to model your data as a graph from different sources and store it in one
place for fast extraction in analytics. Kùzu has integration with Pytorch
Geometric, making it easy to extract graph data and feed it into your PyG models
to perform a GNN task.
This article contains my annotations from when I started exploring how Kùzu database
works. I took a 'depth limited search' approach exploring the code by first
going to the CLI and running a simple query. I used LLDB to debug and learn
more about the overall design of the database.

## Starting from the embedded shell

Starting from the CLI tool, the purpose is to track what is happening
internally from the initialization to a match query.

Kùzu uses [args library](https://github.com/Taywee/args) to parse the arguments.
`#include "args.hxx"`. For instance,  database path (-i parameter) can be
retrieved by:

{% highlight c++ %}
auto databasePath = args::get(inputDirFlag);
uint64_t bpSizeInMB = args::get(bpSizeInMBFlag);
{% endhighlight %}


Initialize default bufferPoolSize as -1u bit mask:
`uint64_t bpSizeInBytes = -1u;`.


### SystemConfig

shell_runner.cpp: SystemConfig systemConfig(bpSizeInBytes);

SystemConfig will initialize 4 variables:

* **systemMemSize**: total memory in the system. This is accomplished by mutiplying
                  the number of pages of physical memory by the size of a page in bytes. Both
                  values are retrieved using sysconf from unistd.h library.

{% highlight c++ %}
database.cpp:
   24           auto systemMemSize =
-> 25               (std::uint64_t)sysconf(_SC_PHYS_PAGES) * (std::uint64_t)sysconf(_SC_PAGESIZE);

(lldb) p systemMemSize
(unsigned long long) $9 = 34359738368

{% endhighlight %}


```
_SC_PHYS_PAGES : the number of pages of physical memory
_SC_PAGESIZE : size of a page in bytes
```


* **bufferPoolSize**: defined by the system memory or UINTPTR_MAX x default
pages buffer ratio. UINTPTR_MAX is the larges value uintptr_t can hold. StorageConfig
is located at include/common/configs.h and contains the struct with many default values used by the application.


{% highlight c++ %}
-> 26           bufferPoolSize = (uint64_t)(StorageConfig::DEFAULT_BUFFER_POOL_RATIO *
   27                                       (double_t)std::min(systemMemSize, (std::uint64_t)UINTPTR_MAX));
{% endhighlight %}

* **defaultPageBufferPoolSize and largePageBufferPoolSize**: the bufferPoolSize
multiplied by the ratio defined for default pages and large pages.

{% highlight c++ %}
   29       defaultPageBufferPoolSize =
-> 30           (uint64_t)((double_t)bufferPoolSize * StorageConfig::DEFAULT_PAGES_BUFFER_RATIO);
   31       largePageBufferPoolSize =
   32           (uint64_t)((double_t)bufferPoolSize * StorageConfig::LARGE_PAGES_BUFFER_RATIO);
{% endhighlight %}

{% highlight c++ %}
include/common/configs.h:
struct StorageConfig {
    // The default ratio of system memory allocated to buffer pools (including default and large).
    static constexpr double DEFAULT_BUFFER_POOL_RATIO = 0.8;
    // The default ratio of buffer allocated to default and large pages.
    static constexpr double DEFAULT_PAGES_BUFFER_RATIO = 0.75;
    static constexpr double LARGE_PAGES_BUFFER_RATIO = 1.0 - DEFAULT_PAGES_BUFFER_RATIO;
    ... (omitted)
};

(lldb) p largePageBufferPoolSize/(1024*1024*1024)
(unsigned long long) $28 = 6
(lldb) p defaultPageBufferPoolSize/(1024*1024*1024)
(unsigned long long) $29 = 19
{% endhighlight %}

* **maxNumThreads**: the number of concurrent threads supported by the available
hardware. This number is only a hint and might not be accurate.

{% highlight c++ %}
(lldb) p maxNumThreads
(uint64_t) $30 = 12
{% endhighlight %}

### Embedded Shell

Initialize an instance of EmbddedShell (tools/shell/embedded_shell.cpp):

{% highlight c++ %}
tools/shell/shell_runner.cpp:
-> 33           auto shell = EmbeddedShell(databasePath, systemConfig);
{% endhighlight %}


tools/shell/embedded_shell.cpp:
{% highlight c++ %}
   201  EmbeddedShell::EmbeddedShell(const std::string& databasePath, const SystemConfig& systemConfig) {
-> 202      linenoiseHistoryLoad(HISTORY_PATH);
   203      linenoiseSetCompletionCallback(completion);
   204      linenoiseSetHighlightCallback(highlight);
   205      database = std::make_unique<Database>(databasePath, systemConfig);
   206      conn = std::make_unique<Connection>(database.get());
   207      updateTableNames();
   208  }
{% endhighlight %}

Initialize the embedded shell using the databasePath from the parameter and
also the systemConfig previously defined:

{% highlight c++ %}
(lldb) p systemConfig
(const kuzu::main::SystemConfig) $31 = {
  defaultPageBufferPoolSize = 20615843020
  largePageBufferPoolSize = 6871947673
  maxNumThreads = 12
}
{% endhighlight %}

[linenoise](https://github.com/antirez/linenoise) is a lightweight library for
editing line, providing useful functionalities such as single and multi line
editing mode, history handling, completion, hints as you type, among others.
It is used in Redis, MongoDB and Android. The library is embedded in the
codebase (tools/shell/linenoise.cpp). I won't get into the details of
linenoise configuration.


{% highlight c++ %}
-> 205      database = std::make_unique<Database>(databasePath, systemConfig);
-> 206      conn = std::make_unique<Connection>(database.get());
{% endhighlight %}

database and conn are both defined in `embedded_shell.h`:

{% highlight c++ %}
private:
    std::unique_ptr<Database> database;
    std::unique_ptr<Connection> conn;
};
{% endhighlight %}

Line 205 and 206 define the database and get the current connection,
respectively. Before getting into connection in the next section, I'll take a
look at the `updateTableNames()`, since now we are dealing with catalogue to read
the database schema.


### updateTableNames()

There are two type of tables: node and relations. updateTableNames will store
the table names for both by fetching from database->catalog. In my database, I
have "person" and "animal" node tables and "hasOwner" and "knows" relations
tables:

{% highlight c++ %}
tools/shell/embedded_shell.cpp:

   67   void EmbeddedShell::updateTableNames() {
   68       nodeTableNames.clear();
   69       relTableNames.clear();
-> 70       for (auto& tableSchema : database->catalog->getReadOnlyVersion()->getNodeTableSchemas()) {
   71           nodeTableNames.push_back(tableSchema.second->tableName);
   72       }
   73       for (auto& tableSchema : database->catalog->getReadOnlyVersion()->getRelTableSchemas()) {
   74           relTableNames.push_back(tableSchema.second->tableName);
   75       }
   76   }
{% endhighlight %}

lldb output:

{% highlight c++ %}
(lldb) p nodeTableNames
(std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >) $41 = size=2 {
  [0] = "person"
  [1] = "animal"
}
(lldb) p relTableNames
(std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >) $42 = size=2 {
  [0] = "hasOwner"
  [1] = "knows"
}
{% endhighlight %}


### Connection (src/main/connection.cpp)

Connection is used to interact with a Database instance, and each Connection is thread-safe.
Multiple connections can connect to the same Database instance in a multi-threaded environment.
The description of the API below was extracted from `src/include/main/connection.h`:

Creates a connection to the database.
{% highlight c++ %}
KUZU_API explicit Connection(Database* database);
{% endhighlight %}

Destructor
{% highlight c++ %}
KUZU_API ~Connection();
{% endhighlight %}

Manually starts a new read-only transaction in the current connection.
{% highlight c++ %}
KUZU_API void beginReadOnlyTransaction();
{% endhighlight %}

Manually starts a new write transaction in the current connection.
{% highlight c++ %}
KUZU_API void beginWriteTransaction();
{% endhighlight %}

Manually commits the current transaction.
{% highlight c++ %}
KUZU_API void commit();
{% endhighlight %}

Manually rollbacks the current transaction.
{% highlight c++ %}
KUZU_API void rollback();
{% endhighlight %}

Sets the maximum number of threads to use for execution in the current connection.
{% highlight c++ %}
KUZU_API void setMaxNumThreadForExec(uint64_t numThreads);
{% endhighlight %}

Returns the maximum number of threads to use for execution in the current connection.
{% highlight c++ %}
KUZU_API uint64_t getMaxNumThreadForExec();
{% endhighlight %}

Executes the given query and returns the result.
{% highlight c++ %}
KUZU_API std::unique_ptr<QueryResult> query(const std::string& query);
{% endhighlight %}

Prepares the given query and returns the prepared statement.
{% highlight c++ %}
KUZU_API std::unique_ptr<PreparedStatement> prepare(const std::string& query);
{% endhighlight %}

Executes the given prepared statement with args and returns the result.
{% highlight c++ %}
KUZU_API template<typename... Args>
inline std::unique_ptr<QueryResult> execute(
    PreparedStatement* preparedStatement, std::pair<std::string, Args>... args) {
    std::unordered_map<std::string, std::shared_ptr<common::Value>> inputParameters;
    return executeWithParams(preparedStatement, inputParameters, args...);
}
{% endhighlight %}

Executes the given prepared statement with inputParams and returns the result.
{% highlight c++ %}
KUZU_API std::unique_ptr<QueryResult> executeWithParams(PreparedStatement* preparedStatement,
    std::unordered_map<std::string, std::shared_ptr<common::Value>>& inputParams);
{% endhighlight %}

Return all node table names in string format.
{% highlight c++ %}
KUZU_API std::string getNodeTableNames();
{% endhighlight %}

Return all rel table names in string format.
{% highlight c++ %}
KUZU_API std::string getRelTableNames();
{% endhighlight %}

Return the node property names.
{% highlight c++ %}
KUZU_API std::string getNodePropertyNames(const std::string& tableName);
{% endhighlight %}

Return the relation property names.
{% highlight c++ %}
KUZU_API std::string getRelPropertyNames(const std::string& relTableName);
{% endhighlight %}


If you wondering what is behind KUZU_API, the datatype is defined in src/include/common/types/types.h:

{% highlight c++ %}
KUZU_API enum DataTypeID : uint8_t {
    ANY = 0,
    NODE = 10,
    REL = 11,

    // physical types

    // fixed size types
    BOOL = 22,
    INT64 = 23,
    DOUBLE = 24,
    DATE = 25,
    TIMESTAMP = 26,
    INTERVAL = 27,

    INTERNAL_ID = 40,

    // variable size types
    STRING = 50,
    LIST = 52,
};
{% endhighlight %}

to be continued...
