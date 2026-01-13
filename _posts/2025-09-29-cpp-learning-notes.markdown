---
layout: post
author: Rui F. David
title:  "C++ Random Learning Notes"
date:   2025-10-15 00:27:00 -0400
draft: true
usemathjax: true
categories: software engineering
toc: true
---

<br>

> Note: This is still work in progress.


There is a plenty of (free) resources available to learn C++. Here I'm trying to consolidate
a few learnings and notes that I found it useful for myself. Most of them are
curiosity that I had throughout my journey and I decided to write about it. I
believe this is a good way to consolidate and review the content.
Everything here is based on multiple sources, including books, courses, and articles.
I have been perusing some files from libc++ (LLVM's C++ standard library used
by clang) {% cite LLVM %} and libstdc++ (GCC's library) {% cite GNU_library %} to understand how things work under the hood.
I called it random because there is no specific order or structure to the topics covered.
Sometimes I just feel like writing about something I found interesting.

## ABI

ABI stands for Application Binary Interface. It is a low-level interface between two program modules,
often between compiled code and the operating system or between different compiled modules.


## Stack and Heap Allocation in C++

In C++, there are two main ways to allocate memory for variables: stack allocation and heap allocation.

### Stack Allocation

Stack allocation is used for local variables and function parameters.
Usually, when a function is invoked, its arguments are typically passed using registers.
If there are more arguments than available registers or if specified by the Application Binary Interface (ABI),
they may be passed using the stack. In this case, a new stack frame is created on the call stack
to hold the function's local variables and parameters. The stack frame is automatically
cleaned up when the function returns, and the memory is reclaimed. {% cite reductors_blog_2023 %}

### Heap Allocation

Heap allocation is used for dynamic memory allocation, where memory is
allocated at runtime. One way to allocate memory on the heap is by using the `new` operator.
When you use `new`, it allocates memory from the heap and returns a pointer to the allocated memory.

## RAII

RAII stands for Resource Acquisition Is Initialization. It is an idiom where you tie the lifetime of a resource
to the lifetime of an object. This a pattern that you must implement.
The core idea is:

- The application acquire resources (memory, file handles, sockets, etc.) in the constructor of an object.
- The application releases resources in the destructor of the object.
- This guarantees that resources are properly released when the object goes out of scope, even in the presence of exceptions.

### Example

```cpp
class FileHandler {
private:
    FILE* fp;

public:
    FileHandler(const char* filename) {
        fp = fopen(filename, "r");
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }
    }

    // close the file, freeing the resource when
    // the object goes out of the scope
    ~FileHandler() {
        if (file) {
            fclose(fp);
        }
    }

    // prevent copying to avoid double-free
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};
```

## Virtual Tables

Dispatching refers to finding the right function to call. When you define a
method inside a class, the compiler will execute every time you call that method.
However, if you define a method as `virtual`, the compiler will create a table
since it is not known at compile time which method to call. This table is called 'vtable` or `virtual table`.
It's how C++ implements polymorphism. Each class with virtual functions has its own vtable.

Example:
```cpp
class Animal {
    public:
        virtual void speak() { ... }
        virtual void move() { ... }
};

class Dog : public Animal {
    public:
        void speak() override { ... }
};

```

When you call `animal_ptr->speak()`, the compiler will:
1. Look up the vtable for the `Animal` class.
2. Find the entry for the `speak` method.
3. Call the `speak` method for the `Dog` class.


## Move Semantics and Perfect Forwarding

C++ 11 introduced move semantics which was one of the most influential
features. The fountation of move semantics is distinguishing expressions that
are rvalues from those that are lvalues.
`lvalue` (left value) is an expression that has an identifiable memory location.
You can take its address. `lvalue` has a name you can refer to later.
On the other hand, `rvalue` is a temporary value that doesn't have a persistent memory location.
You cannot take its address. `rvalue` is temporary, about to be destroyed.

**Move semantics** allow compilers to be able to replace expensive copy operations
with less expensive moves. Move constructors and move assignment operators
offers control over semantics of moving. It also enables the creation of move-only types: `std::unique_ptr`,
`std::thread`, `std::promise`, `std::mutex`, `std::lock_guard`, `std::fstream`, etc.

**Perfect forwarding** allows you to write function templates that forward
their arguments to another function while preserving the value category (lvalue or rvalue) of those arguments.
Consider the following problem:

```cpp
void process(std::string& s) { ... }      // (1) lvalue reference overload
void process(std::string&& s) { ... }     // (2) rvalue reference overload

template <typename T>
void wrapper(T arg) {
    process(arg); // Problem: arg is always an lvalue here
}

std::string str = "Hello";
wrapper(str);                  // Calls process(std::string& s) - correct
wrapper(std::string("World")); // Also calls process(std::string& s) - wrong
```

In this example, the temporary `std::string("World")` is an rvalue, but when passed to `wrapper`, it becomes an lvalue.
Perfect forwarding solve this problem with `std::forward<T>(arg)`. It casts `arg` back to its original value category.

```cpp
template<typename T>
void wrapper(T&& arg) {            // T&& is a forwarding reference
    process(std::forward<T>(arg)); // Perfectly forwards arg
}

wrapper(str);                   // Calls lvalue - correct
wrapper(std::string("World"));  // Calls rvalue - correct
```

The template parameter `T&&` bind to both lvalues and rvalues, making it a forwarding reference. {% cite CPP_and_Beyond_2011 %} {% cite Chen_2023 %}

### std::move()

`std::move` is a standard library function that performs an unconditional cast to an rvalue {% cite LIBSTDC_move %}.


## Function Objects

## Smart Pointers

What is the problem with raw pointers? If you allocate memory using `new`, you need to remember to `delete` it.
That is the first thing that comes to mind. However, there are other problems with raw pointers:
- Its declaration doesn't indicate if it points to a single object or an array.
  If you use object-form `delete` on an array, the behavior is undefined.
- Difficult to ensure you that you perform the destruction exactly once along
  every path in te code.
- You may have pointers to objects that have already been destroyed (dangling
  pointers), and there is no way to tell if the pointer dangles.

Smart pointers are one way to address these issues. C++ 11 introduced smart
pointers, which are wrappers around raw pointers. There are three types of
smart pointers {% cite Meyers_2015 %}:

### std::unique_ptr

`unique_ptr` is a smart pointer that owns a dynamically allocated object exclusively. It makes sure
that only one copy of the object exists. They are the same size of raw pointers
and for most of the operations they execute exactly the same instructions. An
unique pointer can be moved but cannot be copied.

### std::shared_ptr

An object accessed via `shared_ptr`s has its lifetime managed by the objects point to it via reference counting.
When the last `shared_ptr` pointing to an object is destroyed or reset, the object is deleted. This is similar
to garbage collection in other languages, but it is deterministic and happens immediately when the last reference is gone.

### std::weak_ptr

This is a special case smart pointer that doesn't affect the reference count of a `shared_ptr`.
It holds a non-owning ("weak") reference to an object that is managed by `shared_ptr`, and it doesn't
increase the reference count.

C++ 11 also introduced `std::auto_ptr` but it was completely deprecated.

### Writing a custom smart pointer

Before writing custom smart pointers, I want to explain some underlying
concepts behind some chunks of code.

#### Custom Unique Pointer

We want to create an object that takes ownership of another object preserving
some semantics.

```cpp
UniquePtr<MyClass> myClass(new MyClass());
UniquePtr<int> objPtr(new int());
```

In this case, myClass is a unique pointer that wraps up a raw pointer returned from
`new MyClass()`.

> Fun fact: `new int()` initializes with 0 whereas `new int` initializes with garbage.

So we start our custom unique pointer class with the
following:

```cpp
template<typename T>
class UniquePtr {
private:
    T* ptr;

public:
    explicit UniquePtr(T* p) : ptr(p) {}
};
```

We simply create a pointer `ptr` that will hold the raw pointer. The constructor
will take the pointer returned from `new` and store it in `ptr`.

**Why the explicit keyword?**

First, we have to understand implicity conversions. Consider the following:

```cpp
UniquePtr<int> p(new int(42)); // explicit
UniquePtr<int> p1 = new int(42); // implicit
```

For the implicit case, the compiler will automatically convert the `new int(42)`.
Behind the scenes, this should be equivalent to:

```cpp
UniquePtr<int> p1 = UniquePtr<int>(new int(42));
//                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                  implicit conversion happening here
```

**So, why bother?**

This may cause some subtle problems regarding ownership. For example:

```cpp
int* rawPtr = new int(42);
UniquePtr<int> objPtr = rawPtr;
*rawPtr = 100;
```

Here, `rawPtr` is created and the whole idea of unique pointer is to take
control of this object. However, it is still acessible and can even be
modified.

Another issue is to accidentally give away ownership. Example:

```cpp
void process(UniquePtr<int> u) {
    // deletes when done
}

int* rawPtr = new int(42);
process(rawPtr);
```

`process` takes a `UniquePtr<int>` as a parameter, but in the example, it is
being passed a raw pointer. The compiler will implicitly convert the raw pointer
to a `UniquePtr<int>`, and the ownership of the raw pointer is given to the
`process` function. When `process` finishes, it will delete the pointer, and
`rawPtr` will become a dangling pointer.

When using the keyword `explicit`, the compiler will not allow implicit conversions,
raising an error during compilation.

**Copy semantics**

Unique pointer cannot be copied. So the copy constructor and copy assignment
must be deleted. You want to avoid the following situation:


```cpp
UniquePtr<int> p1(new int(42));
UniquePtr<int> p2(p1); // copy constructor
UniquePtr<int> p3 = p1; // this is also a copy constructor, NOT a copy assignment

p3 = p1; // this is a copy assignment
```

You shouldn't be allowed to write the code above since unique pointers are
obviously unique. Note that the second example `UniquePtr<int> p3 = p1` is also a copy
constructor, not a copy assignment. This is because `p3` is being created in this
line, so it is a copy constructor.
So we delete the copy constructor and copy assignment operator:

```cpp
UniquePtr(const UniquePtr&) = delete;
```

**Why const?** : reference const in C++ will bind both to const and non-const,
whereas without const would miss const objects.
**Why not `UniquePtr& something`?** : because you are deleting, so it doesn't
make any difference.

For the copy constructor (in our example `p3 = p1`), we delete by defining the
following:

```cpp
UniquePtr& operator=(const UniquePtr&) = delete;
```

From the previous example, this captures the reference from the left side (`p3`)
and the right side (`p1`) by const reference, as we explained before.

**Move semantics**

We want to be able to move unique pointers. Moving means transferring ownership
from one unique pointer to another. This is done using move constructor and
move assignment operator. In this example, we initialize a unique pointer with
integer object and then we transfer the ownership to the new created unique
pointer p2.

```cpp
UniquePtr<int> p1(new int(42));
UniquePtr<int> p2(std::move(p1));
```

In this case, `p2` should now take ownership of our defined pointer `T* ptr`,
and `p1` should be left in an empty state. Here is how we should handle the
move constructor:

```cpp
// move constructor
UniquePtr(UniquePtr&& other) : ptr(other.ptr) noexcept {
    other.ptr = nullptr;
}
```

The rvalue reference `&&` binds to the rvalue produced by `std::move()`,
allowing the move constructor to transfer ownership of the resource from the
source object. `std::move()` doesn't move anything, it is only a simple cast that converts
its argument into an rvalue.


For move assingment operator, we want to perform the following:

```cpp
UniquePtr<int> p1(new int(42));
UniquePtr<int> p2(new int(99));
p1 = std::move(p2);
```

It can be implemented as follows:

```cpp
UniquePtr& operator=(UniquePtr&& other) noexcept {
    if (this != &other) {
        delete ptr;
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    return *this;
}
```

We cleaned up the current resource (`delete ptr`)  and replace it by the new
assigned resource. We also set the `other.ptr` to `nullptr` to avoid double-free.

**Destructor** will simply remove the resource when the unique pointer goes out of scope:

```cpp
~UniquePtr() {
    // no need for this if since delete nullptr is safe in C++
    if (ptr != nullptr) {
        delete ptr;
    }
}
```

This is the basic implementation of a custom unique pointer. We can also add
get, release, reset and dereference methods to make it more complete.
For instance, we want to implement the following:

```cpp
uniq_ptr->
*uniq_ptr // de-reference operator
uniq_ptr.get() // returns a raw pointer
```

where these should return the object that belongs to this unique pointer. That
can be implemented as:

```cpp
// implementing opertor '->'
// '->' expects a pointer T*
// ptr is already a pointer, so return ptr
T* operator->() { return ptr; }

// implements de-reference operator *uniq_ptr
// we expect the object, so we return the reference
// we need to dereference ptr by returning *ptr
T& operator*() { return *ptr; }

// simply return this pointer
T* get() const { return ptr; }
```

`reset` destroy the current object and optionally take ownership of a new
object. `release` gives up ownership of the object without destroying it.
They can be implemented as follows:

```cpp
void reset(T* newPtr = nullptr) {
    delete ptr;
    ptr = newPtr;
}

T* release() {
    T* temp = ptr;
    ptr = nullptr;
    return temp;
}
```

We may also want to implement bool and comparison operators. For example:

```cpp
// bool operator
if (uniqPtr) {
...
}

if (uniqPtr == otherPtr) {
...
}
```


```cpp
explicit operator bool() const { return ptr != nullptr; }
bool operator==(const UniquePtr& other) const {
    return ptr == other.ptr;
}
```

Here is the complete example:


```cpp
template<typename T>
class UniquePtr {
private:
   T* ptr;

public:
    // Initialization
    explicit UniquePtr(T* p) : ptr(p) {}

    // Destructor
    ~UniquePtr() {
        delete ptr;
    }

    // Copy semantics: unique pointers cannot be copied
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // Move semantics
    UniquePtr(UniquePtr&& other) : ptr(other.ptr) noexcept {
       other.ptr = nullptr;
    }

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    // release semantics: giev up ownership of the current object
    // and return the released object pointer
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }

    // reset semantics: destroy the owned object and replace
    // by another if passed
    void reset(T* newPtr = nullptr) {
        delete ptr;
        ptr = newPtr;
    }

    // other operators
    T* operator->() { return ptr; }
    T& operator*() { return *ptr; }
    T* get() { return ptr; }

    // comparison
    explicit operator bool() const { return ptr != nullptr }
    bool operator==(const UniquePtr& other) const {
        return ptr == other.ptr;
    }
};
```

#### std::make_unique

Consider the following example:

```cpp
auto* obj = new int(42);
std::unique_ptr<int> p1(obj);
std::unique_ptr<int> p2(obj);
```

The code above contains the problem of double ownership that can cause
undefined behaviour when trying to delete the same object twice. When `p1` or `p2`
is out of the scope, the destructor will be called and the object will be deleted.
After deletion, the pointer will remain dangling, and when the second unique pointer
tries to delete the same object, it will lead to undefined behavior. When
calling `delete` on a pointer, it frees the memory, but the pointer still
points to the same memory location.

C++ 14 introduced `std::make_unique`, which is a safer and more efficient way to create
unique pointers. It eliminates the possibility of double ownership by ensuring that the
object is created and owned by a single unique pointer from the start.


#### Custom Shared Pointer

The main difference from unique pointers is that now we can have multiple shared pointers pointing
to the same object. The copy semantics are allowed, but we have to keep track
of the reference count. When the last shared pointer pointing to an object is
destroyed or reset, the object is deleted. For the move semantics, it is similar
to unique pointer, where the ownership is transferred.

You can create multiple shared pointers instances by pointing t othe same
object by **copying or assigning** from an existing shared pointer, but never
multiple instances from a raw pointer.

```
// WRONG!
MyClass* raw = new MyClass();
SharedPtr<MyClass> s1(raw);
SharedPtr<MyClass> s2(raw);
```

The examble above created two independent control blocks, so the object will be
deleted twice since we have two difference reference counters.


```cpp
template<typename T>
class SharedPtr {
private:
    T* ptr;
    size_t* ref_count;

public:
    explicit SharedPtr(T* p) : ptr(p), ref_count(new size_t(1)) {}

    // move constructor
    SharedPtr(SharedPtr&& other) noexcept
        : ptr(other.ptr), ref_count(other.ref_count) {
        other.ptr = nullptr;
        other.ref_count = nullptr;
    }

    // move assignment
    // s1 = str::move(s2)
    SharedPtr& operator=(SharedPtr&& other) noexcept {
       if (this != &other) {
            release();
            ptr = other.ptr;
            ref_count = other.ref_count;
            other.ptr = nullptr;
            ther.ref_count = nullptr;
       }
       return *this;
    }

    // copy constructor
    SharedPtr(const SharedPtr& other)
        : ptr(other.ptr), ref_count(other.ref_count) {
        if (ref_count) {
            ++(*ref_count);
        }
    }

    // copy assignment
    // s1 = s2
    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {
            release();
            ptr = other.ptr;
            ref_count = other.ref_count;
            if (ref_count) {
                ++(*ref_count);
            }
        }
        return *this;
    }

    // destroy
    ~SharedPtr() {
        release();
    }

private:
    void release() {
        if (ref_count) {
            --(*ref_count);
            if (*ref_count == 0) {
                delete ptr;
                delete ref_count;
            }
        }
    }

};
```


## Threads

### Mutex

Mutex ("mutual exclusion") is the well-known synchronization primitive to
prevent multiple threads from accessing a shared resource simultaneously. In
C++, mutexes are provided by the `<mutex>` header. The most common mutex types are:
- `std::mutex`: A simple, exclusive access to a shared resource.
- `std::recursive_mutex`: A mutex that allows the same thread to lock it multiple times. It is used when a function
that locks a mutex calls another function that also locks the same mutex.
- `std::timed_mutex`: A mutex that allows threads to attempt to lock it with a timeout.
- `std::shared_mutex`: A mutex that allows multiple threads to read simultaneously but only one thread to write. Good for
read-heavy workloads.
- `std::lock_guard`: A RAII-style wrapper that automatically locks a mutex when created and unlocks it when destroyed.
- `std::unique_lock`: A more flexible RAII-style wrapper that allows manual locking and unlocking of a mutex.

#### Example: simple mutex & lock_guard

```cpp
std::mutex m;
int shared_counter = 0;

void increment() {
    mtx.lock();
    ++shared_counter;
    mtx.unlock();
}

// using lock_guard
// It automatically releases the lock when it goes out of scope
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared_counter;
}
```

#### Example: timed mutex

```cpp
std::timed_mutex mtx;
int shared_counter = 0;

void worker(int id) {
    if (mtx.try_lock_for(std::chrono::milliseconds(100))) {
        // lock acquired
        ++shared_counter;
        mtx.unlock();
    } else {
        std::cout << "Thread " << id << " could not acquire the lock within 100ms\n";
    }
}
```

#### Example: shared mutex

```cpp
std::shared_mutex smtx;
int shared_data = 0;

void reader() {
    smtx.lock_shared();
    // do some read-only operations
    smtx.unlock_shared();
}

void writer() {
    smtx.lock();
    // do some write operations
    smtx.unlock();
}
```

#### Example: unique_lock

```cpp
std::mutex mtx;

void simple_unique_lock() {
    // this locks immediately, like lock_guard
    std::unique_lock<std::mutex> lock(mtx);
    // do some operations
    // you can unlock if needed or it also unlocks when going out of scope
    lock.unlock();
}

void defer_lock_example() {
    // does not lock immediately
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    lock.lock();
}
```



### Atomics

An atomic operation is an operation that appears to execute entirely or not at
all from the perspective of other threads or processes. This means that
intermediate states of the operation are not visible to other threads. C++
provides atomic operations through the `<atomic>` header. Before entering the
usage of this library, lets understand the problem it solves.

Consider th following multithreaded code:

```cpp
volatile int data = 0;
volatile int ready = 0;

// Thread 1: Producer
data = 42;
ready = 1;

// Thread 2: Consumer
while (ready == 0) {}
printf("Data: %d\n", data);
```

There are multiple serious problems with this code, assuming a multithreaded
execution. First, this code is not linearziable {% cite Linearizability %}. The consumer thread may sequence
the operations in a different order. It may read `data` before `ready` is set to
1. Second, there is no guarantee that the writes to `data` and `ready` will be
visible to the consumer thread in the order they were made. This is due to compiler
and CPU optimizations that may reorder instructions for performance reasons.

> Note: I had to use volatile to "fool" the compiler. Otherwise, when compiling
with `-O3` flag, it knows this is not a multithread application and it doesn't
even respect the fence. Volatile means the variable is guaranteed to be read
from the memory every time it is accessed, and not cached in a register.

#### Memory Ordering

#### Example: implementing my own mutex

Usuaully, it is not a good idea to implement your own mutex. At least, I
haven't beaten the performance of the standard library implementation. For
learning purpose, the implementation using spinlock is quite straightforward.
The following example is a mutex lock using spinlock with exponential backoff:

```cpp
class SpinlockMutex {
private:
    std::atomic<bool> locked(false);
    static constexpr int MAX_BACKOFF = 1024;

public:
    void acquire() {
        bool expected = false;
        int backoff = 1;
        while(!locked.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
            expected = false; // reset after failure
            for (int i = 0; i < backoff; ++i) {
                _mm_pause();
            }
            backoff = std::min(backoff * 2, MAX_BACKOFF);
        }
    }

    void release() {
        locked.store(false, std::memory_order_release);
    }
};
```

As a side note, the initialization with `()` won't work on C++ 17 and prior.
Instead, it should be initialized with `{}` (`std::atomic<bool>
locked{false}`).

There are two main interfaces, `acquire` and `release`. The `release` method
simply sets the atomic boolean `locked` to false, indicating that the mutex is
now available. The `acquire` method tries to set `locked` to true using
`compare_exchange_weak`, which is an atomic operation that compares the current
value of `locked` with `expected`. If they are equal, it sets `locked` to true
and returns true. If they are not equal, it updates `expected` with the current
value of `locked` and returns false. That's why we need to reset `expected` to false.
We use a loop that represents our spinlock. If the `compare_exchange_weak` fails,
we perform an exponential backoff by pausing for a certain number of iterations
before retrying. This helps to reduce contention and improve performance under high load.

`_mm_pause()` is an intrinsic that provides a hint to the processor that we
are in a spin-wait loop. It can help improve performance by reducing power consumption
and improving the efficiency of the spin-wait loop.
Intrinsics are low-level functions that provide direct access to CPU instructions, so you
don't have to directly write assembly code. It is a portable way to access specific CPU features.
There is a nice list of [intel intrinsics here](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html),
which includes `_mm_pause()` {% cite Intel_Intrinsics %}.

As I a just bit curious about `_mm_pause()`, this intrinsics translates to `rep nop`,
which is actually the pause instruction.


![large](/assets/images/rep-nop.png "Godbolt rep nop")
_Figure 1: REP NOP instruction on godbolt.org_


#### Example: implementing my own lock_guard


## Lambda Expressions

## Future and Promises

Futures/Promises are a simpler thread-based mechanism for one time async
results. They were introduced in C++ 11 and are components of `<future>` header.
A `std::promise` acts as a producer, and it is responsible for setting the
result of an asynchoronous operation. A `std::future` acts as a consumer, and
it represents a placeholder for a value that will be available at some point in
the future.

**Example:**

```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future(); // this is non-blocking

std::thread t([&prom]() {
    // do some work
    prom.set_value(42);
});

int result = fut.get(); // this will block until the value is set
t.join();
```

## Co-routines

Coroutines were introduce in C++ 20 and allows functions to suspend and resume
their execution. This is useful for asynchronous and concurrent code in a more
sequential manner. Coroutines are defined using:

`co_await`: pauses the coroutine until an awaited operation (often an asynchronous one) completes,
then resumes execution.
`co_yield`: produces a value and suspends the coroutine, allowing it to act as
a generator for sequences of values.
`co_return`: returns a final value and terminates the coroutine


## Template Metaprogramming

### SFINAE

SFINAE stands for "Substitution Failure Is Not An Error".

### C++ Vexing Parse

Apparently, the term "vexing parse" was coined by Scott Meyers in his book
Effective STL {% cite Meyers_2014 %} to describe a specific syntactic ambiguity in C++.
Meyers highlighted how the C++ grammar can mistakenly interpret what looks like
object instantiation as a function declaration.

For example, consider the following code:

```cpp
class Timer {
public:
    Timer() { std::cout << "Timer created\n"; }
};

int main() {
    Timer t1();  // this declares a function, not an object!
                 // function named t1 that takes no parameters
                 // and returns a Timer

    Timer t2;    // This creates an object
}
```

This example may be easy to spot but in more suble scenarios. Consider the
following:

```cpp
std::ifstream file(std::string("data.txt"));
```

It looks like it is creating an object of type `std::ifstream` named `file`,
initialized with a temporary `std::string` object. However, due to the vexing parse,
the compiler interprets this as a function declaration for a function named
`file` that takes a single parameter of type `std::string` and returns an `std::ifstream`.

To avoid the vexing parse, you can use uniform initialization syntax (curly braces):

```cpp
std::ifstream file{std::string("data.txt")};
```

### Spaceship operator

C++ 20 introduced provides an operator called [spaceship operator](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0515r3.pdf) (`<=>`) that allows to write
one function to take care of all comparison operations {% cite ISOCPP %}. This is useful when implementing custom types that need to be compared.
The spaceship operator returns a value that indicates the result of the comparison, which can be one of five categories:
strong_ordering, weak_ordering, partial_ordering, strong_equality, or weak_equality. Let's start with an example before examing each of these:

```cpp
struct IntWrapper {
    int value;
    constexpr In
}
```

https://devblogs.microsoft.com/cppblog/simplify-your-code-with-rocket-science-c20s-spaceship-operator

### Composable Range Views

C++ 20 introduced Standard Library Ranges, which is a way to transform and
filter data without creating intermediate copies. **Views** are lightweight
objects that represent a sequence of elements (a range) without owning the
underlying data.
An interesting aspect of views is composability. You can chain multiple views
using pipe operator to create transformations. Views are designed to be
efficient. They typically don't allocate memory or copy data.

A common example is creating a simple string split function using views:

```cpp
auto split(std::string_view str, char delimiter) {
    return str | std::views::split(delimiter)
               | std::views::transform([](auto&& subrange) {
                    return std::string_view(subrange.begin(), subrange.end());
                });
}

std::string data = "a,b,c";
auto parts = split(data, ',');
for (auto word : parts) {
    // use word
}
```

When the function is called, `std::string` converts data to `std::string_view`.
The string view class is very lightweight. The object is 16 bytes, defined by
[basic_string_view class](https://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen/a00215_source.html)
{% cite LIBSTDC_string_view %}

```cpp
private:
    size_t        _M_len;      // Just a size_t (8 bytes on 64-bit)
    const _CharT* _M_str;      // Just a pointer (8 bytes on 64-bit)
};
```

It contains a size_t (8 bytes on 64-bit) and a `_CharT` pointer (8 bytes on 64-bit). A `_CharT` is
a character type, which can represent a `char`, `wchar_t`, `char8_t`, `char16_t`, or `char32_t`.

An interesting point is that the `split` function returns a view, which is a lazy, meaning no
computation is performed until the view is iterated over.

Similar to Unix pipes, the pipe operator (`|`) is used to chain multiple views together. So `str`
is first passed to `std::views::split`, which creates a view that splits the string by the specified delimiter.
Then, the result is passed to `std::views::transform`, which applies a transformation function to
each subrange produced by the split operation. The transformation function converts each subrange into a `std::string_view`.


## References

{% bibliography --cited %}
