# Concurrency

The virtual matrices are designed to be **thread-safe to read**, but reading one
matrix concurrently does **not** make it faster. This page explains which solver
work serializes, what that means for multithreaded code, and why the per-arc
solve loop is not threaded.

## Concurrent `getindex` is safe but serialized per matrix

Indexing a [`VirtualPTDF`](@ref), [`VirtualLODF`](@ref), or [`VirtualMODF`](@ref)
from several threads produces correct results with no data races. Each virtual
matrix owns one factorization and one **per-cache `ReentrantLock`** that guards
its row cache and its solver scratch buffers. Cache lookups and inserts, and the
factorization solve itself, happen under this lock.

The consequence is that the expensive part of a row computation — the KLU solve —
runs one at a time **per matrix**. Concurrent readers of the same matrix get
correct answers, but they queue: the throughput of `N` threads all missing the
cache of one matrix is essentially the throughput of one. **Do not expect a
parallel speedup from threading reads of a single matrix.** The value of the
locking is safety and correctness under concurrency, not scaling.

Different matrices do not share a lock. Solves on separate matrices (for example,
one `VirtualPTDF` per system, each used by its own task) run in parallel.

The cache-miss path is written to hold locks as briefly as correctness allows.
The shared `cached_row_lookup` pattern (`src/row_cache.jl`) takes the cache lock
to test for a hit, runs the row computation, then takes the lock again to insert,
double-checking for a row a concurrent producer may have inserted in the
meantime. The compute itself is still serialized — through the per-cache solver
lock — because that is where the libklu work lives.

## Why solves on one factorization must be serialized

A KLU factorization is not just read during a solve: `klu_solve` uses workspace
stored inside the numeric factorization, and the wrapper reuses per-cache scratch
buffers. Two overlapping solves on the same factorization would overwrite each
other's intermediate results. The per-cache lock prevents that.

Separate factorizations are independent: libklu keeps no global state, so calls
on distinct `Numeric`/`Symbolic`/`Common` objects may run at the same time. Earlier
versions of this package additionally serialized every libklu call in the process
through a global `_LIBKLU_LOCK`, after intermittent corruption (`KLU_INVALID`,
segfaults in `klu_solve`) was observed with independent caches. That corruption
was a use-after-free, not a libklu race: a `deepcopy` of a virtual matrix (made by
PowerSimulations during model construction) copied the raw factorization pointers,
and when the original was garbage-collected its finalizer freed the factorization
the copy was still solving with. A `deepcopy` of a KLU cache is now refused with an
error; `scripts/diagnostics/klu_deepcopy_use_after_free.jl` reproduces the failure
without the guard and shows that the guard prevents it. The global lock was then
removed. Stress tests with independent caches on concurrent threads, including
refactors and finalizers of dropped caches running alongside solves, show no
corruption.

The Apple Accelerate backend (`src/AccelerateWrapper/`) is guarded the same way,
by the per-cache `solver_lock`.

## Process exit while tasks are still solving

When Julia exits, it runs every pending finalizer, including those of caches that
tasks on other threads are still solving with. Freeing a factorization then is a
use-after-free (the crash reported in KLU.jl#31). The cache finalizer therefore does nothing once an `atexit` hook
has marked the process as exiting, and the operating system reclaims the memory.
Even so, wait for spawned tasks (`@sync`, `wait`, or `fetch`) before returning
from a script: Julia does not wait for them, so their work is silently cut off.

## Why the per-arc solve loop is not threaded

Building a PTDF or answering a set of contingency queries means one linear solve
per arc (or per contingency) against the factorized `ABA`. It is tempting to
parallelize that loop. It does not pay off here, for two independent reasons:

 1. **One factorization cannot serve concurrent solves.** Parallel solves need a
    separate factorization (or at least separate numeric workspace) per thread,
    which multiplies the memory held by each matrix.
 2. **The query pattern is incremental, not batched.** Sienna consumers ask for
    rows one arc or one contingency at a time, as a study progresses. There is no
    point at which a large batch of right-hand sides is available to solve
    together, so a multi-RHS reformulation — the usual way to extract parallelism
    from a factorized system — does not fit the access pattern.

Because of both, threading the solve loop is not a lever this package offers. The
realized performance work went into making each individual build cheaper (for
example, the Ybus adjacency assembly), not into running solves in parallel.

## Practical guidance

  - **Reading virtual matrices from multiple threads is safe.** Correctness is
    guaranteed; use it when a thread happens to need a row.
  - **Do not thread reads of one matrix hoping for solver speedup.** Its solves
    serialize on the per-cache lock; more threads means more queueing, not more
    throughput.
  - **Independent work can overlap.** Separate matrices, separate KLU caches, and
    your surrounding non-solver logic are not serialized against each other.
    Parallelism is worth pursuing above the solve, in how you organize a study,
    rather than inside it.
  - **Wait for spawned tasks before the program ends.**

For the solver backends themselves, see the
[choose-a-linear-solver how-to](../how_to_guides/generated_choose_linear_solver.md).
