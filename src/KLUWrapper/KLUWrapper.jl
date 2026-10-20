"""
    KLUWrapper

A small, allocation-aware wrapper over `libklu` (provided by `SuiteSparse_jll`)
designed for the access patterns of `PowerNetworkMatrices`:

- Cache the symbolic and numeric factorizations of an SPD/asymmetric sparse
  matrix and reuse them across many solves.
- Refactor (numeric only, or full) without re-allocating.
- Solve dense and **sparse** right-hand sides without materializing N×N
  intermediates when the RHS is structurally sparse.

This module is intentionally lighter than `KLU.jl`: it owns no Julia-side
copies of the matrix values, exposes the symbolic/numeric split directly, and
binds only the `SuiteSparse_long` (`klu_l_*`, `klu_zl_*`) entry points used by
the package.
"""
module KLUWrapper

import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nzrange

"""
    KLU_POOL_DEBUG :: Bool

Compile-time gate for the KLU wrapper's runtime diagnostics — currently
just the precondition snapshot in `solve!` that fires when `klu_l_solve`
returns FALSE. When `false`, the snapshot is folded out by `@static if`
and contributes zero runtime cost. Flip to `true` when debugging an
unexpected `klu_solve` failure.

This is a `const` rather than a `Preferences.@load_preference` flag so
that toggling it forces a precompile rebuild and the production binary
never carries the diagnostic code.
"""
const KLU_POOL_DEBUG = false

"""
    _EXITING :: Threads.Atomic{Bool}

Set by an `atexit` hook. At exit Julia runs every pending finalizer, including
those of caches that tasks on other threads are still solving with, so freeing
the libklu handles then is a use-after-free (SIGSEGV in `klu_l_solve`; see
KLU.jl#31). Once this is set, `_finalize_klu_cache!` leaves the handles for the
OS to reclaim. `atexit` hooks always run before that finalizer sweep.
"""
const _EXITING = Threads.Atomic{Bool}(false)

__init__() = atexit(() -> (_EXITING[] = true))

export KLULinSolveCache,
    klu_factorize,
    symbolic_factor!,
    symbolic_refactor!,
    numeric_refactor!,
    full_factor!,
    full_refactor!,
    solve!,
    tsolve!,
    solve_sparse!,
    solve_sparse,
    sort_factors!,
    condest!,
    rcond!,
    n_valid,
    is_factored,
    get_reuse_symbolic

include("klu_jll_bindings.jl")
include("klu_cache.jl")
include("solve_dense.jl")
include("solve_sparse_rhs.jl")

end # module
