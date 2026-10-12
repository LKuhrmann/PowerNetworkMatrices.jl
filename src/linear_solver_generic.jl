# One function name per factor operation for every cache backend. KLU and MKLPardiso already
# extend the `KLUWrapper` functions. These methods add `AAFactorCache` to them.
# `AAFactorCache` has no transpose solve, so `tsolve!` is KLU-only.

symbolic_factor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.symbolic_factor!(c, A)
symbolic_refactor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.symbolic_refactor!(c, A)
numeric_refactor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.numeric_refactor!(c, A)
full_factor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.full_factor!(c, A)
full_refactor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.full_refactor!(c, A)
solve!(c::AAFactorCache, b::StridedVecOrMat) = AccelerateWrapper.solve!(c, b)
is_factored(c::AAFactorCache) = AccelerateWrapper.is_factored(c)

"""
    make_linear_solver_cache(backend::LinearSolverType, A; snapshot_values = true, kwargs...)

Return an unfactored cache for `A` on `backend`: `KLULinSolveCache` for `KLUSolver`,
`AAFactorCache` for `AppleAccelerateLUSolver`, `PardisoLinSolveCache` for `MKLPardisoSolver`.
`snapshot_values` and `kwargs` go to the KLU constructor. `kwargs` go to the
`AAFactorCache` constructor. `DenseSolver` has no cache and throws an `ArgumentError`.
"""
function make_linear_solver_cache(
    ::KLUSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
    kwargs...,
)
    return KLULinSolveCache(A; snapshot_values = snapshot_values, kwargs...)
end

function make_linear_solver_cache(
    ::AppleAccelerateLUSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
    kwargs...,
)
    return AAFactorCache(A; kwargs...)
end

function make_linear_solver_cache(
    ::MKLPardisoSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
)
    return PardisoLinSolveCache(A)
end

function make_linear_solver_cache(::DenseSolver, ::SparseArrays.SparseMatrixCSC; kwargs...)
    throw(ArgumentError("make_linear_solver_cache: `DenseSolver` has no factor cache."))
end

"""Default relative tolerance of the probe in [`factorization_ok`](@ref)."""
const SINGULAR_PROBE_RTOL = 1e-6

"""
    factorization_ok(cache, A; rtol = SINGULAR_PROBE_RTOL) -> Bool

Return `false` if the factorization in `cache` of `A` is numerically singular. KLU throws
`SingularException` in the factorization, so a `KLULinSolveCache` always returns `true`.
AppleAccelerate and MKLPardiso return finite values for a singular matrix. For them the
function solves `A x = A v` for a known `v` and tests `‖x - v‖ <= rtol * ‖v‖`.
Call it after `full_factor!` or `numeric_refactor!`. `A` must be the matrix of that call.
These functions do not call it.
"""
function factorization_ok end

function factorization_ok(
    ::KLULinSolveCache,
    ::SparseArrays.SparseMatrixCSC;
    rtol::Real = SINGULAR_PROBE_RTOL,
)
    return true
end

function factorization_ok(
    cache::AAFactorCache,
    A::SparseArrays.SparseMatrixCSC;
    rtol::Real = SINGULAR_PROBE_RTOL,
)
    return _solve_probe_ok(cache, A, rtol)
end

function factorization_ok(
    cache::PardisoLinSolveCache,
    A::SparseArrays.SparseMatrixCSC;
    rtol::Real = SINGULAR_PROBE_RTOL,
)
    return _solve_probe_ok(cache, A, rtol)
end

function _solve_probe_ok(
    cache::LinearSolverCache,
    A::SparseArrays.SparseMatrixCSC{T},
    rtol::Real,
) where {T}
    v = T.(1:size(A, 1))
    x = A * v
    solve!(cache, x)
    all(isfinite, x) || return false
    err = sqrt(sum(i -> abs2(x[i] - v[i]), eachindex(v)))
    return err <= rtol * LinearAlgebra.norm(v)
end

"""
    check_factorization(cache, A; rtol = SINGULAR_PROBE_RTOL) -> cache

Throw `LinearAlgebra.SingularException(0)` when [`factorization_ok`](@ref) is `false`.
The exception has no column index, because the probe cannot name one.
"""
function check_factorization(
    cache::LinearSolverCache,
    A::SparseArrays.SparseMatrixCSC;
    rtol::Real = SINGULAR_PROBE_RTOL,
)
    factorization_ok(cache, A; rtol = rtol) || throw(LinearAlgebra.SingularException(0))
    return cache
end
