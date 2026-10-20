# klu_deepcopy_use_after_free.jl
#
# Purpose:
#   Shows the cause of the "libklu corrupts state across independent caches"
#   failures (SuiteSparse#1040) that `_LIBKLU_LOCK` was added for: a `deepcopy`
#   of a Virtual matrix aliases its libklu factorization. Nothing is serialized
#   by a process-wide lock anymore; the deepcopy guard on `KLULinSolveCache` is
#   what prevents the failure.
#
#   The pattern mirrors PowerSimulations' `DecisionModel` before
#   PowerSimulations.jl@88b2806: the model deepcopies the caller's template,
#   which holds a `VirtualPTDF`. `deepcopy` bit-copies the cache's raw
#   `symbolic`/`numeric` pointers, but the finalizer stays attached to the
#   original only. Once the original template is garbage-collected, its
#   finalizer frees the factorization while the copy keeps solving with it:
#   `klu_solve` reads freed memory and returns `KLU_INVALID` ("invalid
#   argument") or crashes with SIGSEGV / EXCEPTION_ACCESS_VIOLATION. Threads
#   are not required; they only change when the GC runs.
#
#   Three scenarios, each run in its own child process (a failure can crash it):
#     alias  - deepcopy with the deepcopy guard removed (PNM before 0103364):
#              expected to crash or return KLU_INVALID / wrong rows
#     guard  - deepcopy with the guard: refused with an error at copy time
#     share  - the matrix shared by reference (PowerSimulations' fix): correct
#
# Run command (from repo root, with this repo dev'ed into the test env):
#   julia --project=test scripts/diagnostics/klu_deepcopy_use_after_free.jl [runs]
#
#   `runs` (default 3) is the number of child processes per scenario. Each child
#   builds and queries 50 fresh VirtualPTDFs on c_sys5 with 4 threads.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "test"); io = devnull)

using PowerNetworkMatrices
using PowerSystemCaseBuilder
using Printf

const PNM = PowerNetworkMatrices
const PSB = PowerSystemCaseBuilder

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const N_ITERS = 50
const N_THREADS = 4
const CHILD_TIMEOUT_S = 600

# Stand-in for PowerSimulations' ProblemTemplate / NetworkModel.
mutable struct Template
    ptdf::Any
end

function _copy_for_model(template::Template, mode::String)
    mode == "share" && return Template(template.ptdf)
    return deepcopy(template)
end

# Runs inside a child process; prints one RESULT/REFUSED line for the driver.
function child_main(mode::String)
    if mode == "alias"
        # Remove the deepcopy guard, restoring the behavior before PNM 0103364.
        guard = which(
            Base.deepcopy_internal,
            Tuple{PNM.KLUWrapper.KLULinSolveCache{Float64, Int}, IdDict},
        )
        Base.delete_method(guard)
    end
    # invokelatest: this call started in a world where the guard still exists.
    return Base.invokelatest(run_scenario, mode)
end

function run_scenario(mode::String)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    reference = VirtualPTDF(sys; linear_solver = "KLU")
    arcs = collect(PNM.get_arc_axis(reference))
    expected = [copy(reference[a, :]) for a in arcs]

    function build_model()
        template = Template(VirtualPTDF(sys; linear_solver = "KLU"))  # caller's template
        return _copy_for_model(template, mode)  # what the model keeps
    end

    n_errors = 0
    n_wrong = 0
    first_error = ""
    for _ in 1:N_ITERS
        model = try
            build_model()
        catch e
            msg = sprint(showerror, e)
            occursin("deepcopy", msg) || rethrow()
            println("REFUSED ", first(replace(msg, '\n' => ' '), 60), "...")
            return
        end
        GC.gc()  # the caller's template is unreachable: its finalizer frees the handles
        _ = [rand(64) for _ in 1:2000]  # allocate, so the freed blocks get reused
        rows = Vector{Any}(undef, length(arcs))
        Threads.@threads for i in eachindex(arcs)
            rows[i] = try
                copy(model.ptdf[arcs[i], :])
            catch e
                sprint(showerror, e)
            end
        end
        for (row, ref) in zip(rows, expected)
            if row isa String
                n_errors += 1
                isempty(first_error) && (first_error = first(row, 120))
            elseif !isapprox(row, ref; atol = 1e-9)
                n_wrong += 1
            end
        end
    end
    println("RESULT errors=$n_errors wrong=$n_wrong rows=$(N_ITERS * length(expected))")
    isempty(first_error) || println("FIRST_ERROR ", replace(first_error, '\n' => ' '))
    return
end

function run_child(mode::String)
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$N_THREADS
        --project=$(Base.active_project()) $(@__FILE__) --child $mode`
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err); wait = false)
    hung = timedwait(() -> process_exited(p), CHILD_TIMEOUT_S) !== :ok
    hung && kill(p)
    wait(p)
    out_s, err_s = String(take!(out)), String(take!(err))
    m_res = match(r"RESULT errors=(\d+) wrong=(\d+) rows=(\d+)", out_s)
    m_err = match(r"FIRST_ERROR (.*)", out_s)
    m_ref = match(r"REFUSED (.*)", out_s)
    crash = match(r"EXCEPTION_[A-Z_]+|signal \(?\d+\)?[^\n]*", err_s)  # Windows | Unix
    frame = match(r"(klu_\w+) at ", err_s)
    if hung
        return (failed = true, text = "HUNG (killed after $(CHILD_TIMEOUT_S) s)")
    elseif m_ref !== nothing
        return (failed = false, text = "deepcopy refused: $(m_ref[1])")
    elseif m_res !== nothing
        n_err, n_wrong, n_rows = parse.(Int, m_res.captures)
        failed = n_err + n_wrong > 0
        text = if failed
            "CORRUPTED: $n_err of $n_rows rows threw, $n_wrong wrong" *
            (m_err === nothing ? "" : " (first error: $(m_err[1]))")
        else
            "OK: $n_rows rows correct"
        end
        return (; failed, text)
    else
        what = crash === nothing ? "exit code $(p.exitcode)" : crash.match
        where_ = frame === nothing ? "" : " in $(frame[1])"
        return (failed = true, text = "CRASHED: $what$where_")
    end
end

function main(runs::Int)
    pkg_dir = normpath(joinpath(dirname(pathof(PowerNetworkMatrices)), ".."))
    pkg_dir == REPO_ROOT || error(
        "The test env loads PowerNetworkMatrices from $pkg_dir, not this repo. Run " *
        "`julia --project=test -e 'using Pkg; Pkg.develop(path=\".\")'` first.",
    )
    println("Julia $(VERSION), $N_THREADS threads per child, $runs runs per scenario, ",
        "$N_ITERS VirtualPTDFs per run (c_sys5, KLU)\n")
    failures = Dict{String, Int}()
    for (mode, label) in (
        ("alias", "deepcopy, guard removed"),
        ("guard", "deepcopy, guard in place"),
        ("share", "shared by reference"),
    )
        failures[mode] = 0
        for r in 1:runs
            res = run_child(mode)
            failures[mode] += res.failed
            @printf("%-26s run %d: %s\n", label, r, res.text)
        end
    end
    println()
    println("Failing runs: alias $(failures["alias"])/$runs, ",
        "guard $(failures["guard"])/$runs, share $(failures["share"])/$runs")
    if failures["alias"] > 0 && failures["guard"] == 0 && failures["share"] == 0
        println(
            "As expected: the use-after-free needs the aliasing deepcopy; the guard prevents it.",
        )
    end
    return failures["guard"] + failures["share"] == 0
end

if length(ARGS) >= 2 && ARGS[1] == "--child"
    child_main(ARGS[2])
else
    exit(main(isempty(ARGS) ? 3 : parse(Int, ARGS[1])) ? 0 : 1)
end
