AI disclosure: I am not a software developer. Claude Code (Claude Opus 5.5) developed and tested this patch; an independent Claude Fable 5.1 review approved it before submission; this is not a claim of human developer review. Claude Code cloud session, CPU only, no GPU testing.

Test scope (CPU build/tests only; no RDMA, no GPU, no model): `cmake --preset cpu-test` on Ubuntu 24.04 (GCC 13.3, `ENGINE_ENABLE_HIP=OFF` and `GUFO_ENABLE_TP2_RDMA=OFF`, so the RDMA communicator is not compiled). `tp_executor_test` passes. `ctest --preset cpu-test -j4 -LE gpu` on this branch was run six times. Five runs gave 117 passed, 6 skipped, and only `serving_bench_test` failing, because matplotlib is not installed on the host; the base commit gives the same result. The first run reported one more failure, and I could not identify it because a rerun overwrote the logs. The TP and server tests (`-R 'tp_|http_server_test|component'`) then passed 8 repeats each under `-j4`. `cmake --build --preset pr`: 62/62 passed. `tools/ci/check-format.py` passes with clang-format 21.1.8. Built in a cloud session (CPU only); Strix Halo validation pending.

## Problem

`TpMirroredRunner` makes a request depend on rank 1's verdict for the work it reuses. Two cases go wrong because the dependency is taken when the state is leased:

1. **A verdict that arrives between the cache's reuse check and the lease is lost.** `ContinuationCache::Acquire` checks `CanReuse` under its own mutex, releases it, and then calls `prepare_state` (`SetPromptContext`) and `restore` (`RestoreOrFork`). If rank 1 rejects the producer in that window, both calls skip it: they only record unsettled producers. `AwaitDependencies` then returns true, and the request is accepted on top of work rank 1 rejected.
2. **A restore keeps the dependency on the work it overwrites.** On a snapshot hit, the cache leases a slot without resetting it, calls `SetPromptContext`, and then restores the snapshot over it. `SetPromptContext` has already made the request depend on the slot's previous producer, and the restore does not drop that dependency. The request then waits for a verdict on work it no longer holds, and fails if that producer is rejected.

On the base commit, a temporary CPU probe (not committed) shows case 2: request 11 restores an agreed snapshot into a slot whose previous producer 8 is still unsettled. It printed `await=0 after 60.0 s`. With this change it printed `await=1 after 0.0 s`.

## Change

- `SetPromptContext` also records a rejected producer, and remembers the dependency it took for the leased state (`inherited_`).
- `RestoreOrFork` and `RestorePersistentSnapshot` drop that dependency before they replace the state's work, but only while the request has neither changed nor reset the state. Work the request already did may rest on the earlier producer's, and its output or published snapshots may carry it. A reset keeps the dependency as before, and a restore after a reset keeps it too.
- `RestoreOrFork` also records a snapshot producer that is already rejected. `RestorePersistentSnapshot` already throws for one.
- `dependencies_` becomes a multiset, so dropping one reuse cannot drop another reuse of the same producer.

## Test

A new block in `tests/cli/tp_executor_test.cpp` uses the mirrored runner alone with `CaptureSink`. It checks five things:

1. Continuing a state whose producer was rejected after the reuse check fails.
2. Restoring a snapshot whose producer was rejected after the reuse check fails.
3. A restore over a slot whose earlier producer is rejected later stands.
4. A restore after the request's own calls keeps its dependency.
5. A restore after a reset of the request's own calls keeps its dependency.

With the base executor, checks 1 to 3 fail and checks 4 and 5 pass. Checks 4 and 5 guard the base behaviour that this change keeps. With the change, all five pass. Each check was verified one at a time.
