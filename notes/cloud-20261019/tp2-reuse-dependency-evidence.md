# Evidence (all on this cloud VM, CPU only)

Base: neuhaus/rdma ff6c44e. Branch cloud/tp2-reuse-dependency, commit bd2c9dd (one commit; amended from fe5543a after review round 1, never pushed). Ubuntu 24.04.5, g++ 13.3.0, CMakeCache: ENGINE_ENABLE_HIP=OFF, GUFO_ENABLE_TP2_RDMA=OFF.

1. Base suite, `ctest --preset cpu-test -j4 -LE gpu` on ff6c44e: Skipped 2,3,4,64,70,113; Failed only 52 serving_bench_test, whose output is "charts need matplotlib; enter `nix develop`".
2. Same on fe5543a: "99% tests passed, 1 tests failed out of 124", with the same 6 skipped and the same single failure (serving_bench_test). 124 - 6 skipped - 1 failed = 117 passed.
3. `cmake --build --preset pr` on fe5543a: "100% tests passed, 0 tests failed out of 62".
4. `PATH=<clang-format 21.1.8> python3 tools/ci/check-format.py`: "PASS: Checked 604 C++ files". The base is also clean with 21.1.8; system clang-format 18 flags many unrelated base files.
5. `tp_executor_test` on fe5543a: "PASS: TP2 executor mirrors greedy, sampled, ...".
6. The three new checks, run one at a time (each Require temporarily turned into a printf), printing 1 when the expectation holds:
   - With the src/ change stashed (base executor): CHECK 0 / CHECK 0 / CHECK 0.
   - With the fix: CHECK 1 / CHECK 1 / CHECK 1.
   - With the base executor, the unmodified new test prints "FAIL: continuing a state rejected after the reuse check fails".
7. Wait probe (temporary, not committed): request 8 prefills slot `stale` and ends, unsettled. Request 9 snapshots `source` and is settled agreed. Request 11 leases `stale` (SetPromptContext), then RestoreOrFork(stale, snapshot of 9), then AwaitDependencies(11) is timed:
   - Base executor: "PROBE unsettled-overwritten await=0 after 60.0 s".
   - Fix: "PROBE unsettled-overwritten await=1 after 0.0 s".
8. Cache flow claim (code reading, continuation_cache.cpp, Acquire): the reuse checks (StateReusable/SnapshotReusable) run under impl_->mutex. `lock.unlock()` comes before `prepare_state(*entry.state)` and `impl_->snapshot_support.restore(...)`. needs_invalidation = entry.dirty && !cache_hit, so a snapshot hit does not reset the slot before prepare_state and restore.
9. Not claimed: frequency in production, any hardware behaviour, any cascade.

## Round 2 (commit bd2c9dd)
10. Removed ForgetInherited from SendInvalidate. ForgetInherited now skips dropping when producers_[state] == the lease's sequence. Two checks added (4: restore after own Advance keeps dep; 5: reset after own Advance keeps dep).
11. Per-check results, each Require turned into a printf (1 = expectation holds):
    - BASE (ff6c44e executor): CHECK1 0 CHECK2 0 CHECK3 0 CHECK4 1 CHECK5 1
    - FE55 (round-1 executor): CHECK1 1 CHECK2 1 CHECK3 1 CHECK4 0 CHECK5 0
    - NEW (bd2c9dd): CHECK1 1 CHECK2 1 CHECK3 1 CHECK4 1 CHECK5 1
12. Full `ctest --preset cpu-test -j4 -LE gpu` on bd2c9dd, six runs:
    - Run 1: "98% tests passed, 2 tests failed out of 124". The grep showed only #52 serving_bench_test as (Failed); the second failure was not identified (its logs were overwritten by the rerun).
    - Runs 2-6: only "52 - serving_bench_test (Failed)", the same 6 skipped.
    - `ctest -j4 -R 'tp_|http_server_test|component' --repeat until-fail:8`: "100% tests passed, 0 tests failed out of 10".
13. `cmake --build --preset pr`: "100% tests passed, 0 tests failed out of 62". check-format (clang-format 21.1.8): "PASS: Checked 604 C++ files".

## Round 3 (commit 64f7c84, pushed to jabata/gufo cloud/tp2-reuse-dependency; NOT reviewed yet, item dropped for the 2026-10-19 run after 2 CHANGES rounds)
14. SendInvalidate erases inherited_[state] for a leased state. Check 5 now restores after the reset.
15. Per-check: BASE 0 0 0 1 1 | BD2C 1 1 1 1 0 | NEW 1 1 1 1 1.
16. ctest -j4 -LE gpu (log in ctest-final.log): "99% tests passed, 1 tests failed out of 124", with only 52 serving_bench_test failing. pr preset: 62/62. check-format: PASS 604 files.
