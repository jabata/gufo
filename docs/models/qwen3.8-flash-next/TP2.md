# Qwen3.8-Flash-Next TP=2

Two-host tensor-parallel serving over RDMA: native InfiniBand, RoCE v2, or USB4
cables as RoCE v2 devices ([below](#usb4)). It serves Q4 and full
Q8, AR and MTP, images, concurrent requests and history reuse. It needs
libibverbs and the build option `GUFO_ENABLE_TP2_RDMA` (off by default). Its
measurements are in [EXPERIMENTS.md](EXPERIMENTS.md#tp2), not in the published
one-host tables.

## Build

```sh
nix develop --inputs-from .#tp2-rdma -c cmake --preset gpu-tp2
nix develop --inputs-from .#tp2-rdma -c cmake --build --preset gpu-tp2 --target gufo

# Release binary with TP2
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON
cmake --build --preset release --parallel 4
```

## Run

Start rank 0 first, and rank 1 within 30 s. `RANK0_ADDRESS` is where rank 1
reaches rank 0's TCP bootstrap. Both ranks need the same model files and the
same `--tp-control-token`, `--context`, `--prefill-chunk`, `--sessions`, MTP
settings and `--mmproj` use; the handshake names any setting that differs.

```sh
# rank 0: public HTTP server
build/gpu-tp2/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 \
  --tp-control-port 18516 --tp-control-token SHARED_TOKEN --sessions 4

# rank 1: worker only, no public HTTP port
build/gpu-tp2/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 1 --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN --sessions 4
```

The link uses port 1 of the host's only RDMA device; `--tp-rdma-device NAME`
and `--tp-rdma-port N` choose another. On InfiniBand it uses GID 0; on RoCE
(an Ethernet link layer) the port's RoCE v2 GID, preferring its IPv4 address's
GID. `--tp-gid-index` overrides that, and is needed when the port has several
candidates. Both ranks must use the same link layer. Each rank logs its choice as
`event=rdma_ready`. RoCE v2 is qualified over USB4 only.

### USB4

Two Strix Halo hosts linked by USB4 cables can run TP2 without network cards:
[thunderbolt-ibverbs](https://github.com/neuhaus/donnerkeule/tree/feat/write-striping)
turns each cable's DMA rings into RoCE v2 RDMA devices. Its write striping
spreads Gufo's single QP over every ring of every cable to the other host;
without it a QP stays on one ring (about 10 Gbit/s). With two 40 Gb/s cables
it reaches 44 Gbit/s one way. Load it with an IPv4 address on its
`roce_netdev` (a dummy netdev will do) and pick any of its devices:

```sh
sudo ip link add tbv0 type dummy
sudo ip addr add 10.77.0.1/24 dev tbv0 && sudo ip link set tbv0 up   # .2 on rank 1
sudo modprobe thunderbolt_ibverbs profile=linux_perf tbnet=prefer_rdma \
  lanes=2 register_verbs=1 roce_netdev=tbv0 native_write_striping=1
ibv_devices                        # usb4_rdma0 .. usb4_rdma3 with two cables
build/gpu-tp2/gufo serve llm ... --tp-rdma-device usb4_rdma0
```

Both ranks produce output identical to InfiniBand. Against FDR InfiniBand
(ConnectX-3, PCIe 3.0 x4), Q4 with a 25.8k-token prompt prefilled at 1824
against 1892 tok/s and decoded at 32.6 against 34.5 tok/s; USB4 costs more
CPU (several cores in kernel workers) and latency per exchange. The links
must train at 2 × 20 Gb/s; check `rx_speed` and `rx_lanes` in
`/sys/bus/thunderbolt/devices/*/` after plugging.

#### Device windows over USB4

The speedup needs the experimental driver mode of thunderbolt-ibverbs
(branch `codex/gpu-verbs-checkpoint`, not the default), loaded on both hosts
with `dmabuf_verbs=1 dmabuf_push_done=1 dmabuf_credits=1` (direct post is on by
default there). With it a GPU window registers as a DMA-BUF, the receiver's
posted windows arm SEND credits, and the sender skips the per-exchange PREPARE
round trip. The stock driver refuses `ibv_reg_dmabuf_mr`, so Gufo logs
`windows=host_write (ibv_reg_dmabuf_mr: Operation not supported)` and runs as
before; the driver mode never changes results, only exchange latency.
Both ranks must run the same Gufo build (bootstrap wire version 8) and agree on
the driver mode, or the pair falls back to host windows.

`qwen38_flash_next_tp_batched_probe --allreduce-bench N` times the exchange,
checks every sum exactly and prints the transport. fuzzy/misty, two USB4
cables, p50 of 1000 exchanges per size, alternating runs, all sums verified:

| Rows | Host windows (write) | Device windows (send, credits, direct post) |
| ---: | ---: | ---: |
| 1 | 32.0 µs | 29.5 µs |
| 2 | 41.8 µs | 31.8 µs |
| 8 | 54.6 µs | 45.6 µs |
| 512 | 1419 µs | 1316 µs |

On the stock driver (host windows) with the final build, the same probe gave
31.4, 40.9, 51.3 and 1443 µs on rank 0 (one run).

End to end on the stock driver (host windows), Q8 MTP, greedy, an empty prompt
cache per server start, three starts each of the branch and its base,
alternating: all four short and all four long (4.9k to 12k tokens, 384-token
answers) outputs equal the previous build's token for token. Median short
decode was 41.3 to 45.4 tok/s on the branch against 42.3 to 45.8 on its base;
a 3.2k-token prefill took 2.63 s on both (the first start of a session 2.93 s
for the branch), and the four long prefills differed by less than run-to-run
noise (for example 12k tokens: 8.47 to 8.76 s against 8.40 to 8.59 s, one base
start 13.3 s). The deferred send completion and the yielding spins therefore
cost nothing measurable on the host path.

Three things that broke a boot-time setup (Ubuntu 26.04, rdma-core 61.0):

- `insmod` of a built `thunderbolt_ibverbs.ko` fails with "Unknown symbol"
  unless `ib_core` and `ib_uverbs` are loaded first
  (`modprobe -a ib_core ib_uverbs`); `modprobe` of an installed module loads
  them itself.
- rdma-core's `60-rdma-persistent-naming.rules` renames the new devices, which
  broke their uverbs nodes. Copy it to `/etc/udev/rules.d/` and add
  `KERNEL!="usb4_rdma*"` to its `rdma_rename` line.
- The devices register only after both hosts have loaded the driver. After a
  reload on both hosts this took more than 180 s, so a fixed sleep is not
  enough: wait for `ACTIVE` in `/sys/class/infiniband/<device>/ports/1/state`,
  and let the supervised ranks keep retrying.

On a kernel that does not export `tb_ring_throttling()` (Ubuntu 26.04's
7.0.0-38 does not), the driver logs "optional tb_ring_throttling() helper
unavailable; using stock NHI interrupt throttling" and ignores
`nhi_interrupt_throttle_ns`, so the rings keep the stock 128 µs interrupt
throttling and decode is bound by the exchange latency. There, `ib_write_lat`
measured a typical 65.3 µs per exchange for both 64 B and 16 KiB, and
`ib_write_bw` 31.86 Gbit/s. With a rebuilt `thunderbolt` module that adds
the helper, plus donnerkeule's kernel patch
`0004-thunderbolt-nhi-clear-pending-before-unmask.patch`, it measured 8.5 µs
(64 B), 28 µs (16 KiB) and 40.5 Gbit/s. One greedy Q8 + MTP request then
decoded at 42.8 instead of 34.7 tok/s (median of three runs each, both with
the MTP draft policy on its one-session costs; see gufo-org/gufo#258). Check
`dmesg` for that line.

Chat and raw completions (including `ignore_eos`), sampling, streaming, stop
sequences, tool calls, images (`--mmproj` on both ranks), cancellation,
`--request-timeout-ms`, history reuse and the disk cache (`--cache-disk` on
both ranks, each on its own disk) work as on one host.
A RAM cache above the automatic budget needs the same `--cache-ram-bytes` on
both ranks; the pair uses the smaller of the two ranks' limits.
Cancellations and timeouts take effect between model calls, so a long prefill
stops at its next chunk. Up to `--sessions` requests run at once, their
decoders batched; more wait up to `--max-pending`, the rest get HTTP 429.

Limits:

- A disagreement fails its request with HTTP 500. A lost peer or link makes
  both ranks exit with status 1; rank 0 first turns `/ready` red and gives
  failed requests a second to report. Run both ranks under a supervisor that
  restarts them (for example systemd `Restart=always`): each waits about 30 s
  for the other before loading the model, so restarted ranks pair up again.
  A peer that exits is noticed at once and a host that vanishes after about
  25 s; a living peer that falls behind, for example while its n-gram reads
  wait on a throttled SSD, is waited for up to 3 minutes.
- Each session holds a full-context state on each rank. Full Q8 (six `Q8_0`
  shards, which do not fit one host) at 262,144 tokens uses 87 GB of each
  host's 127 GB GPU memory with four sessions and 103 GB with eight. Nine
  failed to load on rank 0 (`text runner request-state claim exceeds state
  capacity`) while rank 1 loaded nine at 107.5 GiB.
- Logits differ slightly from one host's (reduction order; see the
  measurements in EXPERIMENTS.md), so a close choice can go the other way. In
  the `tool-reasoning` suite's case with `<|im_end|>` inside a tool argument,
  one host writes a native `<tool_call>` (logit 18.83 against 17.19 for the
  client markup `<`) and TP2 the client markup (17.68 against 18.49);
  `qwen38_flash_next_tp_probe --top N --prompt-file FILE` shows such a choice
  on either.
- Where TP2 is still behind one host (gaps to close, not design choices):
  - No in-pass prompt checkpoint (`PrefillThrough`): the last prefill pass
    stops at the checkpoint boundary and snapshots there.
    `tp_executor_test` fails when the TP2 wrapper turns off a runner
    capability that is not listed there as such a gap.
  - A paired prefill chunk computes every row of the final layer, where one
    host skips the rows nothing reads.

## Design

**Partition.** Both hosts hold the whole checkpoint and each loads its share of
the split weights. Every routed and shared expert's intermediate dimension is
halved on quantization-block boundaries (a routed down projection in a 256-value
super-block type is refused at load); every rank computes all selected experts
on its half. Each GDN layer's 16 key heads and the 48 value heads that read them
(value head h reads key head h % 16) and each attention layer's two KV heads
with their query heads are split, and each rank keeps only its heads' KV cache
and recurrent state. The indexer, draft block, HC, embedding, PLE, LM head and
vision encoder are replicated. A greedy draft scores only its rank's half of
the Q8_0 LM head; the two halves' argmax candidates cross in one all-reduce
row. Target rows, sampled drafts and logprobs keep the full head.

**Exchange.** Each mixer block and each MoE ends in one exchange of partial
sums, 10 KiB per decode row. A stage kernel copies the partial to host memory
and raises a flag; a communicator thread writes it into one of the peer's three
receive windows with an RDMA write with immediate data, whose arrival is the
readiness signal; a one-wave kernel holds the stream until it arrives, and the
GPU adds the peer's partial straight from the window. The host queues a whole
forward without waiting. The rotating windows need no acknowledgement provided
the add of exchange k is queued before exchange k+2 starts. A window holds one
prefill batch's partial (48 MiB for 4,224 rows); loading refuses a batch that
would not fit. Each exchange
header carries the call's scope, an ordinal and the byte count; a mismatch
poisons the communicator. A sum of two operands keeps both ranks bit-identical;
TP2 and one host differ by reduction order. HIP graphs are off under TP2.

**Device windows.** When both ranks can register GPU memory with the RDMA
device as a DMA-BUF (`ibv_reg_dmabuf_mr`), the three windows live in uncached
device memory and each exchange is a SEND with immediate data into a receive
the peer posted for that window, instead of a write into host memory. A window's
receive is posted when its previous use has been read: when this rank sends
the exchange two before the window's next use, after the add that last read it.
The partial starts at the window (direct placement into GPU memory needs a
page-aligned offset), so there is no header: the completion's length (padded to
32 bytes) and an immediate derived from the call's scope and the exchange's
ordinal replace the header's size and identity checks, and a mismatch poisons
the communicator as before. The stage kernel writes the send window and the
NIC reads it, and the NIC writes the receive window and the add reads it, with
no cache between them; hence the uncached allocation.

The ranks agree on the placement during the bootstrap and log it in
`event=rdma_ready` as `windows=gpu_send`. If either rank's device refuses the
registration, both keep the host windows and writes with immediate data of the
sections above and log `windows=host_write (reason)`, for example
`ibv_reg_dmabuf_mr: Operation not supported` on the stock USB4 driver. No option
selects it. The bootstrap's wire version is 8: a rank running an older build
refuses to pair, so **both hosts must run the same build**.

**Waiting.** An exchange finishes when the peer's partial has arrived and this
rank's *earlier* sends have completed, not its own: that completion follows the
peer's acknowledgement (host windows) or placement report (device windows),
both after the partial arrived. The send window is reused three exchanges
later, after the next exchange has collected the completion, and a failed send
fails that next exchange. The communicator's spin waits yield the CPU every 64
iterations: a thread spinning on a core holds off the per-CPU kernel worker that
finishes the USB4 driver's ring completions for a scheduler slice, which showed
up as exchange tails of about 3 ms once direct placement removed the round trip
before each send.

**Prefill overlap.** A prefill step of up to 8,448 tokens runs as two trunk
batches a layer apart (one batch under 384 tokens), so each batch's exchange
crosses the link while the other computes; the results equal two steps of
those sizes bit for bit.

**Rank 1 as executor.** Rank 0 runs the ordinary scheduler, runner pool and
continuation cache through `TpMirroredRunner`. Just before each call that
changes model state, it sends the call to rank 1 over an authenticated TCP
control channel; rank 1's `TpExecutor` makes the same call on the same state,
and both ranks meet in the call's exchanges. Token selection reads only logits
and stays on rank 0. A request opens with `kSingle` (prompt, sampling
configuration and prepared images) and closes with `kEnd`. Mirrored calls:

- prefill chunks, single and batched advances, and single and batched
  multi-token MTP cycles. A cycle carries rank 0's sampler draw state so both
  ranks draw alike, and rank 0's draft count, so both ranks run cycles of the
  same width even when their acceptance histories disagree; a greedy batch
  carries one draft count for the whole batch, a sampled one each member's. A
  request with tools or a response format sends rank 1
  what its constraint was built from (tools, response format, tool choice),
  and rank 1 builds the same constraint with the same code, so it judges
  drafts as rank 0 does; a constraint without that source decodes one token
  at a time;
- state resets, and binding a state to its request's prompt context (images
  can reset the state);
- snapshot, restore, prefix reuse and cancellation preparation, each
  acknowledged by rank 1 before rank 0 continues; snapshot bytes stay on each
  host, and the budget is the smaller rank's (its automatic budget, or its
  `--cache-ram-bytes`).

Every model call binds the communicator's scope to its instruction's index on
both ranks. Rules the compiler does not enforce: every state-changing call must
pass through the wrapper; a request's cancellation check is never passed to the
model session, which could strand rank 1 inside an exchange; TP2 must not
change the cache's boundaries.

**Disk cache.** Each rank keeps its own half of a snapshot on its own disk.
Rank 0's continuation disk store decides what is saved, restored and evicted,
as on one host. Its file holds rank 0's half behind a header that names a
random file key; when it writes the file, rank 1 persists its half under that
key in the background. Snapshots taken for the disk cache are complete on
both ranks, so neither writer reads rows its session still shares.
Restoring a file restores both halves, rank 1's from
its own directory, acknowledged before rank 0 continues. Rank 1 bounds its
directory by least-recent use within its own `--cache-disk-bytes`; a missing,
corrupt or foreign half fails the restore on rank 1, which rank 0's store
treats as a miss (the state is reset on both ranks and prefilled). TP2 files
carry the rank in their cache identity, so neither a one-host server nor the
other rank restores them. Within one process, a file of a request rank 1
rejected is a miss, and a restore before the verdict depends on it.

**Agreement.** Both ranks digest every call and its result; `kEnd` carries rank
0's digest and call count, and rank 1 fails the request on a difference. Under
greedy decoding rank 1 also checks its own argmax against every token rank 0
advances. A capture carries the request's digest so far, and rank 1 captures
only a state both ranks reached the same way. Snapshots and released states
are reused at once, as on one host; a request that reuses what an unjudged
request left fails too if rank 1 later rejects that request, and a rejected
request's states and snapshots are not reused.

## Upstream integration, October 14, 2026

The cache-integration branch and upstream main through `942999e7` are merged
with the RDMA head and expert partitions retained. Packed Q8 decode inputs,
vectorized residual additions, padded SSM output rows, mapped PLE rows and
fused HC inject work also serve TP2. SSM padding retains the rank's 3072-wide
projection tile plan, and packed decode rows use its actual value-head width. Inject work runs after the rank reduction. Weight placement measures
and uploads the same partition-aware traversal, including temporary gathers.

Upstream's wave-owned paired gate/up kernel serves Q4_K and Q5_K. Q8_0 pairs
keep their signed-byte staging and wave pairing; both 64- and 128-row tiles
remain available. The routed oracle checks both widths, including ragged TP2
output and input partitions, against unchanged numerical limits. Down
projections address the separate 64-row map rather than the leading 48-row map.

Single-host Flash-Next serving uses upstream's component cache. TP2 retains
its coordinated rank snapshots and disk cache: component captures and their
ownership transitions are not yet mirrored by the control protocol. In-pass
checkpoints remain disabled in the TP wrapper. Deferred MoE residual fusion
stays on the single-host path, where it cannot bypass a required rank sum.

The configurable CQ spin budget is saved separately on
`codex/wip-tp2-transport-experiments` and is not part of this merge; the DMA-BUF
device windows are ([Design](#design)).

## Tests and probes

- `tp_control_test`, `tp_executor_test` (label `tp2`, in the CPU suite): the
  protocol, and the real scheduler and pool on rank 0 against an executor on
  rank 1 with a toy model, for every request kind, divergence and reuse rule.
- `qwen38_flash_next.tp_partition` and the attention, projection and routed
  operator tests: the split against the full kernels.
- `qwen38_flash_next_tp_probe` (`gpu-tp2`): prompt, decode and logit
  comparison on both ranks; `--split N` checks that a prefill does not depend
  on its steps (every layer's MoE input and output, and the logits, bit for
  bit), `--decode-tail K` compares decoding the last K tokens with
  prefilling them.
- `qwen38_flash_next_tp_batched_probe` (`gpu-tp2`): batched against serial
  decoding on both ranks without the serving stack; `--allreduce-bench N`
  times the exchange.
- `tools/bench/model-bench.py` launches both ranks through a `gufo.tp2`
  overlay in `bench.json` (`remote_host`, `bootstrap_host`, ports, token,
  container and binary); it runs the single-user and users tables, with the
  container `gpu-tp2` build over USB4 (the Nix build lacks its RDMA provider).

## Qualifying a change

On both hosts, from one commit (record it and the binary hash):

1. The format check on a fresh copy of the tree, and the CPU suite and TP2
   tests above with no skips.
2. Two hosts, Q4 and Q8, AR and MTP, greedy: outputs repeat, AR and MTP agree,
   and a prompt of over 30K tokens works (several full prefill steps).
3. Seeded sampling reproduces; streaming, stop sequences, a tool call and
   client disconnects during decode and prefill behave as on one host.
4. A multi-turn cached chat restores its whole prompt on replay; cached token
   counts match one host (outputs may differ after a few turns).
5. Two, four and eight concurrent requests, AR and MTP, each equal their
   one-at-a-time output, seeded sampled ones too; the queue bound returns 429;
   a long prefill runs beside decoders.
6. Images: a request, its cached repeat, a follow-up and two images in one
   request match one host's answers.
7. Disk cache (`--cache-disk` on both ranks): after both ranks restart,
   `tools/serving/check-continuation.py --restore` restores its greedy cases
   from disk with the same outputs.
8. Killing rank 1 mid-request returns 500 promptly, and rank 0 then exits
   with status 1; killing rank 0 makes rank 1 exit with status 1. SIGTERM
   stops either rank promptly.
9. For speed, report the median of several warm requests.

## Open

1. **Target verification on the split head**: only greedy drafts score half
   the vocabulary. Verification rows and the frontier row the host samples
   from still use the full head on both ranks.
2. **Short prompts over USB4**: a fresh 2K prompt prefills more slowly than on
   one host (1,300 against 1,492 tok/s, Q4 AR). On September 29, apparently
   still over the ConnectX link, the pair reached 1,541 tok/s there.
3. **Q8 MTP with several users**: at C8 it trails AR (117.7 against
   131.8 tok/s), and its rates vary by up to 17% between server starts.
4. **Q4 perplexity drift**: TP2 Q4 now scores 2.1182 against 2.1093 on
   September 27 on the same WikiText-2 windows; not attributed.
5. **MTP oracle**: `--mtp-audit` fails its round 9 wide-attention check by a
   relative RMS of about 1e-6, on builds before and after the Q8 expert route.
6. **Seeded sampling after upstream merges**: one seeded case changed output
   after a merge of upstream `main` in late September; not attributed.
7. **Q8 against an official reference**: Gufo and llama.cpp differ by KL
   about 0.09; which is closer to the original model is open.
