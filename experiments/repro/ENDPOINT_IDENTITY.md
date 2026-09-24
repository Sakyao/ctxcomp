# Inference endpoint identity (for reproducibility / paper appendix)

Recorded 2026-09-22/23 on the host that serves both endpoints (`192.168.1.13`).

## Summary

| Endpoint | `--served-model-name` (label) | **weights actually mounted** | `random_seed` | `tp_size` |
|---|---|---|---|---|
| `:18173` | `deepseek-v4-flash` | **`/home/ame/model/DeepSeek-V4-Flash-0731`** | 565077876 | 4 |
| `:18174` | `deepseek-v4.1-flash` | **`/home/ame/model/DeepSeek-V4-Flash-0731`** | 733020309 | 4 |

**Both endpoints serve the same weights.** The two names differ because they were
passed as different `--served-model-name` flags at launch; the flag is a
free-text label and is not read from the checkpoint.

## Why this matters for reported numbers

1. **The `model` field in any request/response/log is meaningless.** The server
   does not validate it -- it echoes back whatever string the client sent:

   ```
   POST :18173  {"model": "THIS-IS-A-FAKE-NAME-12345", ...}
   -> 200 {"model": "THIS-IS-A-FAKE-NAME-12345", ...}
   ```

   So our experiment configs (which pass `model: "deepseek-v4.1-flash"`, the name
   this harness has always used) do not identify the backend. All runs on either
   port went to `DeepSeek-V4-Flash-0731`.

2. **No model swap occurred.** An earlier note suspected the backend had changed
   between two runs with very different accuracy (82.1 vs 32-42). That inference
   came from comparing `served_model_name` across two *different ports* and is
   withdrawn: both ports mount the same directory, and the containers have been
   running continuously (see `random_seed`, which sglang picks at startup and
   which stayed constant across repeated probes).

3. **A restart did happen** on `:18173` at 2026-09-22 16:07:51 (container
   `219516cba448...`), and `:18174` at 12:49:48 (container `6ff509b37023...`).
   Neither restart changed the mounted weights -- both currently point at
   `DeepSeek-V4-Flash-0731`. Whether `:18173` mounted something else *before*
   16:07 cannot be recovered from the API and would have to come from the
   operator.

## Recommended citation form

> Backbone: **DeepSeek-V4-Flash-0731**, served by sglang `0.5.10rc0` with `tp=4`
> on 4x H200 (141 GB each). The harness requested `model="deepseek-v4.1-flash"`;
> this string is a client-side label that the server echoes back unvalidated, not
> the identity of the served checkpoint.

## How identity was established (all local, no cooperation from the server)

```bash
# 1. the sglang launch commands (labels + ports)
ps -eo pid,args | grep launch_server
#   --served-model-name deepseek-v4.1-flash --port 18174 ...
#   --served-model-name deepseek-v4-flash   --port 18173 ...

# 2. which host directory each container mounts as /models -- authoritative
cat /proc/<pid>/mountinfo | grep ' /models '
#   ... /home/ame/model/DeepSeek-V4-Flash-0731 /models rw,relatime - ext4 ...

# 3. behaviour check (secondary): next-token logprob distributions agree between
#    the two endpoints within their own run-to-run noise (cross/within L1 = 1.10;
#    7 of 8 probes had identical top-1 tokens in 12/12 repeats)
python experiments/repro/compare_endpoints.py \
   --a http://192.168.1.13:18173/v1 --b http://192.168.1.13:18174/v1 --repeats 12
```

`fingerprint_endpoint.py` now records `weight_dir` (from `/proc/<pid>/mountinfo`)
on every probe, and flags an ENDPOINT CHANGED warning when any identity field
differs from the previous probe. Raw evidence:

```
logs/endpoint_ab/server_info_18173.json
logs/endpoint_ab/server_info_18174.json
logs/endpoint_ab/compare_18173_vs_18174.json
```

## Throughput

The two ports are **not** different in speed. Measured on the same weights:

| condition | :18173 | :18174 | ratio |
|---|---|---|---|
| 1 request, ~150 prompt tokens | 0.20 s | 0.19 s | 0.97x |
| 1 request, 15k prompt tokens | 0.25 s | 0.29 s | 1.16x |
| 6 concurrent, 15k prompt tokens | 0.55 s | 0.61 s | 1.11x |
| decode throughput | 96.0 tok/s | 96.7 tok/s | 1.01x |

Wall-clock differences between runs come from what else is sharing the server at
the time -- the host had 24 logged-in users, and `:18173` was observed serving a
steady external request while none of our processes were running.
