# Hippocampus · a local knowledge-base retrieval engine

Language: [简体中文](README.md) | **English**

Hybrid retrieval over your own Markdown notes, built for **one operator on a personal server**:
BM25 lexical scores fused with dense ONNX vectors, served from a resident HTTP process that
answers in milliseconds, and every hit carries file name and line number so you can go read the
source. No database, no vector store, no GPU. The files on disk are the only truth.

```
POST /search {"query": "...", "k": 8}        → matching chunks + source file + line number
GET  /health                                 → ok / error / warn / lag / reload_fail
GET  /stats                                  → per-zone chunk counts, index freshness, last 10 reloads
POST /reload {"zone": "main"}                → swap an index without restarting (replace only on success)
```

## Why this is worth reading

The easiest way to fool yourself about retrieval quality is that **nothing tells you when you made
it worse**. So the gate here is a machine, not a memory: `kb/eval.py` finishes and immediately
compares against `kb/baseline.json`; if it doesn't clear the gate it `exit 1`s.
`kb/baseline_privatedomain.json` records, next to the passing baseline, a full set of **levers that
were measured and disproved** — table row-group surgery, per-row column-name labels, a prose window
above tables, cross-encoder re-ranking, semantic chunking / fixed overlap. Each entry carries its
score, its wall-clock cost, and the reason it died. Most of them are negative results, and that is
where the value is: following the list saves you a few weeks.
The evidence lives in `kb/CHUNKING-EVIDENCE.md`, `CORPUS-COMP-REPORT.md` and `kb/VERSION-PROFILE.md`
— those three deep dives are Chinese-only for now; everything you need to run and judge the engine
is in this file and in `corpus/README.md`.

## Quick start

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Models are not shipped with the repo (see below): put the bge-m3 ONNX weights in models/bge-m3-onnx/
#    you need model.onnx + tokenizer.json, or point M3_ONNX_DIR at any directory
# 2. Drop your notes into corpus/ (the repo ships 7 self-written docs about the engine itself,
#    which cut into 34 blocks — run them as-is first, then swap in your own)
python kb/chunker.py            # corpus/ → kb/chunks.jsonl
python kb/index.py              # → kb/bm25.npy + kb/bm_vocab.json (deterministic, b=0.75)
python kb/build_vec.py --chunks kb/chunks.jsonl --out kb/vec_m3.npy
python kb/eval.py grep,bm25,vec,hybrid,hybrid-v2      # auto-gated afterwards, fails the gate → exit 1

python kb/kb_serve.py            # foreground, Ctrl-C to stop; or start it detached with kb/_start.ps1
curl -s -X POST http://127.0.0.1:8310/search -H "Content-Type: application/json" \
     --data-binary '{"query":"log rotation: before or after the write","k":5}'
```

`index.py` prints one line, `WARN: ...vec.npy missing` — that is expected. It builds BM25 only;
the dense file is produced separately by `build_vec.py` and named `vec_m3.npy`. (`kb_search.py`
looks for indexes in that priority order. Historically it defaulted to `vec.npy`, and one run
against the main corpus silently used stale small-model vectors: legal shape, plausible-looking
scores, completely wrong.)

The shipped example corpus cuts into the same block count that `kb/baseline.json` records (34
blocks), so the four commands above run straight through and print PASS. When you swap in your own
corpus, `chunker.py` **refuses to overwrite** on a block-count mismatch — by design, not a bug:
change the block order and every row number in the neighbouring vector file shifts, and the service
would quietly serve a misaligned index. Build your own baseline first (`corpus/README.md` has the
steps).

## Layout

| Path | Contents |
|---|---|
| `kb/chunker.py` `index.py` `build_vec.py` | the ingest trio: chunk → BM25 → dense |
| `kb/kb_search.py` `fuse.py` `m3embed.py` | retrieval, multi-zone fusion, ONNX embedder |
| `kb/kb_serve.py` | resident service, hot reload, dashboard, query log and rotation |
| `kb/kb_atomic.py` `kb_log_archive.py` `write_provenance.py` | atomic index swap / log archiving / provenance on disk |
| `kb/kb_remember.py` `kb_epilogue.py` `mem_audit.py` | agent-memory loop: write-back, retrospective distillation, duplicate and conflict adjudication |
| `kb/coverage_audit.py` `kb_health_probe.py` `kb_soak_test.py` `replay.py` | coverage audit, health probe, concurrency soak, real-query replay |
| `kb/eval.py` `baseline*.json` | evaluation and the machine gate |
| `kb/test_*.py` | the gatekeeper unit tests: 48 run on the author's machine; in a fresh clone there is no peer MCP engine and no live service, so 3 of them explicitly skip (46 run / 3 skip). They all rehearse inside temp directories and never touch a production artifact |
| `corpus/` | the corpus entry point (self-written examples ship with the repo) |
| `tools/` | release packaging and the sanitization gate |

## What this repo deliberately does not ship

The author's real corpus is **not published**, and that is enforced by the packaging script's
whitelist rather than by hand-deletion:

- Document text anywhere outside `corpus/`, plus `kb/chunks.jsonl` (a second copy of that text) and
  `bm_vocab.json` (a vocabulary learned from the corpus — measured: even the local username becomes a term).
- **The memory documents**: the whole `kb/mem/corpus/` directory (conclusions an agent wrote back)
  and `kb/epi/corpus/` (retrospective entries). Their code is in this repo; their data is not.
- `kb/query_log.jsonl`, `kb/reload_history.jsonl`, `eval_per_query*.json` — they contain real query text.
- Model weights (`*.onnx`, 1.1 GB), `venv/`, A/B experiment directories and snapshots.
- `tools/release_local.json`: the author's local-path and third-party-name replacement table, which
  is itself the single thing most likely to need catching by the gate.

Want to reproduce the 81.8% / 96.4% measured on the private corpus? You can't, and you don't need
to — that gold set points at documents that should not be public. The `kb/baseline.json` in this
repo freezes the gate for the **example** corpus, so a fresh clone runs green and shows you PASS.

## Five design rules

1. **Writes go through an atomic swap only.** Write `<dst>.new.<pid>` first, `os.replace` only after
   the build succeeds, and the previous file moves to `.prev` with a timestamp, keeping 3 copies.
   The provenance sidecar travels with its vector file; a chunking result of 0 rows refuses to
   overwrite. Use `kb_atomic` for new files — don't write bare `open(..., "w")` / `np.save`.
2. **The service reads, and never rebuilds on its own.** When the corpus on disk is newer than the
   index, `/health` reports `lag` and the dashboard turns yellow, but nothing is rebuilt
   automatically: the rebuild gate is a human step, and handing write access to a watcher is how
   you'd bypass rule 1's baseline guard.
3. **Quality changes must clear the gate.** Any change to fusion weights, chunking rules or
   dictionaries: run `eval.py` first, then `replay.py --diff` to confirm that real traffic's hits
   did not reshuffle wholesale.
4. **Disk usage has a hard ceiling.** Query log 10 MB × (3+1) = 40 MB, reload history 2 MB × 3, and
   the service logs are archived with a timestamp before every start. Rotation happens **before**
   the write, not after.
5. **No boot-time autostart, no watchdog.** If the process dies it stays dead; monitoring only
   alarms. That is a tradeoff, not an omission: silent restarts turn "which index was swapped in and
   when" into an untraceable question.

## Two Windows traps in the ops scripts

- `_start.ps1` must be **UTF-8 with BOM**. PowerShell 5.1 decodes a BOM-less file as GBK, and the
  Chinese comments then break the whole script's syntax so it never runs (the error looks like
  "string missing terminator").
- Don't call it with `capture_output` or any "wait for the pipe to close" style — the background
  service inherits that pipe and the caller hangs forever. The script isn't wrong; the way you
  invoke it is.

## Production verification

`kb/kb_health_probe.py` runs every criterion once and returns an exit code (0 = no FAIL): listen
address, in-memory block count vs disk, freshness of indexed files, corpus-ahead-of-index,
provenance for all five zones, process private memory and thread count, **whether the code is newer
than the running process**, four-mode smoke tests and latency, log rotation health, coverage audit.
`--heavy-provenance` additionally re-embeds sampled rows and compares cosines — equal row counts
only prove nothing shifted, they don't prove you didn't embed with the wrong model.

Measured numbers from the private deployment: 4469 requests, 0 failures; steady state P50/P95/P99 =
155/607/752 ms; 8-way burst 688/1221/1468 ms; private memory +26.5 MB and threads 59→59 over 21
minutes. That proves "it didn't grow during that window", not "there is no leak".

## License

MIT
