# Chessboo Release Notes

## Unreleased

- Added UCI `Threads` support with Lazy-SMP-style independent search workers.
- Multi-threaded searches keep per-worker board, NNUE accumulator, and move-ordering state while sharing the UCI `Hash` transposition table.
- Helper workers search the full root with rotated root ordering, then the controller uses score/depth-weighted thread voting to reduce completed worker results into one final `info`/`bestmove` line.
- Changed UCI `hashfull` reporting to sample only current-generation transposition table entries instead of all occupied slots.
- Added Syzygy WDL/DTZ tablebase support with `SyzygyPath`, `SyzygyProbeDepth`, `Syzygy50MoveRule`, `SyzygyProbeLimit`, root DTZ best-move probing, non-root WDL cutoffs, and UCI `tbhits` reporting. The default `SyzygyPath` is `<empty>` so tablebase probing is opt-in.
- Started the staged board-backend migration by routing engine board, move, piece, square, and attack-generator imports through an internal `chess` backend seam and moving the Syzygy `shakmaty` conversion boundary into `Position`.

## 1.0.0 - 2026-04-30

- Promoted the Bullet-simple v1 NNUE network trained from the original external Viriformat `.bin` source.
- Embedded `nets\chessboo-v1.nnue` as the default `<internal>` evaluator; SHA-256 `6831a27056391c1a12d31e60ee5ff04fc987e26a6b8fb569fa9843dbbfc5fbd1`.
- Replaced the earlier timing workaround with Stockfish-shaped time management: 50-move horizon, low-clock horizon reduction, overhead-reserved `timeLeft`, `0.8097*time - overhead` hard cap, and 512-call hard-clock polling.
- Restored the default UCI `Move Overhead` to Stockfish's 10 ms.
- Validation: `cargo test` passed with 55 library tests and 8 binary tests.
- Validation: `cargo clippy --all-targets -- -D warnings` passed.
- Validation: `cargo build --release` passed.
- Validation: embedded and file net checks both reported checksum `2ebf7b037d18d654`, `bootstrap false`, and startpos eval 96 cp.
- Validation: release UCI smoke reported `id name Chessboo 1.0.0` and advertised `Move Overhead` default 10.
- Validation: `target\release\chessboo.exe bench --depth 6` reported 147814 total nodes.
- Strength gate: fixed-depth 4 fastchess vs `0.5.1` scored 267.5/400, `+122.04 +/- 25.83` Elo.
- Strength gate: TC `1+0.01` fastchess vs `0.5.1` scored 301.5/400, `+194.34 +/- 29.94` Elo, LOS 100%, W/L/D 241/38/121, with no timeout/crash/error lines.
- Preserved binary: `target\release\chessboo-1.0.0.exe`, SHA-256 `6bca68aa3739836f0ab419994a47b5d35daa2125ab8d602b34808f4640ee2d84`.

## 0.5.1 - 2026-04-29

- Added UCI ponder support for the pre-v1 release line.
- Advertised the standard `Ponder` check option so GUIs such as Cute Chess can enable pondering and send `go ponder`/`ponderhit`.
- `bestmove <move>` now includes `ponder <move>` when the completed PV has a legal opponent reply.
- `go ponder ...` now searches without spending the normal clock budget until `ponderhit`.
- If a ponder search completes before `ponderhit`, Chessboo withholds `bestmove` until `ponderhit` or `stop`, matching GUI expectations.
- Emits an explicit final `info` line and `bestmove` together after search completion so no search info can appear after `bestmove`.
- Validation: `cargo test` passed with 50 library tests and 4 binary tests.
- Validation: `cargo clippy --all-targets -- -D warnings` passed.
- Validation: release UCI smoke reported `id name Chessboo 0.5.1`, advertised `option name Ponder type check default false`, emitted `bestmove b1c3 ponder g8f6`, and withheld ponder-search `bestmove` until `ponderhit`.
- Validation: release UCI smoke verified `bestmove` was immediately preceded by a final `info` line and had no trailing `info` lines.
- Validation: `target\release\chessboo.exe bench --depth 6` reported 107799 nodes.
- Preserved binary: `target\release\chessboo-0.5.1.exe`.

## Historical v1 NNUE Work

- Added NNUE runtime infrastructure for the planned v1 evaluator.
- Added `Use NNUE` and `EvalFile` UCI options with an embedded default net and classical fallback.
- Added `datagen`, `annotate`, `packnet`, and `netcheck` developer commands.
- Added a Chessboo `.nnue` parser/validator, P768 feature indexing, accumulator refresh/update tests, and a bootstrap embedded net.
- Added `docs/nnue_v1.md` and a Bullet training example under `training/`.
- Added parallel `datagen`/`annotate`, `augment` descendant sampling, `annotate --nodes 0` for static-label diagnostics, annotation score filtering/clamping, `netcheck --dataset`, and Bullet training knobs for WDL blend and checkpoint cadence.
- Reworked `datagen` to emit Viriformat binpacks directly, with white-relative search scores stored on each move for Bullet's `ViriBinpackLoader`.
- Added `labelbinpack` to replay raw Viriformat data, preserve or recompute search labels, and physically filter to quiet positions where static eval and search score differ by at most a configurable threshold.
- Replaced the v1 training target with a tiny single-perspective `P768_16_1` feed-forward network instead of the failed dual-perspective P768-1024 path.
- Bootstrap fallback sanity check: 400 no-adjudication games vs `chessboo-0.5.0.exe` scored 193.0/400, `-12.17 +/- 25.73` Elo.
- Dense candidate diagnostics on 10k-50k local positions did not pass promotion checks. The best 50k static-label candidate reached 96 cp holdout MAE but scored only 2.5/40 in a fixed-depth sanity match against `chessboo-0.5.0.exe`.
- One-ply child augmentation improved the 50k static-label diagnostic to about 28 cp child holdout MAE, but fixed-depth play remained around `-250 Elo`; a large release run should wait for a better descendant/search-label data recipe.
- Staged Bullet training is now supported through `CHESSBOO_INIT_CHECKPOINT`. A one-ply bootstrap followed by clipped 4-ply random-prefix fine-tuning trained cleanly and improved the fixed-depth smoke test to roughly `-147 Elo`, but it is still below the v1 promotion bar.
- A 372k-position 2k-node search-label fine-tune was tested next. Search-only fine-tuning failed badly (`-370 Elo` fixed-depth smoke), while a mixed search/static fine-tune was safer and improved the smoke test to roughly `-112 Elo`; still not promotable.
- Pivoted the v1 release path from the HalfKP diagnostic experiment to a small P768 architecture. Old HalfKP candidate nets are rejected by the runtime architecture check.
- First P768 release attempt trained cleanly but did not pass strength gates. The best static centipawn checkpoint reached 15 cp child-holdout MAE and 53 cp random-prefix MAE, but scored only 34.0/80 to 34.5/80 in fixed-depth smoke tests. A mixed search/static fine-tune improved search-label fit only modestly and then scored 9.5/80 in a timed 1+0.01 smoke, so it is not promotable.
- Increased the Viriformat quiet threshold from 8 cp to 100 cp for the first 16M-position run. The q100 filter retained 6,152,770 positions and trained a `P768_16_1` checkpoint with checksum `d14fcc93b213245b`; full-binpack fit was 38 cp MAE with 0 cp bias. Fastchess smoke tests vs `0.5.1` were still negative: 33.0/80 at depth 4 (`-61 +/- 66` Elo) and 35.5/80 at 1+0.01 (`-39 +/- 48` Elo), so it is not promotable yet.
- Lower learning-rate sweeps on q100 were tested with independent fresh runs per superbatch count. `lr=1e-3 -> 1e-5` improved the fit to 33 cp MAE by 160 superbatches, but the best timed smoke was still only 39.0/80 (`-9 +/- 54` Elo) and a larger depth-4 check scored 192.0/400 (`-14 +/- 27` Elo), so the net remains below the release bar.
- Additional q75 and q150 search/static-difference datasets were generated from the same 16M-position source and trained for 160 superbatches with `lr=1e-3 -> 1e-5`. q75 retained 5,649,335 positions and reached 30 cp MAE, while q150 retained 6,691,307 positions and reached 39 cp MAE; both failed the fixed-depth screen vs `0.5.1` at 80 games, scoring 32.5/80 and 33.0/80 respectively.
- A larger external Viriformat source, `run_2026-02-09_00-59-25_10000000g-32t-no_tb-classical-n25000.bin`, was filtered to a fixed 16,000,000-position q100 dataset after reading 97,146,562 source positions. The 160-superbatch `lr=1e-3 -> 1e-5` run produced checksum `96bd5560660d31e1`, startpos eval 6 cp, and 47 cp full-binpack MAE. Strength still regressed: 35.5/80 at depth 4 (`-39 +/- 50` Elo), 37.0/80 at 1+0.01 (`-26 +/- 63` Elo), and 176.0/400 in a depth-4 confirmation (`-42 +/- 25` Elo). The first 400-game fastchess attempt crashed near completion, but a retry at lower concurrency completed cleanly with the reported result.
- Replaced the failed `P768_16_1` training/runtime path with Bullet's `examples/simple.rs` architecture: `(768 -> 128)x2 -> 1`, dual-perspective inference, `l1w/l1b` quantized export sections, default Viriformat filtering, WDL blend `0.75`, and StepLR defaults.
- Full Bullet-simple pipeline on the original external `.bin` source completed with checksum `867d78fe708888bd`, startpos eval 85 cp, and 100k-sample raw-binpack MAE 402 cp with 8 cp bias. Strength was clearly positive against `0.5.1`: 261.5/400 at fixed depth 4 (`+110 +/- 28` Elo) and 258.0/400 at 1+0.01 (`+104 +/- 30` Elo). This is the first v1 candidate to clear the +50 Elo screen in both fixed-depth and timed tests.

The checked-in `nets/chessboo-v1.nnue` is now the promoted v1.0.0 network.

## 0.5.0 - Pre-NNUE Baseline

Chessboo 0.5.0 is the final classical-search baseline before starting the v1 NNUE work.

### Strength Work

- Added clustered transposition table storage with four entries per cluster.
- Replaced the previous legal-move SEE path with a faster bitboard SEE.
- Tightened unsafe pruning around PV nodes, checks, mate-adjacent scores, and suspicious reduced depths.
- Improved move ordering with staged selection, killer and counter moves, quiet history, capture history, and quiet-move maluses.
- Added iterative deepening time-budget adjustments for fail-lows, score drops, and unstable root best moves.
- Kept classical evaluation unchanged for the final baseline to avoid mixing low-confidence eval changes into the NNUE starting point.

### Validation

- `cargo test`: 32 passed.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo run --release -- bench --depth 6`: 107799 nodes.
- Preserved binary: `target\release\chessboo-0.5.0.exe`.

### Release Match

Final no-adjudication fastchess match against the previous preserved release:

```text
chessboo-0.5.0 vs chessboo-0.4.0
TC: 1+0.01
Book: target\books\ianfab-chess.epd
Hash: 16 MB
Games: 400
Score: 249.0/400 (62.25%)
Elo: +86.89 +/- 27.05
LOS: 100.00%
W/L/D: 159/61/180
Log: target\ratings\chessboo-0.5.0_vs_0.4.0_final_noadjud_400g.log
```

### External Anchor Estimate

The Stash anchor estimate is recorded in `stash_ccrl_rating_estimate.txt`.

Approximate Chessboo 0.5.0 CCRL Blitz estimate: 2250, practical range 2200-2290.

### Next

Start v1 NNUE work from this baseline. Initial NNUE scope should cover feature format, data generation, training labels, trainer integration, inference, and a classical-eval fallback path.
