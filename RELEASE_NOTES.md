# Chessboo Release Notes

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
