# Chessboo

Current version: `0.5.0`.

Chessboo is an original Rust UCI chess engine. It uses `cozy-chess` for legal move generation and implements its own position wrapper, classical evaluation, alpha-beta search, transposition table, time management, UCI loop, and developer commands.

## Build

```powershell
cargo build --release
```

The engine binary is:

```powershell
target\release\chessboo.exe
```

## Commands

```powershell
target\release\chessboo.exe uci
target\release\chessboo.exe perft --depth 4
target\release\chessboo.exe perft --fen "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1" --depth 2
target\release\chessboo.exe bench --depth 6
target\release\chessboo.exe selfplay --games 1 --depth 2 --nodes 5000 --plies 40
```

## UCI Options

- `Hash`: transposition table size in MB.
- `Move Overhead`: milliseconds reserved to avoid time forfeits.
- `Clear Hash`: clears the transposition table.

## Current Engine Features

- UCI support for `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`, and `setoption`.
- UCI treats bare `go` as `go infinite`.
- Search with iterative deepening, fail-soft negamax, alpha-beta, principal variation search, quiescence, transposition table, mate scores, null-move pruning, reverse futility pruning, ProbCut, late move pruning, history-sensitive late move reductions, extended futility pruning, SEE-based bad-capture pruning, killer/counter-move/history ordering with quiet-move maluses, and principal variation reporting.
- Classical tapered evaluation with material, piece-square terms, mobility, pawn structure, passed pawns, bishop pair, rook files, king safety, endgame mop-up, and threats.
- Legality-aware static exchange evaluation for capture ordering and quiescence pruning.
- Time management with separate soft and hard search budgets for real clock controls, tuned to avoid front-loading the clock in increment games.
- Perft, bench, and bounded self-play developer commands.

## Next Strength Work

- Tune evaluation weights with self-play or Texel-style data.
- Add multi-threaded search after single-threaded strength stabilizes.
- Add a reproducible NNUE pipeline later, likely with `bullet`.
