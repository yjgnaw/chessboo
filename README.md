<div align="center">

  <img src="logo.png" alt="Chessboo logo" width="128" height="128" />

  <h3>Chessboo</h3>

  A~~n original~~ UCI-compliant chess engine written in Rust.
</div>

It features a custom alpha-beta search, time management, Syzygy tablebase probing, and supports classical evaluation as well as NNUE.

## Features

- **Search**: Iterative deepening, fail-soft negamax, Alpha-Beta, Principal Variation Search (PVS), Quiescence Search, and Lazy-SMP multi-threading.
- **Heuristics**: Null Move Pruning, Reverse Futility Pruning, ProbCut, Late Move Reductions (LMR), Late Move Pruning, SEE-based capture ordering, and History Heuristics.
- **Evaluation**:
  - NNUE evaluation (`(768 -> 128)x2 -> 1` architecture).
  - Tapered classical evaluation (material, piece-square tables, mobility, pawn structure, king safety).
- **Tablebases**: Full Syzygy WDL/DTZ tablebase probing support.
- **UCI Protocol**: Fully compliant, supporting GUI integration, multi-threading, pondering, and rich configuration options.

## Building from Source

```shell
git clone https://www.github.com/yjgnaw/chessboo.git
cd chessboo
cargo build --release
```

The compiled binary will be located at `target/release/chessboo` (or `target\release\chessboo.exe` on Windows).

## Usage

Chessboo is an engine, not a graphical chess program. To play against it or use it for analysis, you should install a UCI-compatible Graphical User Interface such as:

- [Cutechess](https://cutechess.com/)
- [Arena Chess GUI](http://www.playwitharena.de/)
- [BanksiaGUI](https://banksiagui.com/)
- [En Croissant](https://encroissant.org/)

Add the built `chessboo` binary to your chosen GUI as a UCI engine.

### UCI Options

You can configure Chessboo through the GUI or directly via UCI commands. Available options include:

- `Hash`: Size of the transposition table in MB.
- `Threads`: Number of search threads (Lazy SMP).
- `SyzygyPath`: Path to your downloaded Syzygy WDL/DTZ files.
- `SyzygyProbeDepth` / `SyzygyProbeLimit` / `Syzygy50MoveRule`: Advanced tablebase probing configuration.
- `Use NNUE`: Toggle the NNUE evaluator.
- `EvalFile`: Path to an external `.nnue` file.
- `Move Overhead`: Configurable safety margin for time management in milliseconds.
