use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use chessboo::eval;
use chessboo::nnue::{self, INTERNAL_EVAL_FILE, NnueNet};
use chessboo::perft::perft;
use chessboo::position::Position;
use chessboo::search::{SearchLimits, SearchOptions, Searcher};
use chessboo::tt::TranspositionTable;
use shakmaty::{Color, Move as EngineMove, Role as EnginePiece, Square as EngineSquare};
use viriformat::{
    chess::{
        board::{Board as ViriBoard, DrawType, GameOutcome, WinType},
        chessmove::{Move as ViriMove, MoveFlags as ViriMoveFlags},
        piece::PieceType as ViriPieceType,
        types::Square as ViriSquare,
    },
    dataformat::Filter as ViriFilter,
    dataformat::{Game as ViriGame, WDL as ViriWdl},
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("uci");

    match command {
        "uci" => chessboo::uci::run().map_err(|err| err.to_string()),
        "perft" => {
            args.remove(0);
            run_perft(&args)
        }
        "bench" => {
            args.remove(0);
            run_bench(&args)
        }
        "selfplay" => {
            args.remove(0);
            run_selfplay(&args)
        }
        "datagen" => {
            args.remove(0);
            run_datagen(&args)
        }
        "annotate" => {
            args.remove(0);
            run_annotate(&args)
        }
        "labelbinpack" => {
            args.remove(0);
            run_labelbinpack(&args)
        }
        "countbinpack" => {
            args.remove(0);
            run_countbinpack(&args)
        }
        "augment" => {
            args.remove(0);
            run_augment(&args)
        }
        "packnet" => {
            args.remove(0);
            run_packnet(&args)
        }
        "netcheck" => {
            args.remove(0);
            run_netcheck(&args)
        }
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command `{other}`; try `chessboo --help`")),
    }
}

fn run_perft(args: &[String]) -> Result<(), String> {
    let fen = value_after(args, "--fen").unwrap_or_else(|| Position::STARTPOS_FEN.to_string());
    let depth = value_after(args, "--depth")
        .or_else(|| args.first().cloned())
        .ok_or("perft needs --depth <n>")?
        .parse::<u32>()
        .map_err(|_| "depth must be a positive integer".to_string())?;

    let position = Position::from_fen(&fen).map_err(|err| err.to_string())?;
    let started = std::time::Instant::now();
    let nodes = perft(&position, depth);
    let elapsed = started.elapsed();
    let nps = if elapsed.as_millis() > 0 {
        nodes.saturating_mul(1000) / elapsed.as_millis() as u64
    } else {
        nodes
    };
    println!("fen {fen}");
    println!("depth {depth}");
    println!("nodes {nodes}");
    println!("time_ms {}", elapsed.as_millis());
    println!("nps {nps}");
    Ok(())
}

fn run_bench(args: &[String]) -> Result<(), String> {
    let depth = value_after(args, "--depth")
        .unwrap_or_else(|| "6".to_string())
        .parse::<u32>()
        .map_err(|_| "depth must be a positive integer".to_string())?;
    let options = SearchOptions::default();
    let positions = [
        Position::STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPPBQPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/2K5/3Q4 w - - 0 1",
        "rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 8",
    ];

    let mut total_nodes = 0_u64;
    let started = std::time::Instant::now();
    for fen in positions {
        let position = Position::from_fen(fen).map_err(|err| err.to_string())?;
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let limits = SearchLimits {
            depth: Some(depth),
            ..SearchLimits::default()
        };
        let mut searcher = Searcher::new(position, options.clone(), limits, tt, stop, None);
        let outcome = searcher.search();
        total_nodes = total_nodes.saturating_add(outcome.nodes);
        println!(
            "bench depth {} score {} nodes {} bestmove {} fen {}",
            outcome.depth,
            outcome.score,
            outcome.nodes,
            outcome
                .best_move
                .map(|mv| outcome.root.to_uci(mv))
                .unwrap_or_else(|| "0000".to_string()),
            fen
        );
    }
    let elapsed = started.elapsed();
    let nps = if elapsed.as_millis() > 0 {
        total_nodes.saturating_mul(1000) / elapsed.as_millis() as u64
    } else {
        total_nodes
    };
    println!(
        "bench total_nodes {total_nodes} time_ms {} nps {nps}",
        elapsed.as_millis()
    );
    Ok(())
}

fn run_selfplay(args: &[String]) -> Result<(), String> {
    let games = value_after(args, "--games")
        .unwrap_or_else(|| "2".to_string())
        .parse::<usize>()
        .map_err(|_| "games must be a positive integer".to_string())?;
    let depth = value_after(args, "--depth")
        .unwrap_or_else(|| "3".to_string())
        .parse::<u32>()
        .map_err(|_| "depth must be a positive integer".to_string())?;
    let nodes = value_after(args, "--nodes")
        .unwrap_or_else(|| "20000".to_string())
        .parse::<u64>()
        .map_err(|_| "nodes must be a positive integer".to_string())?;
    let max_plies = value_after(args, "--plies")
        .unwrap_or_else(|| "160".to_string())
        .parse::<usize>()
        .map_err(|_| "plies must be a positive integer".to_string())?;

    for game in 1..=games {
        let mut position = Position::startpos();
        let mut ply = 0;
        while !position.is_terminal() && ply < max_plies {
            let options = SearchOptions {
                hash_mb: 8,
                ..SearchOptions::default()
            };
            let limits = SearchLimits {
                depth: Some(depth),
                nodes: Some(nodes),
                ..SearchLimits::default()
            };
            let stop = Arc::new(AtomicBool::new(false));
            let tt = TranspositionTable::new(options.hash_mb);
            let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
            let outcome = searcher.search();
            let Some(best) = outcome.best_move else {
                break;
            };
            position.play(best).map_err(|err| err.to_string())?;
            ply += 1;
        }
        println!(
            "selfplay game {game} plies {ply} result {} final_fen {}",
            position.result_string(),
            position.board()
        );
    }
    Ok(())
}

fn run_datagen(args: &[String]) -> Result<(), String> {
    let positions = value_after(args, "--positions")
        .ok_or("datagen needs --positions <n>")?
        .parse::<usize>()
        .map_err(|_| "positions must be a positive integer".to_string())?;
    let out = value_after(args, "--out").ok_or("datagen needs --out <path>")?;
    let book =
        value_after(args, "--book").unwrap_or_else(|| "target/books/ianfab-chess.epd".to_string());
    let seed = value_after(args, "--seed")
        .unwrap_or_else(|| "1".to_string())
        .parse::<u64>()
        .map_err(|_| "seed must be an integer".to_string())?;
    let threads = value_after(args, "--threads")
        .unwrap_or_else(|| "1".to_string())
        .parse::<usize>()
        .map_err(|_| "threads must be a positive integer".to_string())?
        .max(1);

    if let Some(parent) = Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }
    let book_positions = Arc::new(load_book_positions(&book)?);

    let (progress, progress_handle) = start_datagen_progress(positions);
    let result =
        write_datagen_binpack_file(&out, positions, book_positions, seed, threads, &progress);
    progress.finish();
    let _ = progress_handle.join();
    let stats = result?;
    println!(
        "datagen wrote {} positions in {} games to {out}",
        stats.positions, stats.games
    );
    Ok(())
}

fn write_datagen_binpack_file(
    out: &str,
    positions: usize,
    book_positions: Arc<Vec<String>>,
    seed: u64,
    threads: usize,
    progress: &Arc<DatagenProgress>,
) -> Result<DatagenStats, String> {
    if threads == 1 {
        let file =
            fs::File::create(out).map_err(|err| format!("could not create `{out}`: {err}"))?;
        let mut writer = BufWriter::new(file);
        let stats = write_datagen_binpack(
            &mut writer,
            positions,
            &book_positions,
            seed,
            Some(progress),
        )?;
        writer.flush().map_err(|err| err.to_string())?;
        return Ok(stats);
    }

    let file = fs::File::create(out).map_err(|err| format!("could not create `{out}`: {err}"))?;
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));
    let mut handles = Vec::with_capacity(threads);
    for index in 0..threads {
        let target = positions / threads + usize::from(index < positions % threads);
        if target == 0 {
            continue;
        }
        let worker_book = Arc::clone(&book_positions);
        let worker_writer = Arc::clone(&writer);
        let worker_progress = Arc::clone(progress);
        let worker_seed =
            seed.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1));
        handles.push(std::thread::spawn(move || {
            write_datagen_binpack_with_sink(
                target,
                &worker_book,
                worker_seed,
                |game| {
                    let mut bytes = Vec::new();
                    game.serialise_into(&mut bytes)
                        .map_err(|err| err.to_string())?;
                    let mut writer = worker_writer
                        .lock()
                        .map_err(|_| "datagen output writer mutex poisoned".to_string())?;
                    writer.write_all(&bytes).map_err(|err| err.to_string())
                },
                Some(&worker_progress),
            )
        }));
    }

    let mut stats = DatagenStats::default();
    for handle in handles {
        stats += handle
            .join()
            .map_err(|_| "datagen worker panicked".to_string())??;
    }

    writer
        .lock()
        .map_err(|_| "datagen output writer mutex poisoned".to_string())?
        .flush()
        .map_err(|err| err.to_string())?;
    Ok(stats)
}

#[derive(Clone, Copy, Debug, Default)]
struct DatagenStats {
    positions: usize,
    games: usize,
}

impl std::ops::AddAssign for DatagenStats {
    fn add_assign(&mut self, rhs: Self) {
        self.positions += rhs.positions;
        self.games += rhs.games;
    }
}

struct DatagenProgress {
    target: usize,
    positions: AtomicUsize,
    games: AtomicUsize,
    done: AtomicBool,
    started: Instant,
}

impl DatagenProgress {
    fn new(target: usize) -> Self {
        Self {
            target,
            positions: AtomicUsize::new(0),
            games: AtomicUsize::new(0),
            done: AtomicBool::new(false),
            started: Instant::now(),
        }
    }

    fn add_game(&self, positions: usize) {
        self.positions.fetch_add(positions, Ordering::Relaxed);
        self.games.fetch_add(1, Ordering::Relaxed);
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

fn start_datagen_progress(target: usize) -> (Arc<DatagenProgress>, JoinHandle<()>) {
    let progress = Arc::new(DatagenProgress::new(target));
    let worker_progress = Arc::clone(&progress);
    let handle = std::thread::spawn(move || {
        loop {
            render_datagen_progress(&worker_progress);
            if worker_progress.done.load(Ordering::Relaxed) {
                eprintln!();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
    (progress, handle)
}

fn render_datagen_progress(progress: &DatagenProgress) {
    let current = progress
        .positions
        .load(Ordering::Relaxed)
        .min(progress.target);
    let games = progress.games.load(Ordering::Relaxed);
    let percent = if progress.target == 0 {
        100.0
    } else {
        current as f64 * 100.0 / progress.target as f64
    };
    let width = 32_usize;
    let filled = if progress.target == 0 {
        width
    } else {
        current.saturating_mul(width) / progress.target
    };
    let bar = format!("{}{}", "#".repeat(filled), ".".repeat(width - filled));
    let elapsed = progress.started.elapsed().as_secs().max(1);
    let rate = current as u64 / elapsed;
    eprint!(
        "\rdatagen [{bar}] {current}/{} ({percent:5.1}%) games {games} {rate}/s",
        progress.target
    );
    let _ = std::io::stderr().flush();
}

fn write_datagen_binpack<W: Write>(
    writer: &mut W,
    positions: usize,
    book_positions: &[String],
    seed: u64,
    progress: Option<&DatagenProgress>,
) -> Result<DatagenStats, String> {
    write_datagen_binpack_with_sink(
        positions,
        book_positions,
        seed,
        |game| game.serialise_into(writer).map_err(|err| err.to_string()),
        progress,
    )
}

fn write_datagen_binpack_with_sink<F>(
    positions: usize,
    book_positions: &[String],
    seed: u64,
    mut write_game: F,
    progress: Option<&DatagenProgress>,
) -> Result<DatagenStats, String>
where
    F: FnMut(&ViriGame) -> Result<(), String>,
{
    let mut rng = Lcg::new(seed);
    let mut stats = DatagenStats::default();

    while stats.positions < positions {
        let fen = if book_positions.is_empty() {
            Position::STARTPOS_FEN.to_string()
        } else {
            book_positions[rng.index(book_positions.len())].clone()
        };
        let Ok(mut position) = Position::from_fen(&fen) else {
            continue;
        };
        let Ok(mut viri_board) = viri_board_from_fen(&fen) else {
            continue;
        };
        let mut game = ViriGame::new(&viri_board);
        let remaining = positions - stats.positions;
        let mut ply = 0_usize;
        while !position.is_terminal()
            && ply < 160
            && game.len() < remaining
            && game.len() < ViriGame::MAX_SPLATTABLE_GAME_SIZE
        {
            let Some((mv, score)) = search_classical(&position, 3, 5000).or_else(|| {
                random_legal_move(&position, &mut rng).map(|mv| (mv, eval::evaluate(&position)))
            }) else {
                break;
            };
            let viri_move = viri_move_from_engine(&position, mv)?;
            let white_score = score_to_white_relative(position.side_to_move(), score);
            let move_text = position.to_uci(mv);
            game.add_move(viri_move, clamp_i16(white_score));
            position.play(mv).map_err(|err| err.to_string())?;
            if !viri_board.make_move_simple(viri_move) {
                return Err(format!(
                    "Viriformat rejected generated move {move_text} from {fen}"
                ));
            }
            ply += 1;
        }

        if !game.is_empty() {
            game.set_outcome(viri_outcome(&position));
            write_game(&game)?;
            stats.positions += game.len();
            stats.games += 1;
            if let Some(progress) = progress {
                progress.add_game(game.len());
            }
        }
    }

    Ok(stats)
}

fn run_annotate(args: &[String]) -> Result<(), String> {
    let input = value_after(args, "--in").ok_or("annotate needs --in <path>")?;
    let out = value_after(args, "--out").ok_or("annotate needs --out <path>")?;
    let nodes = value_after(args, "--nodes")
        .unwrap_or_else(|| "50000".to_string())
        .parse::<u64>()
        .map_err(|_| "nodes must be a non-negative integer".to_string())?;
    let threads = value_after(args, "--threads")
        .unwrap_or_else(|| "1".to_string())
        .parse::<usize>()
        .map_err(|_| "threads must be a positive integer".to_string())?
        .max(1);
    let max_abs_score = value_after(args, "--max-abs-score")
        .unwrap_or_else(|| "9999".to_string())
        .parse::<i32>()
        .map_err(|_| "max-abs-score must be a non-negative integer".to_string())?;
    if max_abs_score < 0 {
        return Err("max-abs-score must be a non-negative integer".to_string());
    }
    let clamp_score = value_after(args, "--clamp-score")
        .map(|value| {
            value
                .parse::<i32>()
                .map_err(|_| "clamp-score must be a positive integer".to_string())
        })
        .transpose()?;
    if clamp_score.is_some_and(|score| score <= 0) {
        return Err("clamp-score must be a positive integer".to_string());
    }
    let options = AnnotationOptions {
        nodes,
        max_abs_score,
        clamp_score,
    };

    if let Some(parent) = Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }
    let reader = BufReader::new(
        fs::File::open(&input).map_err(|err| format!("could not open `{input}`: {err}"))?,
    );
    let mut writer = BufWriter::new(
        fs::File::create(&out).map_err(|err| format!("could not create `{out}`: {err}"))?,
    );
    let mut annotated = 0_usize;
    let block_size = 1024 * threads;
    let mut block = Vec::with_capacity(block_size);

    for line in reader.lines() {
        let line = line.map_err(|err| err.to_string())?;
        block.push(line);
        if block.len() >= block_size {
            annotated += write_annotated_block(&block, options, threads, &mut writer)?;
            block.clear();
        }
    }
    if !block.is_empty() {
        annotated += write_annotated_block(&block, options, threads, &mut writer)?;
    }

    writer.flush().map_err(|err| err.to_string())?;
    println!("annotate wrote {annotated} labeled positions to {out}");
    Ok(())
}

fn run_labelbinpack(args: &[String]) -> Result<(), String> {
    let input = value_after(args, "--in").ok_or("labelbinpack needs --in <path>")?;
    let out = value_after(args, "--out").ok_or("labelbinpack needs --out <path>")?;
    let options = LabelBinpackOptions {
        nodes: value_after(args, "--nodes")
            .unwrap_or_else(|| "50000".to_string())
            .parse::<u64>()
            .map_err(|_| "nodes must be a non-negative integer".to_string())?,
        threads: value_after(args, "--threads")
            .unwrap_or_else(|| "1".to_string())
            .parse::<usize>()
            .map_err(|_| "threads must be a positive integer".to_string())?
            .max(1),
        min_ply: value_after(args, "--min-ply")
            .unwrap_or_else(|| "16".to_string())
            .parse::<usize>()
            .map_err(|_| "min-ply must be a non-negative integer".to_string())?,
        quiet_threshold: value_after(args, "--quiet-threshold")
            .unwrap_or_else(|| "32".to_string())
            .parse::<i32>()
            .map_err(|_| "quiet-threshold must be a non-negative integer".to_string())?,
        max_abs_score: value_after(args, "--max-abs-score")
            .unwrap_or_else(|| "2000".to_string())
            .parse::<i32>()
            .map_err(|_| "max-abs-score must be a non-negative integer".to_string())?,
        clamp_score: value_after(args, "--clamp-score")
            .map(|value| {
                value
                    .parse::<i32>()
                    .map_err(|_| "clamp-score must be a positive integer".to_string())
            })
            .transpose()?,
        limit: value_after(args, "--limit")
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "limit must be a positive integer".to_string())
            })
            .transpose()?,
        use_input_labels: has_flag(args, "--use-input-labels"),
    };
    if options.quiet_threshold < 0 {
        return Err("quiet-threshold must be a non-negative integer".to_string());
    }
    if options.max_abs_score < 0 {
        return Err("max-abs-score must be a non-negative integer".to_string());
    }
    if options.clamp_score.is_some_and(|score| score <= 0) {
        return Err("clamp-score must be a positive integer".to_string());
    }
    if options.limit == Some(0) {
        return Err("limit must be a positive integer".to_string());
    }

    if let Some(parent) = Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }

    let (progress, progress_handle) = start_label_progress();
    let result = label_binpack_file(&input, &out, options, Some(&progress));
    progress.finish();
    let _ = progress_handle.join();
    let stats = result?;
    println!(
        "labelbinpack read {} positions in {} games, considered {}, kept {} quiet labeled positions to {out}",
        stats.input_positions, stats.input_games, stats.candidates, stats.kept
    );
    Ok(())
}

fn run_countbinpack(args: &[String]) -> Result<(), String> {
    let input = value_after(args, "--in").ok_or("countbinpack needs --in <path>")?;
    let batch_size = value_after(args, "--batch-size")
        .unwrap_or_else(|| "16384".to_string())
        .parse::<u64>()
        .map_err(|_| "batch-size must be a positive integer".to_string())?;
    if batch_size == 0 {
        return Err("batch-size must be a positive integer".to_string());
    }

    let file = fs::File::open(&input).map_err(|err| format!("could not open `{input}`: {err}"))?;
    let mut reader = BufReader::new(file);
    let filter = ViriFilter::default();
    let mut move_buffer = Vec::new();
    let mut games = 0_u64;
    let mut positions = 0_u64;
    let mut filtered_positions = 0_u64;

    loop {
        let game = match ViriGame::deserialise_from(&mut reader, std::mem::take(&mut move_buffer)) {
            Ok(game) => game,
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("could not read Viriformat game: {err}")),
        };
        games += 1;
        positions += game.len() as u64;
        filtered_positions += game.filter_pass_count(&filter);
        move_buffer = game.into_move_buffer();
    }

    let batches_floor = filtered_positions / batch_size;
    let batches_ceil = filtered_positions.div_ceil(batch_size);
    let remainder = filtered_positions % batch_size;
    println!("countbinpack file {input}");
    println!("countbinpack games {games}");
    println!("countbinpack positions {positions}");
    println!("countbinpack filter default");
    println!("countbinpack filtered_positions {filtered_positions}");
    println!("countbinpack batch_size {batch_size}");
    println!("countbinpack batches_floor {batches_floor}");
    println!("countbinpack batches_ceil {batches_ceil}");
    println!("countbinpack remainder {remainder}");
    println!(
        "countbinpack floor_positions {}",
        batches_floor.saturating_mul(batch_size)
    );
    println!(
        "countbinpack ceil_positions {}",
        batches_ceil.saturating_mul(batch_size)
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct AnnotationOptions {
    nodes: u64,
    max_abs_score: i32,
    clamp_score: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
struct LabelBinpackOptions {
    nodes: u64,
    threads: usize,
    min_ply: usize,
    quiet_threshold: i32,
    max_abs_score: i32,
    clamp_score: Option<i32>,
    limit: Option<usize>,
    use_input_labels: bool,
}

#[derive(Debug)]
struct BinpackCandidate {
    fen: String,
    board: ViriBoard,
    mv: ViriMove,
    input_score: i32,
    outcome: GameOutcome,
}

#[derive(Debug)]
struct LabeledBinpackPosition {
    board: ViriBoard,
    mv: ViriMove,
    score: i16,
    outcome: GameOutcome,
}

#[derive(Clone, Copy, Debug, Default)]
struct LabelBinpackStats {
    input_games: usize,
    input_positions: usize,
    candidates: usize,
    kept: usize,
}

impl std::ops::AddAssign for LabelBinpackStats {
    fn add_assign(&mut self, rhs: Self) {
        self.input_games += rhs.input_games;
        self.input_positions += rhs.input_positions;
        self.candidates += rhs.candidates;
        self.kept += rhs.kept;
    }
}

struct LabelProgress {
    positions: AtomicUsize,
    games: AtomicUsize,
    candidates: AtomicUsize,
    kept: AtomicUsize,
    done: AtomicBool,
    started: Instant,
}

impl LabelProgress {
    fn new() -> Self {
        Self {
            positions: AtomicUsize::new(0),
            games: AtomicUsize::new(0),
            candidates: AtomicUsize::new(0),
            kept: AtomicUsize::new(0),
            done: AtomicBool::new(false),
            started: Instant::now(),
        }
    }

    fn add_read(&self, games: usize, positions: usize, candidates: usize) {
        self.games.fetch_add(games, Ordering::Relaxed);
        self.positions.fetch_add(positions, Ordering::Relaxed);
        self.candidates.fetch_add(candidates, Ordering::Relaxed);
    }

    fn add_kept(&self, kept: usize) {
        self.kept.fetch_add(kept, Ordering::Relaxed);
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

fn start_label_progress() -> (Arc<LabelProgress>, JoinHandle<()>) {
    let progress = Arc::new(LabelProgress::new());
    let worker_progress = Arc::clone(&progress);
    let handle = std::thread::spawn(move || {
        loop {
            render_label_progress(&worker_progress);
            if worker_progress.done.load(Ordering::Relaxed) {
                eprintln!();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
    (progress, handle)
}

fn render_label_progress(progress: &LabelProgress) {
    let positions = progress.positions.load(Ordering::Relaxed);
    let games = progress.games.load(Ordering::Relaxed);
    let candidates = progress.candidates.load(Ordering::Relaxed);
    let kept = progress.kept.load(Ordering::Relaxed);
    let elapsed = progress.started.elapsed().as_secs().max(1);
    let rate = positions as u64 / elapsed;
    eprint!(
        "\rlabelbinpack read {positions} positions in {games} games, candidates {candidates}, kept {kept}, {rate}/s"
    );
    let _ = std::io::stderr().flush();
}

fn label_binpack_file(
    input: &str,
    out: &str,
    options: LabelBinpackOptions,
    progress: Option<&LabelProgress>,
) -> Result<LabelBinpackStats, String> {
    let mut reader = BufReader::new(
        fs::File::open(input).map_err(|err| format!("could not open `{input}`: {err}"))?,
    );
    let mut writer = BufWriter::new(
        fs::File::create(out).map_err(|err| format!("could not create `{out}`: {err}"))?,
    );
    let stats = label_binpack_stream(&mut reader, &mut writer, options, progress)?;
    writer.flush().map_err(|err| err.to_string())?;
    Ok(stats)
}

fn label_binpack_stream<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: LabelBinpackOptions,
    progress: Option<&LabelProgress>,
) -> Result<LabelBinpackStats, String> {
    let mut stats = LabelBinpackStats::default();
    let mut block = Vec::with_capacity(1024 * options.threads);
    let mut move_buffer = Vec::new();

    loop {
        let game = match ViriGame::deserialise_from(reader, std::mem::take(&mut move_buffer)) {
            Ok(game) => game,
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("could not read Viriformat game: {err}")),
        };
        let candidates = collect_label_candidates(&game, options);
        stats.input_games += 1;
        stats.input_positions += game.len();
        stats.candidates += candidates.len();
        if let Some(progress) = progress {
            progress.add_read(1, game.len(), candidates.len());
        }
        block.extend(candidates);
        move_buffer = game.into_move_buffer();

        if block.len() >= 1024 * options.threads {
            write_labeled_candidate_block(&mut block, options, writer, &mut stats, progress)?;
            if options.limit.is_some_and(|limit| stats.kept >= limit) {
                break;
            }
        }
    }

    if !block.is_empty() && options.limit.is_none_or(|limit| stats.kept < limit) {
        write_labeled_candidate_block(&mut block, options, writer, &mut stats, progress)?;
    }
    Ok(stats)
}

fn collect_label_candidates(
    game: &ViriGame,
    options: LabelBinpackOptions,
) -> Vec<BinpackCandidate> {
    let mut board = game.initial_position();
    let outcome = game_outcome_from_wdl(game.outcome());
    let mut candidates = Vec::new();
    for (mv, eval) in &game.moves {
        if binpack_prefilter_accepts(&board, *mv, options) {
            candidates.push(BinpackCandidate {
                fen: board.to_string(),
                board: board.clone(),
                mv: *mv,
                input_score: i32::from(eval.get()),
                outcome,
            });
        }
        board.make_move_simple(*mv);
    }
    candidates
}

fn binpack_prefilter_accepts(
    board: &ViriBoard,
    mv: ViriMove,
    options: LabelBinpackOptions,
) -> bool {
    board.ply() >= options.min_ply
        && !board.in_check()
        && !board.is_tactical(mv)
        && board.n_men() >= 4
}

fn write_labeled_candidate_block<W: Write>(
    block: &mut Vec<BinpackCandidate>,
    options: LabelBinpackOptions,
    writer: &mut W,
    stats: &mut LabelBinpackStats,
    progress: Option<&LabelProgress>,
) -> Result<(), String> {
    let kept = label_candidate_block(block, options)?;
    let mut written = 0_usize;
    for position in kept {
        if options.limit.is_some_and(|limit| stats.kept >= limit) {
            break;
        }
        write_labeled_binpack_position(writer, &position)?;
        stats.kept += 1;
        written += 1;
    }
    if let Some(progress) = progress {
        progress.add_kept(written);
    }
    block.clear();
    Ok(())
}

fn label_candidate_block(
    block: &[BinpackCandidate],
    options: LabelBinpackOptions,
) -> Result<Vec<LabeledBinpackPosition>, String> {
    let chunk_size = block.len().div_ceil(options.threads).max(1);
    let groups = std::thread::scope(|scope| {
        let handles: Vec<_> = block
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|candidate| label_candidate(candidate, options))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut groups = Vec::with_capacity(handles.len());
        for handle in handles {
            groups.push(
                handle
                    .join()
                    .map_err(|_| "labelbinpack worker panicked".to_string())?,
            );
        }
        Ok::<Vec<Vec<LabeledBinpackPosition>>, String>(groups)
    })?;

    Ok(groups.into_iter().flatten().collect())
}

fn label_candidate(
    candidate: &BinpackCandidate,
    options: LabelBinpackOptions,
) -> Option<LabeledBinpackPosition> {
    let position = Position::from_fen(&candidate.fen).ok()?;
    if !training_position_ok(&position) {
        return None;
    }
    let static_score = static_score_white_relative(&position);
    let raw_score = if options.use_input_labels {
        candidate.input_score
    } else {
        annotate_score(&position, options.nodes)
    };
    if !quiet_label_ok(static_score, raw_score, options) {
        return None;
    }
    let score = options
        .clamp_score
        .map_or(raw_score, |limit| raw_score.clamp(-limit, limit));
    Some(LabeledBinpackPosition {
        board: candidate.board.clone(),
        mv: candidate.mv,
        score: clamp_i16(score),
        outcome: candidate.outcome,
    })
}

fn quiet_label_ok(static_score: i32, search_score: i32, options: LabelBinpackOptions) -> bool {
    search_score.abs() <= options.max_abs_score
        && (search_score - static_score).abs() <= options.quiet_threshold
}

fn write_labeled_binpack_position<W: Write>(
    writer: &mut W,
    position: &LabeledBinpackPosition,
) -> Result<(), String> {
    let mut game = ViriGame::new(&position.board);
    game.set_outcome(position.outcome);
    game.add_move(position.mv, position.score);
    game.serialise_into(writer).map_err(|err| err.to_string())
}

fn game_outcome_from_wdl(wdl: ViriWdl) -> GameOutcome {
    match wdl {
        ViriWdl::Win => GameOutcome::WhiteWin(WinType::Adjudication),
        ViriWdl::Loss => GameOutcome::BlackWin(WinType::Adjudication),
        ViriWdl::Draw => GameOutcome::Draw(DrawType::Adjudication),
    }
}

fn run_augment(args: &[String]) -> Result<(), String> {
    let input = value_after(args, "--in").ok_or("augment needs --in <path>")?;
    let out = value_after(args, "--out").ok_or("augment needs --out <path>")?;
    let samples = value_after(args, "--samples")
        .unwrap_or_else(|| "0".to_string())
        .parse::<usize>()
        .map_err(|_| "samples must be a non-negative integer".to_string())?;
    let plies = value_after(args, "--plies")
        .unwrap_or_else(|| "1".to_string())
        .parse::<usize>()
        .map_err(|_| "plies must be a positive integer".to_string())?
        .max(1);
    let max_input = value_after(args, "--max-input")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "max-input must be a positive integer".to_string())
        })
        .transpose()?;
    let seed = value_after(args, "--seed")
        .unwrap_or_else(|| "1".to_string())
        .parse::<u64>()
        .map_err(|_| "seed must be an integer".to_string())?;
    let include_root = has_flag(args, "--include-root");
    let include_prefixes = has_flag(args, "--include-prefixes");
    if samples == 0 && plies > 1 {
        return Err("augment needs --samples <n> when --plies is greater than 1".to_string());
    }

    if let Some(parent) = Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }

    let reader = BufReader::new(
        fs::File::open(&input).map_err(|err| format!("could not open `{input}`: {err}"))?,
    );
    let mut writer = BufWriter::new(
        fs::File::create(&out).map_err(|err| format!("could not create `{out}`: {err}"))?,
    );
    let mut rng = Lcg::new(seed);
    let mut read = 0_usize;
    let mut written = 0_usize;

    for line in reader.lines() {
        if max_input.is_some_and(|max| read >= max) {
            break;
        }
        let line = line.map_err(|err| err.to_string())?;
        let Some((fen, result)) = parse_datagen_line(&line) else {
            continue;
        };
        let Ok(position) = Position::from_fen(&fen) else {
            continue;
        };
        read += 1;

        if include_root && training_position_ok(&position) {
            writeln!(writer, "{} | 0 | {result}", position.board())
                .map_err(|err| err.to_string())?;
            written += 1;
        }

        if samples == 0 {
            let moves = position.legal_moves();
            for mv in moves {
                let child = position.after_move(mv);
                if training_position_ok(&child) {
                    writeln!(writer, "{} | 0 | {result}", child.board())
                        .map_err(|err| err.to_string())?;
                    written += 1;
                }
            }
        } else {
            for _ in 0..samples {
                let mut child = position.clone();
                let mut reached_target = true;
                for ply in 0..plies {
                    let moves = child.legal_moves();
                    if moves.is_empty() {
                        reached_target = false;
                        break;
                    }
                    child = child.after_move(moves[rng.index(moves.len())]);
                    if include_prefixes && training_position_ok(&child) {
                        writeln!(writer, "{} | 0 | {result}", child.board())
                            .map_err(|err| err.to_string())?;
                        written += 1;
                    }
                    if child.is_terminal() && ply + 1 < plies {
                        reached_target = false;
                        break;
                    }
                }
                if reached_target && !include_prefixes && training_position_ok(&child) {
                    writeln!(writer, "{} | 0 | {result}", child.board())
                        .map_err(|err| err.to_string())?;
                    written += 1;
                }
            }
        }
    }

    writer.flush().map_err(|err| err.to_string())?;
    println!("augment read {read} positions and wrote {written} positions to {out}");
    Ok(())
}

fn write_annotated_block<W: Write>(
    block: &[String],
    options: AnnotationOptions,
    threads: usize,
    writer: &mut W,
) -> Result<usize, String> {
    let chunk_size = block.len().div_ceil(threads).max(1);
    let groups = std::thread::scope(|scope| {
        let handles: Vec<_> = block
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|line| annotate_line(line, options))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut groups = Vec::with_capacity(handles.len());
        for handle in handles {
            groups.push(
                handle
                    .join()
                    .map_err(|_| "annotate worker panicked".to_string())?,
            );
        }
        Ok::<Vec<Vec<String>>, String>(groups)
    })?;

    let mut written = 0_usize;
    for group in groups {
        for line in group {
            writeln!(writer, "{line}").map_err(|err| err.to_string())?;
            written += 1;
        }
    }
    Ok(written)
}

fn annotate_line(line: &str, options: AnnotationOptions) -> Option<String> {
    let (fen, result) = parse_datagen_line(line)?;
    let position = Position::from_fen(&fen).ok()?;
    if !training_position_ok(&position) {
        return None;
    }
    let raw_score = annotate_score(&position, options.nodes);
    if raw_score.abs() > options.max_abs_score {
        return None;
    }
    let score = options
        .clamp_score
        .map_or(raw_score, |limit| raw_score.clamp(-limit, limit));
    Some(format!("{fen} | {score} | {result}"))
}

fn run_packnet(args: &[String]) -> Result<(), String> {
    let checkpoint = value_after(args, "--checkpoint").ok_or("packnet needs --checkpoint <dir>")?;
    let out = value_after(args, "--out").ok_or("packnet needs --out <path>")?;
    let requested_format = value_after(args, "--format")
        .map(|value| parse_packnet_format(&value))
        .transpose()?
        .flatten();
    let checkpoint = Path::new(&checkpoint);
    let quantised = checkpoint.join("quantised.bin");
    if quantised.exists() {
        if NnueNet::from_file(&quantised).is_ok() {
            copy_file(&quantised, Path::new(&out))?;
            println!(
                "packnet copied existing Chessboo NNUE {}",
                quantised.display()
            );
            return run_netcheck(&["--file".to_string(), out]);
        }
        let bytes = fs::read(&quantised)
            .map_err(|err| format!("could not read `{}`: {err}", quantised.display()))?;
        for format in packnet_candidates(requested_format) {
            let expected = packnet_payload_len(format);
            if bytes.len() >= expected && bytes.len() - expected < 64 {
                if let Some(parent) = Path::new(&out).parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent)
                        .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
                }
                write_packnet_payload(format, &bytes[..expected], &out)
                    .map_err(|err| err.to_string())?;
                println!(
                    "packnet wrapped Bullet {} quantised payload {}",
                    format.name(),
                    quantised.display()
                );
                return run_netcheck(&["--file".to_string(), out]);
            }
        }
    }

    let format = match requested_format {
        Some(format) => format,
        None => detect_packnet_section_format(checkpoint)?,
    };
    let mut payload = Vec::new();
    for (name, expected_len) in format.section_lengths() {
        let path = checkpoint.join(name);
        let bytes =
            fs::read(&path).map_err(|err| format!("could not read `{}`: {err}", path.display()))?;
        if bytes.len() != expected_len {
            return Err(format!(
                "`{}` has {} bytes, expected {expected_len}",
                path.display(),
                bytes.len()
            ));
        }
        payload.extend_from_slice(&bytes);
    }

    if let Some(parent) = Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }
    write_packnet_payload(format, &payload, &out).map_err(|err| err.to_string())?;
    println!("packnet wrote {} {out}", format.name());
    run_netcheck(&["--file".to_string(), out])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackNetFormat {
    V1,
    V2,
}

impl PackNetFormat {
    fn name(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }

    fn section_lengths(self) -> [(&'static str, usize); 4] {
        match self {
            Self::V1 => nnue::dense_section_lengths(),
            Self::V2 => nnue::output_bucket_section_lengths(),
        }
    }
}

fn parse_packnet_format(value: &str) -> Result<Option<PackNetFormat>, String> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "v1" | "dense" => Ok(Some(PackNetFormat::V1)),
        "v2" | "output-buckets" | "output_buckets" => Ok(Some(PackNetFormat::V2)),
        _ => Err("packnet --format must be auto, v1, or v2".to_string()),
    }
}

fn packnet_candidates(requested: Option<PackNetFormat>) -> Vec<PackNetFormat> {
    requested.map_or_else(
        || vec![PackNetFormat::V2, PackNetFormat::V1],
        |format| vec![format],
    )
}

fn packnet_payload_len(format: PackNetFormat) -> usize {
    format
        .section_lengths()
        .into_iter()
        .map(|(_, len)| len)
        .sum()
}

fn write_packnet_payload(
    format: PackNetFormat,
    payload: &[u8],
    out: impl AsRef<Path>,
) -> Result<(), nnue::NnueError> {
    match format {
        PackNetFormat::V1 => nnue::write_dense_payload(payload, out),
        PackNetFormat::V2 => nnue::write_output_bucket_payload(payload, out),
    }
}

fn detect_packnet_section_format(checkpoint: &Path) -> Result<PackNetFormat, String> {
    let l0w = checkpoint.join("l0w.bin");
    let len = fs::metadata(&l0w)
        .map_err(|err| format!("could not read `{}`: {err}", l0w.display()))?
        .len() as usize;
    let mut v1_l0w = 0;
    let mut v2_l0w = 0;
    for format in [PackNetFormat::V2, PackNetFormat::V1] {
        let expected = format
            .section_lengths()
            .into_iter()
            .find(|(name, _)| *name == "l0w.bin")
            .map(|(_, len)| len)
            .expect("l0w section exists");
        match format {
            PackNetFormat::V1 => v1_l0w = expected,
            PackNetFormat::V2 => v2_l0w = expected,
        }
        if len == expected {
            return Ok(format);
        }
    }
    Err(format!(
        "`{}` has {len} bytes; expected {} for v1 or {} for v2",
        l0w.display(),
        v1_l0w,
        v2_l0w
    ))
}

fn run_netcheck(args: &[String]) -> Result<(), String> {
    let file = value_after(args, "--file").unwrap_or_else(|| "nets/chessboo-v1.nnue".to_string());
    let dataset = value_after(args, "--dataset");
    let binpack_dataset = value_after(args, "--binpack-dataset");
    if dataset.is_some() && binpack_dataset.is_some() {
        return Err("netcheck accepts only one of --dataset or --binpack-dataset".to_string());
    }
    let limit = value_after(args, "--limit")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "limit must be a positive integer".to_string())
        })
        .transpose()?;
    let net = if file == INTERNAL_EVAL_FILE {
        NnueNet::embedded().map_err(|err| err.to_string())?
    } else {
        NnueNet::from_file(&file).map_err(|err| err.to_string())?
    };
    net.validate().map_err(|err| err.to_string())?;
    let startpos = Position::startpos();
    println!("net file {file}");
    println!("net checksum {:016x}", net.checksum());
    println!("net bootstrap {}", net.is_bootstrap());
    println!("net startpos_eval {}", net.evaluate_position(&startpos));
    if let Some(dataset) = dataset {
        run_net_dataset_check(&net, &dataset, limit)?;
    }
    if let Some(dataset) = binpack_dataset {
        run_net_binpack_check(&net, &dataset, limit)?;
    }
    Ok(())
}

fn run_net_dataset_check(net: &NnueNet, dataset: &str, limit: Option<usize>) -> Result<(), String> {
    let reader = BufReader::new(
        fs::File::open(dataset).map_err(|err| format!("could not open `{dataset}`: {err}"))?,
    );
    let mut count = 0_usize;
    let mut sum_abs_error = 0_i64;
    let mut sum_error = 0_i64;
    let mut max_abs_error = 0_i32;

    for line in reader.lines() {
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
        let line = line.map_err(|err| err.to_string())?;
        let Some((fen, expected)) = parse_labeled_line(&line) else {
            continue;
        };
        let Ok(position) = Position::from_fen(&fen) else {
            continue;
        };
        let eval = match position.side_to_move() {
            Color::White => net.evaluate_position(&position),
            Color::Black => -net.evaluate_position(&position),
        };
        let error = eval - expected;
        let abs_error = error.abs();
        count += 1;
        sum_abs_error += i64::from(abs_error);
        sum_error += i64::from(error);
        max_abs_error = max_abs_error.max(abs_error);
    }

    if count == 0 {
        return Err("dataset check found no labeled positions".to_string());
    }

    println!("net dataset {dataset}");
    println!("net dataset_count {count}");
    println!("net dataset_mae {}", sum_abs_error / count as i64);
    println!("net dataset_bias {}", sum_error / count as i64);
    println!("net dataset_max_abs_error {max_abs_error}");
    Ok(())
}

fn run_net_binpack_check(net: &NnueNet, dataset: &str, limit: Option<usize>) -> Result<(), String> {
    let mut reader = BufReader::new(
        fs::File::open(dataset).map_err(|err| format!("could not open `{dataset}`: {err}"))?,
    );
    let mut move_buffer = Vec::new();
    let mut count = 0_usize;
    let mut sum_abs_error = 0_i64;
    let mut sum_error = 0_i64;
    let mut max_abs_error = 0_i32;

    'games: loop {
        let game = match ViriGame::deserialise_from(&mut reader, std::mem::take(&mut move_buffer)) {
            Ok(game) => game,
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("could not read Viriformat game: {err}")),
        };
        let mut board = game.initial_position();
        for (mv, expected) in &game.moves {
            if limit.is_some_and(|limit| count >= limit) {
                break 'games;
            }
            let fen = board.to_string();
            if let Ok(position) = Position::from_fen(&fen) {
                let eval = match position.side_to_move() {
                    Color::White => net.evaluate_position(&position),
                    Color::Black => -net.evaluate_position(&position),
                };
                let error = eval - i32::from(expected.get());
                let abs_error = error.abs();
                count += 1;
                sum_abs_error += i64::from(abs_error);
                sum_error += i64::from(error);
                max_abs_error = max_abs_error.max(abs_error);
            }
            board.make_move_simple(*mv);
        }
        move_buffer = game.into_move_buffer();
    }

    if count == 0 {
        return Err("binpack dataset check found no labeled positions".to_string());
    }

    println!("net binpack_dataset {dataset}");
    println!("net binpack_dataset_count {count}");
    println!("net binpack_dataset_mae {}", sum_abs_error / count as i64);
    println!("net binpack_dataset_bias {}", sum_error / count as i64);
    println!("net binpack_dataset_max_abs_error {max_abs_error}");
    Ok(())
}

fn value_after(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == key)
        .map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|arg| arg == key)
}

fn load_book_positions(path: &str) -> Result<Vec<String>, String> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(Vec::new()),
    };
    let reader = BufReader::new(file);
    let mut positions = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|err| err.to_string())?;
        let fields: Vec<_> = line.split_whitespace().take(4).collect();
        if fields.len() == 4 {
            positions.push(format!(
                "{} {} {} {} 0 1",
                fields[0], fields[1], fields[2], fields[3]
            ));
        }
    }
    Ok(positions)
}

fn training_position_ok(position: &Position) -> bool {
    !position.is_terminal() && position.checkers().is_empty()
}

fn search_classical(position: &Position, depth: u32, nodes: u64) -> Option<(EngineMove, i32)> {
    let options = SearchOptions {
        hash_mb: 8,
        use_nnue: false,
        eval_file: None,
        nnue: None,
        ..SearchOptions::default()
    };
    let limits = SearchLimits {
        depth: Some(depth),
        nodes: Some(nodes),
        ..SearchLimits::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    let tt = TranspositionTable::new(options.hash_mb);
    let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
    let outcome = searcher.search();
    outcome.best_move.map(|mv| (mv, outcome.score))
}

fn random_legal_move(position: &Position, rng: &mut Lcg) -> Option<EngineMove> {
    let moves = position.legal_moves();
    (!moves.is_empty()).then(|| moves[rng.index(moves.len())])
}

fn viri_board_from_fen(fen: &str) -> Result<ViriBoard, String> {
    let mut board = ViriBoard::new();
    board
        .set_from_fen(fen, false)
        .map_err(|err| format!("Viriformat could not parse `{fen}`: {err}"))?;
    Ok(board)
}

fn viri_move_from_engine(position: &Position, mv: EngineMove) -> Result<ViriMove, String> {
    let from = viri_square(engine_move_from(mv)?);
    let to = viri_square(mv.to());
    if let Some(promotion) = mv.promotion() {
        return Ok(ViriMove::new_with_promo(
            from,
            to,
            viri_promotion_piece(promotion)?,
        ));
    }
    if is_internal_castle_move(position, mv) {
        return Ok(ViriMove::new_with_flags(from, to, ViriMoveFlags::Castle));
    }
    if is_en_passant_move(position, mv) {
        return Ok(ViriMove::new_with_flags(from, to, ViriMoveFlags::EnPassant));
    }
    Ok(ViriMove::new(from, to))
}

fn viri_square(square: EngineSquare) -> ViriSquare {
    ViriSquare::new(square.to_u32() as u8).expect("engine square index is always valid")
}

fn viri_promotion_piece(piece: EnginePiece) -> Result<ViriPieceType, String> {
    match piece {
        EnginePiece::Knight => Ok(ViriPieceType::Knight),
        EnginePiece::Bishop => Ok(ViriPieceType::Bishop),
        EnginePiece::Rook => Ok(ViriPieceType::Rook),
        EnginePiece::Queen => Ok(ViriPieceType::Queen),
        _ => Err(format!("invalid promotion piece {piece:?}")),
    }
}

fn is_internal_castle_move(position: &Position, mv: EngineMove) -> bool {
    let _ = position;
    mv.is_castle()
}

fn is_en_passant_move(position: &Position, mv: EngineMove) -> bool {
    let _ = position;
    mv.is_en_passant()
}

fn engine_move_from(mv: EngineMove) -> Result<EngineSquare, String> {
    mv.from()
        .ok_or_else(|| "Viriformat export does not support drop moves".to_string())
}

fn score_to_white_relative(side_to_move: Color, score: i32) -> i32 {
    match side_to_move {
        Color::White => score,
        Color::Black => -score,
    }
}

fn clamp_i16(score: i32) -> i16 {
    score.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn viri_outcome(position: &Position) -> GameOutcome {
    match position.result_string() {
        "1-0" => GameOutcome::WhiteWin(WinType::Mate),
        "0-1" => GameOutcome::BlackWin(WinType::Mate),
        _ => GameOutcome::Draw(DrawType::Adjudication),
    }
}

fn parse_datagen_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.split('|').map(str::trim);
    let fen = parts.next()?.to_string();
    let _score = parts.next();
    let result = parts.next().unwrap_or("0.5").to_string();
    Some((fen, result))
}

fn parse_labeled_line(line: &str) -> Option<(String, i32)> {
    let mut parts = line.split('|').map(str::trim);
    let fen = parts.next()?.to_string();
    let score = parts.next()?.parse().ok()?;
    Some((fen, score))
}

fn annotate_score(position: &Position, nodes: u64) -> i32 {
    if nodes == 0 {
        return static_score_white_relative(position);
    }

    let options = SearchOptions {
        hash_mb: 16,
        use_nnue: false,
        eval_file: None,
        nnue: None,
        ..SearchOptions::default()
    };
    let limits = SearchLimits {
        nodes: Some(nodes),
        ..SearchLimits::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    let tt = TranspositionTable::new(options.hash_mb);
    let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
    let score = searcher.search().score;
    match position.side_to_move() {
        Color::White => score,
        Color::Black => -score,
    }
}

fn static_score_white_relative(position: &Position) -> i32 {
    let score = eval::evaluate(position);
    match position.side_to_move() {
        Color::White => score,
        Color::Black => -score,
    }
}

fn copy_file(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(parent) = to.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("could not create `{}`: {err}", parent.display()))?;
    }
    fs::copy(from, to).map(|_| ()).map_err(|err| {
        format!(
            "could not copy `{}` to `{}`: {err}",
            from.display(),
            to.display()
        )
    })
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len
    }
}

fn print_help() {
    println!(
        "Chessboo {version}

Usage:
  chessboo uci
  chessboo perft --fen <fen> --depth <n>
  chessboo bench [--depth <n>]
  chessboo selfplay [--games <n>] [--depth <n>] [--nodes <n>] [--plies <n>]
  chessboo datagen --positions <n> --out <binpack> --book <epd> --seed <n> [--threads <n>]
  chessboo annotate --in <path> --out <path> --nodes <n> --threads <n> [--max-abs-score <cp>] [--clamp-score <cp>]
  chessboo labelbinpack --in <raw.binpack> --out <quiet.binpack> [--nodes <n>] [--threads <n>] [--quiet-threshold <cp>] [--min-ply <n>] [--max-abs-score <cp>] [--limit <n>] [--use-input-labels]
  chessboo countbinpack --in <raw.binpack> [--batch-size <n>]
  chessboo augment --in <path> --out <path> [--samples <n>] [--plies <n>] [--max-input <n>] [--include-root] [--include-prefixes]
  chessboo packnet --checkpoint <dir> --out nets/chessboo-v1.nnue [--format auto|v1|v2]
  chessboo netcheck --file <path> [--dataset <text-labels> | --binpack-dataset <viri.binpack>] [--limit <n>]",
        version = chessboo::ENGINE_VERSION
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUEEN_ADVANTAGE: &str = "4k3/8/8/8/8/8/2K5/3Q4 w - - 0 1 | 0 | 0.5";

    fn test_label_options() -> LabelBinpackOptions {
        LabelBinpackOptions {
            nodes: 0,
            threads: 1,
            min_ply: 0,
            quiet_threshold: 0,
            max_abs_score: 2_000,
            clamp_score: None,
            limit: None,
            use_input_labels: false,
        }
    }

    #[test]
    fn annotate_can_clamp_large_scores() {
        let line = annotate_line(
            QUEEN_ADVANTAGE,
            AnnotationOptions {
                nodes: 0,
                max_abs_score: 9_999,
                clamp_score: Some(10),
            },
        )
        .expect("queen advantage should be annotated");

        let (_, score) = parse_labeled_line(&line).expect("annotated score should parse");
        assert_eq!(score, 10);
    }

    #[test]
    fn annotate_can_filter_large_scores() {
        let line = annotate_line(
            QUEEN_ADVANTAGE,
            AnnotationOptions {
                nodes: 0,
                max_abs_score: 10,
                clamp_score: None,
            },
        );

        assert!(line.is_none());
    }

    #[test]
    fn quiet_label_filter_rejects_search_static_divergence() {
        let mut options = test_label_options();
        options.quiet_threshold = 32;
        assert!(quiet_label_ok(100, 132, options));
        assert!(!quiet_label_ok(100, 133, options));
    }

    #[test]
    fn labelbinpack_respects_min_ply_prefilter() {
        let position = Position::startpos();
        let mv = viri_move_from_engine(&position, position.uci_to_move("e2e4").unwrap()).unwrap();
        let board = viri_board_from_fen(Position::STARTPOS_FEN).unwrap();
        let mut options = test_label_options();
        options.min_ply = 16;

        assert!(!binpack_prefilter_accepts(&board, mv, options));
    }

    #[test]
    fn converts_special_moves_to_viriformat() {
        let castle = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let castle_move = castle.uci_to_move("e1g1").unwrap();
        let viri_castle = viri_move_from_engine(&castle, castle_move).unwrap();
        assert!(viri_castle.is_castle());
        assert_eq!(format!("{}", viri_castle.display(false)), "e1g1");

        let ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let ep_move = ep.uci_to_move("e5d6").unwrap();
        assert!(viri_move_from_engine(&ep, ep_move).unwrap().is_ep());

        let promo = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo_move = promo.uci_to_move("a7a8q").unwrap();
        assert_eq!(
            viri_move_from_engine(&promo, promo_move)
                .unwrap()
                .promotion_type(),
            Some(ViriPieceType::Queen)
        );
    }

    #[test]
    fn datagen_writes_readable_viri_binpack() {
        let mut bytes = Vec::new();
        let stats = write_datagen_binpack(&mut bytes, 1, &[], 7, None).unwrap();
        assert_eq!(stats.positions, 1);
        assert_eq!(stats.games, 1);

        let mut reader = BufReader::new(bytes.as_slice());
        let game = ViriGame::deserialise_from(&mut reader, Vec::new()).unwrap();
        assert_eq!(game.len(), 1);
    }

    #[test]
    fn labelbinpack_writes_readable_quiet_one_move_games() {
        let position = Position::startpos();
        let mut board = viri_board_from_fen(Position::STARTPOS_FEN).unwrap();
        let mv = viri_move_from_engine(&position, position.uci_to_move("e2e4").unwrap()).unwrap();
        let mut input_game = ViriGame::new(&board);
        input_game.set_outcome(GameOutcome::Draw(DrawType::Adjudication));
        input_game.add_move(mv, 123);

        let mut input = Vec::new();
        input_game.serialise_into(&mut input).unwrap();
        let mut output = Vec::new();
        let stats = label_binpack_stream(
            &mut BufReader::new(input.as_slice()),
            &mut output,
            test_label_options(),
            None,
        )
        .unwrap();

        assert_eq!(stats.input_positions, 1);
        assert_eq!(stats.candidates, 1);
        assert_eq!(stats.kept, 1);

        let mut reader = BufReader::new(output.as_slice());
        let output_game = ViriGame::deserialise_from(&mut reader, Vec::new()).unwrap();
        assert_eq!(output_game.len(), 1);
        assert_eq!(
            output_game.initial_position().to_string(),
            board.to_string()
        );
        assert_eq!(
            output_game.moves[0].1.get(),
            static_score_white_relative(&position) as i16
        );
        assert!(board.make_move_simple(output_game.moves[0].0));
    }

    #[test]
    fn labelbinpack_can_preserve_input_search_labels() {
        let position = Position::startpos();
        let board = viri_board_from_fen(Position::STARTPOS_FEN).unwrap();
        let mv = viri_move_from_engine(&position, position.uci_to_move("e2e4").unwrap()).unwrap();
        let mut candidate = BinpackCandidate {
            fen: Position::STARTPOS_FEN.to_string(),
            board,
            mv,
            input_score: 7,
            outcome: GameOutcome::Draw(DrawType::Adjudication),
        };
        let mut options = test_label_options();
        options.use_input_labels = true;
        options.quiet_threshold = 10;
        let labeled = label_candidate(&candidate, options).unwrap();
        assert_eq!(labeled.score, 7);

        candidate.input_score = 99;
        assert!(label_candidate(&candidate, options).is_none());
    }
}
