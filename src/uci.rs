use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use shakmaty::Move;
use vampirc_uci::{UciMessage, UciTimeControl, parse_one};

use crate::nnue::{INTERNAL_EVAL_FILE, NnueNet};
use crate::position::Position;
use crate::search::{
    SearchInfo, SearchLimits, SearchOptions, SearchOutcome, SearchReporter, Searcher,
    format_uci_score,
};
use crate::syzygy::{
    DEFAULT_SYZYGY_PATH, DEFAULT_SYZYGY_PROBE_DEPTH, DEFAULT_SYZYGY_PROBE_LIMIT, EMPTY_SYZYGY_PATH,
    SyzygyTablebase,
};
use crate::tt::TranspositionTable;
use crate::{ENGINE_AUTHOR, ENGINE_NAME, ENGINE_VERSION};

const MAX_THREADS: usize = 512;

pub fn run() -> io::Result<()> {
    let stdin = io::stdin();
    let mut engine = UciEngine::new();

    for line in stdin.lock().lines() {
        let line = line?;
        let should_quit = engine.handle_line(&line);
        if should_quit {
            break;
        }
    }

    engine.stop_search();
    Ok(())
}

struct UciEngine {
    position: Position,
    options: SearchOptions,
    ponder_enabled: bool,
    tt: Option<TranspositionTable>,
    search: Option<SearchHandle>,
}

struct SearchHandle {
    stop: Arc<AtomicBool>,
    ponder_hit: Option<Arc<AtomicBool>>,
    handle: JoinHandle<TranspositionTable>,
}

impl UciEngine {
    fn new() -> Self {
        let mut options = SearchOptions::default();
        options.syzygy = load_syzygy_path(&options.syzygy_path);
        Self {
            position: Position::startpos(),
            tt: Some(TranspositionTable::new(options.hash_mb)),
            options,
            ponder_enabled: false,
            search: None,
        }
    }

    fn handle_line(&mut self, line: &str) -> bool {
        self.reap_finished();
        let ponder = is_ponder_go(line);
        let message = parse_one(line);
        match message {
            UciMessage::Uci => self.identify(),
            UciMessage::IsReady => {
                println!("readyok");
                flush_stdout();
            }
            UciMessage::UciNewGame => {
                self.stop_search();
                self.position = Position::startpos();
                self.clear_hash();
            }
            UciMessage::Position {
                startpos,
                fen,
                moves,
            } => {
                self.stop_search();
                self.set_position(
                    startpos,
                    fen.map(|fen| fen.0),
                    moves.iter().map(ToString::to_string),
                );
            }
            UciMessage::Go {
                time_control,
                search_control,
            } => {
                self.start_search(time_control, search_control, ponder);
            }
            UciMessage::Stop => self.stop_search(),
            UciMessage::Quit => {
                self.stop_search();
                return true;
            }
            UciMessage::SetOption { name, value } => self.set_option(&name, value.as_deref()),
            UciMessage::Debug(_) | UciMessage::Register { .. } | UciMessage::Unknown(_, _) => {}
            UciMessage::PonderHit => self.ponder_hit(),
            _ => {}
        }
        false
    }

    fn identify(&self) {
        println!("id name {ENGINE_NAME} {ENGINE_VERSION}");
        println!("id author {ENGINE_AUTHOR}");
        println!(
            "option name Hash type spin default {} min 1 max 4096",
            SearchOptions::default().hash_mb
        );
        println!(
            "option name Move Overhead type spin default {} min 0 max 5000",
            SearchOptions::default().move_overhead.as_millis()
        );
        println!(
            "option name Threads type spin default {} min 1 max {MAX_THREADS}",
            SearchOptions::default().threads
        );
        println!("option name Ponder type check default false");
        println!("option name Use NNUE type check default true");
        println!("option name EvalFile type string default {INTERNAL_EVAL_FILE}");
        println!("option name SyzygyPath type string default {DEFAULT_SYZYGY_PATH}");
        println!(
            "option name SyzygyProbeDepth type spin default {DEFAULT_SYZYGY_PROBE_DEPTH} min 1 max 100"
        );
        println!("option name Syzygy50MoveRule type check default true");
        println!(
            "option name SyzygyProbeLimit type spin default {DEFAULT_SYZYGY_PROBE_LIMIT} min 0 max 7"
        );
        println!("option name Clear Hash type button");
        println!("uciok");
        flush_stdout();
    }

    fn set_position<I>(&mut self, startpos: bool, fen: Option<String>, moves: I)
    where
        I: IntoIterator<Item = String>,
    {
        let mut position = if startpos {
            Position::startpos()
        } else if let Some(fen) = fen {
            match Position::from_fen(&fen) {
                Ok(position) => position,
                Err(err) => {
                    println!("info string {err}");
                    flush_stdout();
                    return;
                }
            }
        } else {
            Position::startpos()
        };

        for mv in moves {
            if let Err(err) = position.play_uci(&mv) {
                println!("info string could not apply move {mv}: {err}");
                flush_stdout();
                return;
            }
        }
        self.position = position;
    }

    fn set_option(&mut self, name: &str, value: Option<&str>) {
        match name.to_ascii_lowercase().as_str() {
            "hash" => {
                self.stop_search();
                if let Some(value) = value.and_then(|value| value.parse::<usize>().ok()) {
                    self.options.hash_mb = value.clamp(1, 4096);
                    self.tt = Some(TranspositionTable::new(self.options.hash_mb));
                }
            }
            "move overhead" => {
                if let Some(value) = value.and_then(|value| value.parse::<u64>().ok()) {
                    self.options.move_overhead = Duration::from_millis(value.min(5000));
                }
            }
            "threads" => {
                self.stop_search();
                if let Some(value) = value.and_then(|value| value.parse::<usize>().ok()) {
                    self.options.threads = value.clamp(1, MAX_THREADS);
                }
            }
            "ponder" => {
                if let Some(value) = value.and_then(parse_uci_bool) {
                    self.ponder_enabled = value;
                }
            }
            "use nnue" => {
                if let Some(value) = value.and_then(parse_uci_bool) {
                    self.stop_search();
                    self.options.use_nnue = value;
                }
            }
            "evalfile" => {
                if let Some(value) = value {
                    self.stop_search();
                    match load_eval_file(value) {
                        Ok(net) => {
                            self.options.nnue = Some(net);
                            self.options.eval_file = Some(value.to_string());
                        }
                        Err(err) => {
                            println!("info string {err}");
                            flush_stdout();
                        }
                    }
                }
            }
            "syzygypath" => {
                self.stop_search();
                let value = value.unwrap_or(EMPTY_SYZYGY_PATH).trim();
                self.options.syzygy_path = value.to_string();
                self.options.syzygy = load_syzygy_path(value);
                if let Some(syzygy) = &self.options.syzygy {
                    println!(
                        "info string loaded {} Syzygy files from {} directories, max pieces {}",
                        syzygy.loaded_files(),
                        syzygy.directory_count(),
                        syzygy.max_pieces()
                    );
                    flush_stdout();
                } else if !value.is_empty() && !value.eq_ignore_ascii_case(EMPTY_SYZYGY_PATH) {
                    println!("info string could not load SyzygyPath `{value}`");
                    flush_stdout();
                }
            }
            "syzygyprobedepth" => {
                if let Some(value) = value.and_then(|value| value.parse::<u32>().ok()) {
                    self.options.syzygy_probe_depth = value.clamp(1, 100);
                }
            }
            "syzygy50moverule" => {
                if let Some(value) = value.and_then(parse_uci_bool) {
                    self.options.syzygy_50_move_rule = value;
                }
            }
            "syzygyprobelimit" => {
                if let Some(value) = value.and_then(|value| value.parse::<usize>().ok()) {
                    self.options.syzygy_probe_limit = value.clamp(0, 7);
                }
            }
            "clear hash" => {
                self.stop_search();
                self.clear_hash();
            }
            _ => {}
        }
    }

    fn clear_hash(&mut self) {
        if let Some(tt) = &mut self.tt {
            tt.clear();
        } else {
            self.tt = Some(TranspositionTable::new(self.options.hash_mb));
        }
    }

    fn start_search(
        &mut self,
        time_control: Option<UciTimeControl>,
        search_control: Option<vampirc_uci::UciSearchControl>,
        ponder: bool,
    ) {
        self.stop_search();

        let mut limits = SearchLimits::default();
        if let Some(search_control) = search_control {
            limits.depth = search_control.depth.map(u32::from);
            limits.nodes = search_control.nodes;
            for mv in search_control.search_moves {
                if let Ok(internal) = self.position.uci_to_move(&mv.to_string()) {
                    limits.search_moves.push(internal);
                }
            }
        }
        if let Some(time_control) = time_control {
            apply_time_control(&mut limits, time_control);
        }
        let ponder_hit = ponder.then(|| Arc::new(AtomicBool::new(false)));
        if let Some(ponder_hit) = &ponder_hit {
            limits.ponder_hit = Some(Arc::clone(ponder_hit));
        }
        apply_default_go_behavior(&mut limits);

        let position = self.position.clone();
        let options = self.options.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_ponder_hit = ponder_hit.clone();
        let tt = self
            .tt
            .take()
            .unwrap_or_else(|| TranspositionTable::new(options.hash_mb));
        let reporter: SearchReporter = Box::new(print_info);

        let handle = thread::spawn(move || {
            let (outcome, tt) = search_with_threads(
                position,
                options,
                limits,
                tt,
                Arc::clone(&worker_stop),
                reporter,
            );
            let bestmove = format_bestmove(&outcome);
            if let Some(ponder_hit) = &worker_ponder_hit {
                while !ponder_hit.load(Ordering::Relaxed) && !worker_stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            print_final_info_and_bestmove(&outcome, &bestmove);
            tt
        });

        self.search = Some(SearchHandle {
            stop,
            ponder_hit,
            handle,
        });
    }

    fn stop_search(&mut self) {
        if let Some(search) = self.search.take() {
            search.stop.store(true, Ordering::Relaxed);
            match search.handle.join() {
                Ok(tt) => self.tt = Some(tt),
                Err(_) => self.tt = Some(TranspositionTable::new(self.options.hash_mb)),
            }
        }
    }

    fn ponder_hit(&mut self) {
        if let Some(search) = &self.search
            && let Some(ponder_hit) = &search.ponder_hit
        {
            ponder_hit.store(true, Ordering::Relaxed);
        }
    }

    fn reap_finished(&mut self) {
        if self
            .search
            .as_ref()
            .is_some_and(|search| search.handle.is_finished())
            && let Some(search) = self.search.take()
        {
            match search.handle.join() {
                Ok(tt) => self.tt = Some(tt),
                Err(_) => self.tt = Some(TranspositionTable::new(self.options.hash_mb)),
            }
        }
    }
}

struct WorkerResult {
    outcome: SearchOutcome,
    tt: TranspositionTable,
}

struct ThreadedReporter {
    stats: Arc<Mutex<ThreadedReporterStats>>,
    reporter: Arc<Mutex<SearchReporter>>,
}

#[derive(Debug)]
struct ThreadedReporterStats {
    latest: Vec<Option<SearchInfo>>,
    emitted_depth: u32,
}

impl ThreadedReporter {
    fn new(worker_count: usize, reporter: SearchReporter) -> Self {
        Self {
            stats: Arc::new(Mutex::new(ThreadedReporterStats {
                latest: vec![None; worker_count],
                emitted_depth: 0,
            })),
            reporter: Arc::new(Mutex::new(reporter)),
        }
    }

    fn reporter_for(&self, worker_index: usize) -> SearchReporter {
        let stats = Arc::clone(&self.stats);
        let reporter = Arc::clone(&self.reporter);
        Box::new(move |info| {
            let mut stats = stats.lock().expect("threaded reporter stats lock poisoned");
            let aggregated = aggregate_thread_info(&mut stats, worker_index, info);
            if let Some(aggregated) = aggregated {
                let reporter = reporter.lock().expect("threaded reporter lock poisoned");
                reporter.as_ref()(aggregated);
            }
        })
    }
}

fn aggregate_thread_info(
    stats: &mut ThreadedReporterStats,
    worker_index: usize,
    info: SearchInfo,
) -> Option<SearchInfo> {
    if let Some(slot) = stats.latest.get_mut(worker_index) {
        *slot = Some(info.clone());
    }
    if info.depth <= stats.emitted_depth {
        return None;
    }
    if !stats.latest.iter().all(|latest| {
        latest
            .as_ref()
            .is_some_and(|latest| latest.depth >= info.depth)
    }) {
        return None;
    }

    let mut aggregate = info;
    aggregate.nodes = stats
        .latest
        .iter()
        .flatten()
        .fold(0_u64, |nodes, info| nodes.saturating_add(info.nodes));
    aggregate.tbhits = stats
        .latest
        .iter()
        .flatten()
        .fold(0_u64, |tbhits, info| tbhits.saturating_add(info.tbhits));
    aggregate.seldepth = stats
        .latest
        .iter()
        .flatten()
        .map(|info| info.seldepth)
        .max()
        .unwrap_or(aggregate.seldepth);
    aggregate.elapsed_ms = stats
        .latest
        .iter()
        .flatten()
        .map(|info| info.elapsed_ms)
        .max()
        .unwrap_or(aggregate.elapsed_ms)
        .max(1);
    aggregate.nps = aggregate.nodes.saturating_mul(1000) / aggregate.elapsed_ms as u64;

    stats.emitted_depth = aggregate.depth;
    Some(aggregate)
}

fn search_with_threads(
    position: Position,
    options: SearchOptions,
    limits: SearchLimits,
    tt: TranspositionTable,
    stop: Arc<AtomicBool>,
    reporter: SearchReporter,
) -> (SearchOutcome, TranspositionTable) {
    let root_moves = root_moves_for_limits(&position, &limits);
    let root_in_tablebase = options
        .syzygy
        .as_ref()
        .is_some_and(|syzygy| syzygy.can_probe(&position, options.syzygy_probe_limit));
    let worker_count = if root_in_tablebase {
        1
    } else {
        effective_thread_count(options.threads, root_moves.len())
    };
    if worker_count <= 1 {
        let tt = tt.into_local();
        let mut searcher = Searcher::new(position, options, limits, tt, stop, Some(reporter));
        let outcome = searcher.search();
        return (outcome, searcher.into_tt());
    }

    let started = Instant::now();
    let node_limit = limits.nodes;
    let mut handles = Vec::with_capacity(worker_count);
    let threaded_reporter = ThreadedReporter::new(worker_count, reporter);
    let tt = tt.into_shared();
    tt.new_search();

    for index in 0..worker_count {
        let position = position.clone();
        let mut options = options.clone();
        options.threads = 1;
        options.reset_tt = false;

        let mut limits = limits.clone();
        limits.search_moves = rotated_root_moves(&root_moves, index);
        if let Some(nodes) = node_limit {
            limits.nodes = Some((nodes / worker_count as u64).max(1));
        }

        let stop = Arc::clone(&stop);
        let reporter = Some(threaded_reporter.reporter_for(index));
        let tt = tt.clone();
        handles.push(thread::spawn(move || {
            let mut searcher = Searcher::new(position, options, limits, tt, stop, reporter);
            let outcome = searcher.search();
            WorkerResult {
                outcome,
                tt: searcher.into_tt(),
            }
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        if let Ok(result) = handle.join() {
            results.push(result);
        }
    }

    reduce_worker_results(results, started.elapsed(), options.hash_mb)
}

fn root_moves_for_limits(position: &Position, limits: &SearchLimits) -> Vec<Move> {
    let mut moves: Vec<Move> = position.legal_moves().into_iter().collect();
    if !limits.search_moves.is_empty() {
        moves = limits
            .search_moves
            .iter()
            .copied()
            .filter(|mv| moves.contains(mv))
            .collect();
    }
    moves
}

fn effective_thread_count(requested: usize, root_move_count: usize) -> usize {
    requested.clamp(1, MAX_THREADS).min(root_move_count.max(1))
}

fn rotated_root_moves(root_moves: &[Move], worker_index: usize) -> Vec<Move> {
    if root_moves.is_empty() {
        return Vec::new();
    }

    let offset = worker_index % root_moves.len();
    root_moves[offset..]
        .iter()
        .chain(root_moves[..offset].iter())
        .copied()
        .collect()
}

fn reduce_worker_results(
    mut results: Vec<WorkerResult>,
    elapsed: Duration,
    hash_mb: usize,
) -> (SearchOutcome, TranspositionTable) {
    if results.is_empty() {
        return (
            SearchOutcome {
                root: Position::startpos(),
                best_move: None,
                score: 0,
                depth: 0,
                nodes: 0,
                tbhits: 0,
                elapsed,
                pv: Vec::new(),
            },
            TranspositionTable::new(hash_mb),
        );
    }

    let total_nodes = results.iter().fold(0_u64, |nodes, result| {
        nodes.saturating_add(result.outcome.nodes)
    });
    let best_index = select_best_worker_index(&results);
    let mut best = results.swap_remove(best_index);
    best.outcome.nodes = total_nodes;
    best.outcome.tbhits = results.iter().fold(best.outcome.tbhits, |tbhits, result| {
        tbhits.saturating_add(result.outcome.tbhits)
    });
    best.outcome.elapsed = elapsed;
    (best.outcome, best.tt)
}

fn select_best_worker_index(results: &[WorkerResult]) -> usize {
    let Some(min_score) = results.iter().map(|result| result.outcome.score).min() else {
        return 0;
    };
    let votes = worker_votes(results, min_score);
    let mut best = 0;
    for index in 1..results.len() {
        if compare_worker_for_selection(&results[index].outcome, &results[best].outcome, &votes)
            .is_gt()
        {
            best = index;
        }
    }
    best
}

fn worker_votes(results: &[WorkerResult], min_score: i32) -> HashMap<Move, i64> {
    let mut votes = HashMap::new();
    for result in results {
        let Some(best_move) = result.outcome.best_move else {
            continue;
        };
        *votes.entry(best_move).or_default() += worker_vote_value(&result.outcome, min_score);
    }
    votes
}

fn worker_vote_value(outcome: &SearchOutcome, min_score: i32) -> i64 {
    let score_delta = i64::from(outcome.score) - i64::from(min_score);
    let depth = i64::from(outcome.depth.max(1));
    (score_delta + 14).max(1) * depth
}

fn compare_worker_for_selection(
    left: &SearchOutcome,
    right: &SearchOutcome,
    votes: &HashMap<Move, i64>,
) -> std::cmp::Ordering {
    worker_vote(left, votes)
        .cmp(&worker_vote(right, votes))
        .then_with(|| compare_worker_outcomes(left, right))
}

fn worker_vote(outcome: &SearchOutcome, votes: &HashMap<Move, i64>) -> i64 {
    outcome
        .best_move
        .and_then(|best_move| votes.get(&best_move).copied())
        .unwrap_or(i64::MIN)
}

fn compare_worker_outcomes(left: &SearchOutcome, right: &SearchOutcome) -> std::cmp::Ordering {
    left.score
        .cmp(&right.score)
        .then_with(|| left.depth.cmp(&right.depth))
        .then_with(|| left.nodes.cmp(&right.nodes))
}

fn is_ponder_go(line: &str) -> bool {
    let mut tokens = line.split_whitespace();
    matches!(tokens.next(), Some(command) if command.eq_ignore_ascii_case("go"))
        && tokens.any(|token| token.eq_ignore_ascii_case("ponder"))
}

fn format_bestmove(outcome: &SearchOutcome) -> String {
    let bestmove = outcome
        .best_move
        .map(|mv| outcome.root.to_uci(mv))
        .unwrap_or_else(|| "0000".to_string());
    let Some(ponder) = outcome
        .best_move
        .and_then(|best| ponder_move(outcome, best))
    else {
        return format!("bestmove {bestmove}");
    };
    format!("bestmove {bestmove} ponder {ponder}")
}

fn ponder_move(outcome: &SearchOutcome, best: Move) -> Option<String> {
    let mut pv = outcome.pv.iter().copied();
    if pv.next()? != best {
        return None;
    }
    let mut position = outcome.root.clone();
    position.play(best).ok()?;
    let ponder = pv.next()?;
    if !position.is_legal(ponder) {
        return None;
    }
    Some(position.to_uci(ponder))
}

fn format_final_info(outcome: &SearchOutcome) -> String {
    let elapsed_ms = outcome.elapsed.as_millis().max(1);
    let nps = outcome.nodes.saturating_mul(1000) / elapsed_ms as u64;
    let mut position = outcome.root.clone();
    let mut pv_uci = Vec::with_capacity(outcome.pv.len());
    for &mv in &outcome.pv {
        if position.is_terminal() || !position.is_legal(mv) {
            break;
        }
        pv_uci.push(position.to_uci(mv));
        position = position.after_move(mv);
    }
    let pv = if pv_uci.is_empty() {
        String::new()
    } else {
        format!(" pv {}", pv_uci.join(" "))
    };
    format!(
        "info depth {} score {} nodes {} nps {} time {} tbhits {}{}",
        outcome.depth,
        format_uci_score(outcome.score),
        outcome.nodes,
        nps,
        elapsed_ms,
        outcome.tbhits,
        pv
    )
}

fn print_final_info_and_bestmove(outcome: &SearchOutcome, bestmove: &str) {
    let final_info = format_final_info(outcome);
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{final_info}");
    let _ = writeln!(stdout, "{bestmove}");
    let _ = stdout.flush();
}

fn apply_time_control(limits: &mut SearchLimits, time_control: UciTimeControl) {
    match time_control {
        UciTimeControl::Infinite | UciTimeControl::Ponder => limits.infinite = true,
        UciTimeControl::MoveTime(duration) => {
            limits.movetime = duration_from_millis(duration.num_milliseconds())
        }
        UciTimeControl::TimeLeft {
            white_time,
            black_time,
            white_increment,
            black_increment,
            moves_to_go,
        } => {
            limits.white_time =
                white_time.and_then(|duration| duration_from_millis(duration.num_milliseconds()));
            limits.black_time =
                black_time.and_then(|duration| duration_from_millis(duration.num_milliseconds()));
            limits.white_increment = white_increment
                .and_then(|duration| duration_from_millis(duration.num_milliseconds()));
            limits.black_increment = black_increment
                .and_then(|duration| duration_from_millis(duration.num_milliseconds()));
            limits.moves_to_go = moves_to_go.map(u32::from);
        }
    }
}

fn apply_default_go_behavior(limits: &mut SearchLimits) {
    if limits.depth.is_none()
        && limits.nodes.is_none()
        && limits.movetime.is_none()
        && !limits.infinite
        && limits.white_time.is_none()
        && limits.black_time.is_none()
    {
        limits.infinite = true;
    }
}

fn duration_from_millis(millis: i64) -> Option<Duration> {
    (millis > 0).then(|| Duration::from_millis(millis as u64))
}

fn parse_uci_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn load_eval_file(value: &str) -> Result<std::sync::Arc<NnueNet>, String> {
    if value.trim().eq_ignore_ascii_case(INTERNAL_EVAL_FILE) {
        NnueNet::embedded().map_err(|err| err.to_string())
    } else {
        NnueNet::from_file(value).map_err(|err| err.to_string())
    }
}

fn load_syzygy_path(value: &str) -> Option<Arc<SyzygyTablebase>> {
    if value.trim().is_empty() || value.trim().eq_ignore_ascii_case(EMPTY_SYZYGY_PATH) {
        return None;
    }
    SyzygyTablebase::load(value).ok()
}

fn print_info(info: SearchInfo) {
    let pv = if info.pv.is_empty() {
        String::new()
    } else {
        format!(" pv {}", info.pv.join(" "))
    };
    println!(
        "info depth {} seldepth {} score {} nodes {} nps {} time {} hashfull {} tbhits {}{}",
        info.depth,
        info.seldepth,
        format_uci_score(info.score),
        info.nodes,
        info.nps,
        info.elapsed_ms,
        info.hashfull,
        info.tbhits,
        pv
    );
    flush_stdout();
}

fn flush_stdout() {
    let _ = io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Color;

    #[test]
    fn parses_position_with_moves() {
        let mut engine = UciEngine::new();
        engine.handle_line("position startpos moves e2e4 e7e5");
        assert_eq!(engine.position.side_to_move(), Color::White);
    }

    #[test]
    fn hash_option_resizes_table() {
        let mut engine = UciEngine::new();
        engine.handle_line("setoption name Hash value 4");
        assert_eq!(engine.options.hash_mb, 4);
    }

    #[test]
    fn ponder_option_is_parsed() {
        let mut engine = UciEngine::new();
        engine.handle_line("setoption name Ponder value true");
        assert!(engine.ponder_enabled);
        engine.handle_line("setoption name Ponder value false");
        assert!(!engine.ponder_enabled);
    }

    #[test]
    fn threads_option_is_parsed_and_clamped() {
        let mut engine = UciEngine::new();
        engine.handle_line("setoption name Threads value 4");
        assert_eq!(engine.options.threads, 4);
        engine.handle_line("setoption name Threads value 0");
        assert_eq!(engine.options.threads, 1);
        engine.handle_line("setoption name Threads value 9999");
        assert_eq!(engine.options.threads, MAX_THREADS);
    }

    #[test]
    fn nnue_options_are_parsed() {
        let mut engine = UciEngine::new();
        engine.handle_line("setoption name Use NNUE value false");
        assert!(!engine.options.use_nnue);
        engine.handle_line("setoption name EvalFile value <internal>");
        assert!(engine.options.nnue.is_some());
    }

    #[test]
    fn invalid_eval_file_keeps_previous_net() {
        let mut engine = UciEngine::new();
        let checksum = engine.options.nnue.as_ref().map(|net| net.checksum());
        engine.handle_line("setoption name EvalFile value missing-file.nnue");
        assert_eq!(
            engine.options.nnue.as_ref().map(|net| net.checksum()),
            checksum
        );
    }

    #[test]
    fn syzygy_options_are_parsed() {
        let mut engine = UciEngine::new();

        engine.handle_line("setoption name SyzygyPath value <empty>");
        engine.handle_line("setoption name SyzygyProbeDepth value 4");
        engine.handle_line("setoption name Syzygy50MoveRule value false");
        engine.handle_line("setoption name SyzygyProbeLimit value 5");

        assert_eq!(engine.options.syzygy_path, "<empty>");
        assert!(engine.options.syzygy.is_none());
        assert_eq!(engine.options.syzygy_probe_depth, 4);
        assert!(!engine.options.syzygy_50_move_rule);
        assert_eq!(engine.options.syzygy_probe_limit, 5);

        engine.handle_line("setoption name SyzygyProbeDepth value 999");
        engine.handle_line("setoption name SyzygyProbeLimit value 99");

        assert_eq!(engine.options.syzygy_probe_depth, 100);
        assert_eq!(engine.options.syzygy_probe_limit, 7);
    }

    #[test]
    fn bare_go_defaults_to_infinite() {
        let mut limits = SearchLimits::default();
        apply_default_go_behavior(&mut limits);
        assert!(limits.infinite);
        assert!(limits.depth.is_none());
    }

    #[test]
    fn timed_go_keeps_clock_limits() {
        let mut limits = SearchLimits {
            white_time: Some(Duration::from_millis(1000)),
            black_time: Some(Duration::from_millis(1000)),
            ..SearchLimits::default()
        };
        apply_default_go_behavior(&mut limits);
        assert!(!limits.infinite);
    }

    #[test]
    fn root_moves_are_rotated_for_helper_workers() {
        let position = Position::startpos();
        let root_moves = root_moves_for_limits(&position, &SearchLimits::default());
        let rotated = rotated_root_moves(&root_moves, 3);

        assert_eq!(rotated.len(), root_moves.len());
        assert_eq!(rotated[0], root_moves[3]);
        assert_eq!(rotated[root_moves.len() - 3], root_moves[0]);
    }

    #[test]
    fn worker_reduction_sums_nodes_and_keeps_best_score() {
        let root = Position::startpos();
        let e2e4 = root.uci_to_move("e2e4").unwrap();
        let d2d4 = root.uci_to_move("d2d4").unwrap();
        let results = vec![
            WorkerResult {
                outcome: SearchOutcome {
                    root: root.clone(),
                    best_move: Some(e2e4),
                    score: 10,
                    depth: 5,
                    nodes: 100,
                    tbhits: 1,
                    elapsed: Duration::from_millis(5),
                    pv: vec![e2e4],
                },
                tt: TranspositionTable::new(1),
            },
            WorkerResult {
                outcome: SearchOutcome {
                    root: root.clone(),
                    best_move: Some(d2d4),
                    score: 30,
                    depth: 4,
                    nodes: 200,
                    tbhits: 2,
                    elapsed: Duration::from_millis(5),
                    pv: vec![d2d4],
                },
                tt: TranspositionTable::new(1),
            },
        ];

        let (outcome, _) = reduce_worker_results(results, Duration::from_millis(10), 4);
        assert_eq!(outcome.best_move, Some(d2d4));
        assert_eq!(outcome.nodes, 300);
        assert_eq!(outcome.tbhits, 3);
        assert_eq!(outcome.elapsed, Duration::from_millis(10));
    }

    #[test]
    fn worker_reduction_uses_thread_voting() {
        let root = Position::startpos();
        let e2e4 = root.uci_to_move("e2e4").unwrap();
        let d2d4 = root.uci_to_move("d2d4").unwrap();
        let results = vec![
            WorkerResult {
                outcome: SearchOutcome {
                    root: root.clone(),
                    best_move: Some(e2e4),
                    score: 20,
                    depth: 5,
                    nodes: 100,
                    tbhits: 1,
                    elapsed: Duration::from_millis(5),
                    pv: vec![e2e4],
                },
                tt: TranspositionTable::new(1),
            },
            WorkerResult {
                outcome: SearchOutcome {
                    root: root.clone(),
                    best_move: Some(e2e4),
                    score: 25,
                    depth: 4,
                    nodes: 200,
                    tbhits: 2,
                    elapsed: Duration::from_millis(5),
                    pv: vec![e2e4],
                },
                tt: TranspositionTable::new(1),
            },
            WorkerResult {
                outcome: SearchOutcome {
                    root,
                    best_move: Some(d2d4),
                    score: 50,
                    depth: 2,
                    nodes: 300,
                    tbhits: 4,
                    elapsed: Duration::from_millis(5),
                    pv: vec![d2d4],
                },
                tt: TranspositionTable::new(1),
            },
        ];

        let (outcome, _) = reduce_worker_results(results, Duration::from_millis(10), 4);
        assert_eq!(outcome.best_move, Some(e2e4));
        assert_eq!(outcome.score, 25);
        assert_eq!(outcome.nodes, 600);
        assert_eq!(outcome.tbhits, 7);
    }

    #[test]
    fn threaded_reporter_aggregates_nodes_and_nps() {
        let root = Position::startpos();
        let e2e4 = root.uci_to_move("e2e4").unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_for_reporter = Arc::clone(&seen);
        let reporter = ThreadedReporter::new(
            2,
            Box::new(move |info| {
                seen_for_reporter
                    .lock()
                    .expect("seen lock poisoned")
                    .push(info);
            }),
        );
        let worker_0 = reporter.reporter_for(0);
        let worker_1 = reporter.reporter_for(1);

        worker_0(SearchInfo {
            depth: 3,
            seldepth: 5,
            score: 12,
            nodes: 100,
            nps: 1,
            elapsed_ms: 10,
            hashfull: 0,
            tbhits: 1,
            pv: vec![root.to_uci(e2e4)],
        });
        assert!(seen.lock().expect("seen lock poisoned").is_empty());
        worker_1(SearchInfo {
            depth: 3,
            seldepth: 7,
            score: 20,
            nodes: 250,
            nps: 1,
            elapsed_ms: 10,
            hashfull: 0,
            tbhits: 2,
            pv: vec![root.to_uci(e2e4)],
        });

        let seen = seen.lock().expect("seen lock poisoned");
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].nodes, 350);
        assert_eq!(seen[0].nps, 35_000);
        assert_eq!(seen[0].seldepth, 7);
        assert_eq!(seen[0].tbhits, 3);
    }

    #[test]
    fn bestmove_line_includes_ponder_from_pv() {
        let root = Position::startpos();
        let best = root.uci_to_move("e2e4").unwrap();
        let after_best = root.after_move(best);
        let ponder = after_best.uci_to_move("e7e5").unwrap();
        let outcome = SearchOutcome {
            root,
            best_move: Some(best),
            score: 0,
            depth: 2,
            nodes: 2,
            tbhits: 0,
            elapsed: Duration::from_millis(1),
            pv: vec![best, ponder],
        };

        assert_eq!(format_bestmove(&outcome), "bestmove e2e4 ponder e7e5");
    }

    #[test]
    fn final_info_line_is_derived_from_outcome() {
        let root = Position::startpos();
        let best = root.uci_to_move("e2e4").unwrap();
        let after_best = root.after_move(best);
        let ponder = after_best.uci_to_move("e7e5").unwrap();
        let outcome = SearchOutcome {
            root,
            best_move: Some(best),
            score: 23,
            depth: 2,
            nodes: 42,
            tbhits: 3,
            elapsed: Duration::from_millis(2),
            pv: vec![best, ponder],
        };

        assert_eq!(
            format_final_info(&outcome),
            "info depth 2 score cp 23 nodes 42 nps 21000 time 2 tbhits 3 pv e2e4 e7e5"
        );
    }

    #[test]
    fn bestmove_line_omits_ponder_when_pv_does_not_match_bestmove() {
        let root = Position::startpos();
        let best = root.uci_to_move("e2e4").unwrap();
        let other = root.uci_to_move("d2d4").unwrap();
        let outcome = SearchOutcome {
            root,
            best_move: Some(best),
            score: 0,
            depth: 2,
            nodes: 2,
            tbhits: 0,
            elapsed: Duration::from_millis(1),
            pv: vec![other],
        };

        assert_eq!(format_bestmove(&outcome), "bestmove e2e4");
    }

    #[test]
    fn detects_go_ponder_with_clock_args() {
        assert!(is_ponder_go("go ponder wtime 1000 btime 1000"));
        assert!(is_ponder_go("GO WTIME 1000 PONDER BTIME 1000"));
        assert!(!is_ponder_go("go wtime 1000 btime 1000"));
        assert!(!is_ponder_go("position startpos moves e2e4"));
    }

    #[test]
    fn ponder_hit_without_search_is_noop() {
        let mut engine = UciEngine::new();
        engine.ponder_hit();
    }
}
