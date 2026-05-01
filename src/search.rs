use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use shakmaty::{Color, KnownOutcome, Move, Role as Piece, Square};

use crate::eval::piece_value;
use crate::nnue::{INTERNAL_EVAL_FILE, NnueNet, NnuePosition};
use crate::position::Position;
use crate::see::static_exchange_eval;
use crate::syzygy::{
    DEFAULT_SYZYGY_PATH, DEFAULT_SYZYGY_PROBE_DEPTH, DEFAULT_SYZYGY_PROBE_LIMIT, SyzygyTablebase,
};
use crate::time::TimeControl;
use crate::tt::{Bound, Entry, TranspositionTable};

const INF: i32 = 32_000;
const MATE: i32 = 30_000;
const MATE_THRESHOLD: i32 = 29_000;
const MAX_PLY: usize = 128;
const MAX_CHECK_EXTENSION_PLY: usize = 8;
const MAX_QS_PLY: usize = 32;
const COUNTER_MOVE_SCORE: i32 = 785_000;
const QUIET_BASE_SCORE: i32 = 400_000;
const BAD_CAPTURE_BASE_SCORE: i32 = 100_000;
const MAX_HISTORY_SCORE: i32 = 32_000;
const TIME_CHECK_INTERVAL_NODES: u32 = 512;
const NODE_LIMIT_CHECK_DIVISOR: u64 = 1024;
const REVERSE_FUTILITY_MAX_DEPTH: i32 = 3;
const FUTILITY_MAX_DEPTH: i32 = 3;
const LATE_MOVE_PRUNING_MAX_DEPTH: i32 = 3;
const PROBCUT_MIN_DEPTH: i32 = 5;
const PROBCUT_REDUCTION: i32 = 3;
const PROBCUT_MARGIN: i32 = 180;
const PROBCUT_MAX_MOVES: usize = 8;
const MAX_SOFT_EXTENSIONS: u32 = 2;
const TB_WIN_SCORE: i32 = MATE_THRESHOLD - MAX_PLY as i32 - 1;
const COLOR_COUNT: usize = 2;
const PIECE_COUNT: usize = 6;

pub type SearchReporter = Box<dyn Fn(SearchInfo) + Send + 'static>;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub hash_mb: usize,
    pub move_overhead: Duration,
    pub threads: usize,
    pub reset_tt: bool,
    pub use_nnue: bool,
    pub eval_file: Option<String>,
    pub nnue: Option<Arc<NnueNet>>,
    pub syzygy_path: String,
    pub syzygy: Option<Arc<SyzygyTablebase>>,
    pub syzygy_probe_depth: u32,
    pub syzygy_50_move_rule: bool,
    pub syzygy_probe_limit: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            hash_mb: 32,
            move_overhead: Duration::from_millis(10),
            threads: 1,
            reset_tt: true,
            use_nnue: true,
            eval_file: Some(INTERNAL_EVAL_FILE.to_string()),
            nnue: NnueNet::embedded().ok(),
            syzygy_path: DEFAULT_SYZYGY_PATH.to_string(),
            syzygy: None,
            syzygy_probe_depth: DEFAULT_SYZYGY_PROBE_DEPTH,
            syzygy_50_move_rule: true,
            syzygy_probe_limit: DEFAULT_SYZYGY_PROBE_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub infinite: bool,
    pub ponder_hit: Option<Arc<AtomicBool>>,
    pub white_time: Option<Duration>,
    pub black_time: Option<Duration>,
    pub white_increment: Option<Duration>,
    pub black_increment: Option<Duration>,
    pub moves_to_go: Option<u32>,
    pub search_moves: Vec<Move>,
}

impl SearchLimits {
    fn time_control(&self, move_overhead: Duration, ply: u32) -> TimeControl {
        TimeControl {
            depth: self.depth,
            nodes: self.nodes,
            movetime: self.movetime,
            infinite: self.infinite,
            white_time: self.white_time,
            black_time: self.black_time,
            white_increment: self.white_increment,
            black_increment: self.black_increment,
            moves_to_go: self.moves_to_go,
            move_overhead,
            ply,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchInfo {
    pub depth: u32,
    pub seldepth: u32,
    pub score: i32,
    pub nodes: u64,
    pub nps: u64,
    pub elapsed_ms: u128,
    pub hashfull: u64,
    pub tbhits: u64,
    pub pv: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub root: Position,
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
    pub tbhits: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
}

#[derive(Debug, Clone, Copy)]
struct NodeContext {
    allow_null: bool,
    previous_move: Option<Move>,
}

#[derive(Debug, Clone, Copy)]
struct ScoredMove {
    mv: Move,
    score: i32,
}

impl NodeContext {
    fn after_move(mv: Move) -> Self {
        Self {
            allow_null: true,
            previous_move: Some(mv),
        }
    }

    fn after_null_move() -> Self {
        Self {
            allow_null: false,
            previous_move: None,
        }
    }
}

pub struct Searcher {
    root: NnuePosition,
    options: SearchOptions,
    limits: SearchLimits,
    tt: TranspositionTable,
    stop: Arc<AtomicBool>,
    reporter: Option<SearchReporter>,
    start: Instant,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    ponder_clock_started: bool,
    time_check_countdown: u32,
    time_check_interval: u32,
    nodes: u64,
    tbhits: u64,
    seldepth: usize,
    aborted: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    counter_moves: [[Option<Move>; 64]; 64],
    quiet_history: [[[i32; 64]; PIECE_COUNT]; COLOR_COUNT],
    capture_history: [[[i32; 64]; PIECE_COUNT]; PIECE_COUNT],
}

impl Searcher {
    pub fn new(
        root: Position,
        options: SearchOptions,
        limits: SearchLimits,
        tt: TranspositionTable,
        stop: Arc<AtomicBool>,
        reporter: Option<SearchReporter>,
    ) -> Self {
        let net = options
            .use_nnue
            .then(|| options.nnue.clone())
            .flatten()
            .filter(|net| !net.is_bootstrap());
        Self {
            root: NnuePosition::new(root, net),
            options,
            limits,
            tt,
            stop,
            reporter,
            start: Instant::now(),
            soft_deadline: None,
            hard_deadline: None,
            ponder_clock_started: false,
            time_check_countdown: 0,
            time_check_interval: TIME_CHECK_INTERVAL_NODES,
            nodes: 0,
            tbhits: 0,
            seldepth: 0,
            aborted: false,
            killers: [[None; 2]; MAX_PLY],
            counter_moves: [[None; 64]; 64],
            quiet_history: [[[0; 64]; PIECE_COUNT]; COLOR_COUNT],
            capture_history: [[[0; 64]; PIECE_COUNT]; PIECE_COUNT],
        }
    }

    pub fn search(&mut self) -> SearchOutcome {
        self.start = Instant::now();
        self.ponder_clock_started = self.limits.ponder_hit.is_none();
        if self.ponder_clock_started {
            let deadlines = self
                .time_control()
                .deadlines(self.root.side_to_move(), self.start);
            self.soft_deadline = deadlines.soft;
            self.hard_deadline = deadlines.hard;
        } else {
            self.soft_deadline = None;
            self.hard_deadline = None;
        }
        self.nodes = 0;
        self.tbhits = 0;
        self.time_check_countdown = 0;
        self.time_check_interval = time_check_interval(&self.limits);
        self.seldepth = 0;
        self.aborted = false;
        if self.options.reset_tt {
            self.tt.new_search();
        }

        let root_moves = self.root_moves();
        if root_moves.is_empty() {
            let score = terminal_score(&self.root, 0).unwrap_or(0);
            return SearchOutcome {
                root: self.root.position().clone(),
                best_move: None,
                score,
                depth: 0,
                nodes: self.nodes,
                tbhits: self.tbhits,
                elapsed: self.start.elapsed(),
                pv: Vec::new(),
            };
        }

        if let Some(outcome) = self.syzygy_root_outcome(&root_moves) {
            return outcome;
        }

        let time_control = self.time_control();
        let max_depth = if self.limits.ponder_hit.is_some() && self.limits.depth.is_none() {
            128
        } else {
            time_control.max_depth()
        };
        let mut best_move = root_moves.first().copied();
        let mut best_score = -INF;
        let mut completed_depth = 0;
        let mut completed_pv = Vec::new();
        let mut instability = 0_u32;
        let mut soft_extensions = 0_u32;

        for depth in 1..=max_depth {
            self.maybe_start_ponder_clock();
            if completed_depth > 0
                && self.soft_time_expired()
                && !self.can_extend_after_soft(instability, soft_extensions)
            {
                break;
            }
            if self.should_stop() && completed_depth > 0 {
                break;
            }
            self.aborted = false;
            self.seldepth = 0;

            let previous_best_move = best_move;
            let previous_score = best_score;
            let mut aspiration_failed_low = false;
            let mut aspiration_failed_high = false;
            let (score, candidate) = if depth >= 4 && best_score.abs() < MATE_THRESHOLD {
                let mut window = 50;
                loop {
                    let alpha = best_score - window;
                    let beta = best_score + window;
                    let result = self.root_search(depth as i32, alpha, beta, &root_moves);
                    if self.aborted || result.0 <= alpha || result.0 >= beta {
                        aspiration_failed_low |= result.0 <= alpha;
                        aspiration_failed_high |= result.0 >= beta;
                        if window >= 800 || self.aborted {
                            break self.root_search(depth as i32, -INF, INF, &root_moves);
                        }
                        window *= 2;
                    } else {
                        break result;
                    }
                }
            } else {
                self.root_search(depth as i32, -INF, INF, &root_moves)
            };

            if self.aborted {
                if completed_depth == 0 {
                    best_score = self.root.evaluate();
                }
                break;
            }

            best_score = score;
            best_move = candidate.or(best_move);
            if completed_depth > 0 {
                let best_changed = candidate.is_some() && candidate != previous_best_move;
                let score_drop = score + 80 < previous_score;
                let score_swing = (score - previous_score).abs() >= 160;
                instability = update_instability(
                    instability,
                    best_changed,
                    score_drop || aspiration_failed_low,
                    score_swing || aspiration_failed_high,
                );
            }
            completed_depth = depth;
            completed_pv = self.extract_pv(depth as usize);
            if let Some(best) = best_move {
                completed_pv = ensure_pv_starts_with(best, completed_pv);
            }
            self.report(completed_depth, best_score, &completed_pv);

            if best_score.abs() >= MATE_THRESHOLD && !self.pondering_before_hit() {
                break;
            }
            if self.soft_time_expired() {
                if self.can_extend_after_soft(instability, soft_extensions) {
                    soft_extensions += 1;
                } else {
                    break;
                }
            }
        }

        SearchOutcome {
            root: self.root.position().clone(),
            best_move,
            score: best_score,
            depth: completed_depth,
            nodes: self.nodes,
            tbhits: self.tbhits,
            elapsed: self.start.elapsed(),
            pv: completed_pv,
        }
    }

    pub fn into_tt(self) -> TranspositionTable {
        self.tt
    }

    fn root_moves(&self) -> Vec<Move> {
        let mut moves = self.root.legal_moves();
        if !self.limits.search_moves.is_empty() {
            moves = self
                .limits
                .search_moves
                .iter()
                .copied()
                .filter(|mv| moves.contains(mv))
                .collect();
        }
        moves
    }

    fn root_search(
        &mut self,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        root_moves: &[Move],
    ) -> (i32, Option<Move>) {
        let alpha_original = alpha;
        let moves = root_moves.to_vec();
        let tt_move = self
            .tt
            .probe(self.root.hash())
            .and_then(|entry| entry.best_move);
        let mut moves = self.score_moves(&self.root.clone(), moves, tt_move, 0, None);

        let mut best_score = -INF;
        let mut best_move = None;
        for searched in 0..moves.len() {
            let mv = pick_next(&mut moves, searched);
            if self.should_stop() {
                self.aborted = true;
                break;
            }

            let child = self.root.after_move(mv);
            let mut child_depth = depth - 1;
            if !child.checkers().is_empty() {
                child_depth += 1;
            }
            let score = if searched == 0 {
                -self.negamax(
                    &child,
                    child_depth,
                    -beta,
                    -alpha,
                    1,
                    NodeContext::after_move(mv),
                )
            } else {
                let score = -self.negamax(
                    &child,
                    child_depth,
                    -alpha - 1,
                    -alpha,
                    1,
                    NodeContext::after_move(mv),
                );
                if !self.aborted && score > alpha && score < beta {
                    -self.negamax(
                        &child,
                        child_depth,
                        -beta,
                        -alpha,
                        1,
                        NodeContext::after_move(mv),
                    )
                } else {
                    score
                }
            };
            if self.aborted {
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
        }

        if !self.aborted
            && let Some(best_move) = best_move
        {
            let bound = if best_score <= alpha_original {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt.store(Entry {
                key: self.root.hash(),
                depth: depth as i16,
                score: score_to_tt(best_score, 0),
                bound,
                best_move: Some(best_move),
            });
        }

        (best_score, best_move)
    }

    fn negamax(
        &mut self,
        position: &NnuePosition,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        context: NodeContext,
    ) -> i32 {
        if ply >= MAX_PLY - 1 {
            return position.evaluate();
        }
        self.seldepth = self.seldepth.max(ply);
        self.nodes = self.nodes.saturating_add(1);
        if self.should_stop() {
            self.aborted = true;
            return position.evaluate();
        }

        if let Some(score) = terminal_score(position, ply) {
            return score;
        }
        if position.is_draw() {
            return 0;
        }
        if let Some(score) = self.probe_syzygy_node(position, depth, ply) {
            return score;
        }
        if depth <= 0 {
            return self.quiescence(position, alpha, beta, ply);
        }

        let in_check = !position.checkers().is_empty();
        if in_check && ply < MAX_CHECK_EXTENSION_PLY {
            depth += 1;
        }

        let alpha_original = alpha;
        let is_pv_node = beta > alpha + 1;
        let key = position.hash();
        let tt_entry = self.tt.probe(key);
        if let Some(entry) = tt_entry
            && entry.depth >= depth as i16
        {
            let score = score_from_tt(entry.score, ply);
            match entry.bound {
                Bound::Exact => return score,
                Bound::Lower if score >= beta => return score,
                Bound::Upper if score <= alpha => return score,
                _ => {}
            }
        }

        let static_eval = position.evaluate();
        let near_mate_window = alpha.abs() >= MATE_THRESHOLD || beta.abs() >= MATE_THRESHOLD;
        if !in_check
            && !is_pv_node
            && context.allow_null
            && depth <= REVERSE_FUTILITY_MAX_DEPTH
            && !near_mate_window
            && has_non_pawn_material(position)
            && static_eval - reverse_futility_margin(depth) >= beta
        {
            return static_eval - reverse_futility_margin(depth);
        }

        if !in_check
            && !is_pv_node
            && context.allow_null
            && depth >= 4
            && static_eval >= beta
            && !near_mate_window
            && has_non_pawn_material(position)
            && let Some(null_position) = position.null_move()
        {
            let reduction = 2 + depth / 6;
            let null_depth = depth - 1 - reduction;
            if null_depth < 1 {
                return static_eval;
            }
            let score = -self.negamax(
                &null_position,
                null_depth,
                -beta,
                -beta + 1,
                ply + 1,
                NodeContext::after_null_move(),
            );
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return score;
            }
        }

        if !in_check
            && !is_pv_node
            && context.allow_null
            && !near_mate_window
            && let Some(score) = self.probcut(
                position,
                depth,
                beta,
                ply,
                tt_entry.and_then(|entry| entry.best_move),
                context.previous_move,
            )
        {
            return score;
        }

        let moves = position.legal_moves();
        if moves.is_empty() {
            return terminal_score(position, ply).unwrap_or(0);
        }
        let mut moves = self.score_moves(
            position,
            moves,
            tt_entry.and_then(|entry| entry.best_move),
            ply,
            context.previous_move,
        );

        let mut best_score = -INF;
        let mut best_move = None;
        let mut searched = 0;
        let mut searched_quiets = Vec::with_capacity(16);

        for index in 0..moves.len() {
            let mv = pick_next(&mut moves, index);
            if self.should_stop() {
                self.aborted = true;
                break;
            }

            let quiet = position.is_quiet(mv);
            let mut child = None;
            if !in_check
                && !is_pv_node
                && quiet
                && depth <= FUTILITY_MAX_DEPTH
                && searched > 0
                && !near_mate_window
                && static_eval + futility_margin(depth) <= alpha
            {
                let next = position.after_move(mv);
                if next.checkers().is_empty() {
                    continue;
                }
                child = Some(next);
            }
            if !in_check
                && !is_pv_node
                && quiet
                && depth <= LATE_MOVE_PRUNING_MAX_DEPTH
                && searched >= late_move_pruning_threshold(depth)
                && !near_mate_window
            {
                let next = child.get_or_insert_with(|| position.after_move(mv));
                if next.checkers().is_empty() {
                    continue;
                }
            }
            if !in_check
                && !is_pv_node
                && depth <= 3
                && position.is_capture(mv)
                && mv.promotion().is_none()
                && searched > 0
                && !near_mate_window
                && static_exchange_eval(position.position(), mv) < -120 * depth
            {
                continue;
            }

            let child = child.unwrap_or_else(|| position.after_move(mv));
            let mut child_depth = depth - 1;
            if !child.checkers().is_empty() {
                child_depth += 1;
            }

            let mut score;
            if index >= 4 && depth >= 3 && quiet && !in_check {
                let reduction =
                    late_move_reduction(depth, index, self.history_score(position, mv), is_pv_node);
                score = -self.negamax(
                    &child,
                    (child_depth - reduction).max(0),
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    NodeContext::after_move(mv),
                );
                if score > alpha && !self.aborted {
                    score = -self.negamax(
                        &child,
                        child_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        NodeContext::after_move(mv),
                    );
                }
            } else if searched > 0 {
                score = -self.negamax(
                    &child,
                    child_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    NodeContext::after_move(mv),
                );
                if score > alpha && score < beta && !self.aborted {
                    score = -self.negamax(
                        &child,
                        child_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        NodeContext::after_move(mv),
                    );
                }
            } else {
                score = -self.negamax(
                    &child,
                    child_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    NodeContext::after_move(mv),
                );
            }

            if self.aborted {
                return 0;
            }

            searched += 1;
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    if quiet {
                        self.record_quiet_cutoff(
                            position,
                            mv,
                            depth,
                            ply,
                            context.previous_move,
                            &searched_quiets,
                        );
                    } else if position.is_capture(mv) {
                        self.record_capture_cutoff(position, mv, depth);
                    }
                    break;
                }
            }
            if quiet && !in_check {
                searched_quiets.push(mv);
            }
        }

        if searched == 0 {
            return self.quiescence(position, alpha_original, beta, ply);
        }

        let bound = if best_score <= alpha_original {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt.store(Entry {
            key,
            depth: depth as i16,
            score: score_to_tt(best_score, ply),
            bound,
            best_move,
        });

        best_score
    }

    fn probcut(
        &mut self,
        position: &NnuePosition,
        depth: i32,
        beta: i32,
        ply: usize,
        tt_move: Option<Move>,
        previous_move: Option<Move>,
    ) -> Option<i32> {
        if depth < PROBCUT_MIN_DEPTH || beta.abs() >= MATE_THRESHOLD - PROBCUT_MARGIN {
            return None;
        }

        let threshold = beta + PROBCUT_MARGIN;
        let moves: Vec<Move> = position
            .legal_moves()
            .into_iter()
            .filter(|&mv| {
                position.is_tactical(mv) && static_exchange_eval(position.position(), mv) >= 0
            })
            .collect();
        if moves.is_empty() {
            return None;
        }
        let mut moves = self.score_moves(position, moves, tt_move, ply, previous_move);

        let reduced_depth = (depth - PROBCUT_REDUCTION - 1).max(0);
        for index in 0..moves.len().min(PROBCUT_MAX_MOVES) {
            let mv = pick_next(&mut moves, index);
            if self.should_stop() {
                self.aborted = true;
                return Some(0);
            }

            let child = position.after_move(mv);
            let score = -self.negamax(
                &child,
                reduced_depth,
                -threshold,
                -threshold + 1,
                ply + 1,
                NodeContext::after_move(mv),
            );
            if self.aborted {
                return Some(0);
            }
            if score >= threshold {
                return Some(score - PROBCUT_MARGIN);
            }
        }

        None
    }

    fn quiescence(
        &mut self,
        position: &NnuePosition,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        if ply >= MAX_PLY - 1 {
            return position.evaluate();
        }
        self.seldepth = self.seldepth.max(ply);
        self.nodes = self.nodes.saturating_add(1);
        if self.should_stop() {
            self.aborted = true;
            return position.evaluate();
        }

        if let Some(score) = terminal_score(position, ply) {
            return score;
        }
        if position.is_draw() {
            return 0;
        }
        if let Some(score) = self.probe_syzygy_node(position, 0, ply) {
            return score;
        }
        if ply >= MAX_QS_PLY {
            return position.evaluate();
        }

        let in_check = !position.checkers().is_empty();
        let stand_pat = position.evaluate();
        if !in_check {
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves = position.legal_moves();
        if !in_check {
            moves.retain(|mv| position.is_tactical(*mv));
        }
        let mut moves = self.score_moves(position, moves, None, ply, None);

        for index in 0..moves.len() {
            let mv = pick_next(&mut moves, index);
            if !in_check {
                let see = static_exchange_eval(position.position(), mv);
                if see < -90 || stand_pat + see + 120 <= alpha {
                    continue;
                }
            }

            let child = position.after_move(mv);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    fn score_moves(
        &self,
        position: &NnuePosition,
        moves: Vec<Move>,
        tt_move: Option<Move>,
        ply: usize,
        previous_move: Option<Move>,
    ) -> Vec<ScoredMove> {
        moves
            .into_iter()
            .map(|mv| ScoredMove {
                mv,
                score: self.move_score(position, mv, tt_move, ply, previous_move),
            })
            .collect()
    }

    fn move_score(
        &self,
        position: &NnuePosition,
        mv: Move,
        tt_move: Option<Move>,
        ply: usize,
        previous_move: Option<Move>,
    ) -> i32 {
        if Some(mv) == tt_move {
            return 2_000_000;
        }
        let moved = position.moved_piece(mv).unwrap_or(Piece::Pawn);

        if position.is_capture(mv) {
            let victim = position.captured_piece(mv).map(piece_value).unwrap_or(0);
            let victim_piece = position.captured_piece(mv).unwrap_or(Piece::Pawn);
            let attacker = piece_value(moved);
            let see = static_exchange_eval(position.position(), mv);
            let history =
                self.capture_history[piece_index(moved)][piece_index(victim_piece)][mv.to().to_usize()];
            if see >= 0 {
                return 1_100_000 + see * 32 + victim * 16 - attacker + history;
            }
            return BAD_CAPTURE_BASE_SCORE + see * 32 + victim * 16 - attacker + history;
        }

        if let Some(promotion) = mv.promotion() {
            let see = static_exchange_eval(position.position(), mv);
            return 1_000_000 + see * 16 + piece_value(promotion);
        }

        if ply < MAX_PLY {
            if self.killers[ply][0] == Some(mv) {
                return 800_000;
            }
            if self.killers[ply][1] == Some(mv) {
                return 790_000;
            }
        }
        if let Some(previous_move) = previous_move
            && self.counter_moves[move_from(previous_move).to_usize()][previous_move.to().to_usize()]
                == Some(mv)
        {
            return COUNTER_MOVE_SCORE;
        }

        QUIET_BASE_SCORE + self.history_score(position, mv)
    }

    fn history_score(&self, position: &NnuePosition, mv: Move) -> i32 {
        let side = position.side_to_move();
        let moved = position.moved_piece(mv).unwrap_or(Piece::Pawn);
        self.quiet_history[color_index(side)][piece_index(moved)][mv.to().to_usize()]
    }

    fn record_quiet_cutoff(
        &mut self,
        position: &NnuePosition,
        mv: Move,
        depth: i32,
        ply: usize,
        previous_move: Option<Move>,
        searched_quiets: &[Move],
    ) {
        if ply < MAX_PLY && self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
        if let Some(previous_move) = previous_move {
            self.counter_moves[move_from(previous_move).to_usize()][previous_move.to().to_usize()] = Some(mv);
        }
        let bonus = history_bonus(depth);
        self.update_quiet_history(position, mv, bonus);
        for &quiet in searched_quiets {
            if quiet != mv {
                self.update_quiet_history(position, quiet, -bonus / 2);
            }
        }
    }

    fn record_capture_cutoff(&mut self, position: &NnuePosition, mv: Move, depth: i32) {
        let Some(moved) = position.moved_piece(mv) else {
            return;
        };
        let Some(captured) = position.captured_piece(mv) else {
            return;
        };
        let entry =
            &mut self.capture_history[piece_index(moved)][piece_index(captured)][mv.to().to_usize()];
        update_history_stat(entry, history_bonus(depth) / 2);
    }

    fn update_quiet_history(&mut self, position: &NnuePosition, mv: Move, bonus: i32) {
        let side = position.side_to_move();
        let Some(moved) = position.moved_piece(mv) else {
            return;
        };
        let entry = &mut self.quiet_history[color_index(side)][piece_index(moved)][mv.to().to_usize()];
        update_history_stat(entry, bonus);
    }

    fn should_stop(&mut self) -> bool {
        self.maybe_start_ponder_clock();
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(nodes) = self.limits.nodes
            && self.nodes >= nodes
        {
            return true;
        }
        if let Some(deadline) = self.hard_deadline {
            return self.hard_time_expired(deadline);
        }
        false
    }

    fn hard_time_expired(&mut self, deadline: Instant) -> bool {
        if self.time_check_countdown > 0 {
            self.time_check_countdown -= 1;
            if self.time_check_countdown > 0 {
                return false;
            }
        }

        self.time_check_countdown = self.time_check_interval;
        Instant::now() >= deadline
    }

    fn soft_time_expired(&mut self) -> bool {
        self.maybe_start_ponder_clock();
        self.soft_deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn maybe_start_ponder_clock(&mut self) {
        if self.ponder_clock_started {
            return;
        }
        let Some(ponder_hit) = &self.limits.ponder_hit else {
            self.ponder_clock_started = true;
            return;
        };
        if !ponder_hit.load(Ordering::Relaxed) {
            return;
        }

        self.ponder_clock_started = true;
        let clock_start = Instant::now();
        let deadlines = self
            .time_control()
            .deadlines(self.root.side_to_move(), clock_start);
        self.soft_deadline = deadlines.soft;
        self.hard_deadline = deadlines.hard;
    }

    fn time_control(&self) -> TimeControl {
        self.limits
            .time_control(self.options.move_overhead, self.root_ply())
    }

    fn root_ply(&self) -> u32 {
        self.root
            .position()
            .history()
            .len()
            .saturating_sub(1)
            .try_into()
            .unwrap_or(u32::MAX)
    }

    fn pondering_before_hit(&self) -> bool {
        self.limits
            .ponder_hit
            .as_ref()
            .is_some_and(|ponder_hit| !ponder_hit.load(Ordering::Relaxed))
    }

    fn can_extend_after_soft(&self, instability: u32, extensions_used: u32) -> bool {
        if instability == 0 || extensions_used >= MAX_SOFT_EXTENSIONS {
            return false;
        }
        let Some(hard_deadline) = self.hard_deadline else {
            return false;
        };
        Instant::now() + Duration::from_millis(2) < hard_deadline
    }

    fn extract_pv(&self, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut position = self.root.clone();
        for _ in 0..max_len.min(64) {
            if position.is_terminal() {
                break;
            }
            let Some(entry) = self.tt.probe(position.hash()) else {
                break;
            };
            let Some(mv) = entry.best_move else {
                break;
            };
            if !position.is_legal(mv) {
                break;
            }
            pv.push(mv);
            position = position.after_move(mv);
        }
        pv
    }

    fn report(&self, depth: u32, score: i32, pv: &[Move]) {
        let Some(reporter) = &self.reporter else {
            return;
        };
        let elapsed = self.start.elapsed();
        let elapsed_ms = elapsed.as_millis().max(1);
        let nps = self.nodes.saturating_mul(1000) / elapsed_ms as u64;
        let mut position = self.root.clone();
        let mut pv_uci = Vec::with_capacity(pv.len());
        for &mv in pv {
            if position.is_terminal() {
                break;
            }
            pv_uci.push(position.to_uci(mv));
            position = position.after_move(mv);
        }

        reporter(SearchInfo {
            depth,
            seldepth: self.seldepth as u32,
            score,
            nodes: self.nodes,
            nps,
            elapsed_ms,
            hashfull: self.tt.hashfull(),
            tbhits: self.tbhits,
            pv: pv_uci,
        });
    }

    fn syzygy_root_outcome(&mut self, root_moves: &[Move]) -> Option<SearchOutcome> {
        let (best_move, wdl) = self.syzygy_best_move(self.root.position())?;
        if !root_moves.contains(&best_move) {
            return None;
        }

        self.tbhits = self.tbhits.saturating_add(1);
        let score = syzygy_score(wdl, 0, self.options.syzygy_50_move_rule);
        let pv = vec![best_move];
        self.report(0, score, &pv);
        Some(SearchOutcome {
            root: self.root.position().clone(),
            best_move: Some(best_move),
            score,
            depth: 0,
            nodes: self.nodes,
            tbhits: self.tbhits,
            elapsed: self.start.elapsed(),
            pv,
        })
    }

    fn syzygy_best_move(&self, position: &Position) -> Option<(Move, shakmaty_syzygy::Wdl)> {
        let syzygy = self.options.syzygy.as_ref()?;
        if !syzygy.can_probe(position, self.options.syzygy_probe_limit) {
            return None;
        }
        syzygy.best_move(position, self.options.syzygy_50_move_rule)
    }

    fn probe_syzygy_node(
        &mut self,
        position: &NnuePosition,
        depth: i32,
        ply: usize,
    ) -> Option<i32> {
        if depth < self.options.syzygy_probe_depth as i32 {
            return None;
        }
        let syzygy = self.options.syzygy.as_ref()?;
        let position = position.position();
        if position.halfmove_clock() != 0
            || !syzygy.can_probe_at_depth(
                position,
                self.options.syzygy_probe_limit,
                self.options.syzygy_probe_depth,
                depth.max(0) as u32,
            )
        {
            return None;
        }

        let wdl = syzygy.probe_wdl_after_zeroing(position)?;
        self.tbhits = self.tbhits.saturating_add(1);
        Some(syzygy_score(wdl, ply, self.options.syzygy_50_move_rule))
    }
}

fn pick_next(moves: &mut [ScoredMove], index: usize) -> Move {
    let mut best = index;
    for candidate in (index + 1)..moves.len() {
        if moves[candidate].score > moves[best].score {
            best = candidate;
        }
    }
    moves.swap(index, best);
    moves[index].mv
}

fn update_instability(
    current: u32,
    best_changed: bool,
    fail_low_or_drop: bool,
    fail_high_or_swing: bool,
) -> u32 {
    let mut next = current.saturating_sub((!best_changed && !fail_low_or_drop) as u32);
    if best_changed {
        next = next.saturating_add(2);
    }
    if fail_low_or_drop {
        next = next.saturating_add(2);
    } else if fail_high_or_swing {
        next = next.saturating_add(1);
    }
    next.min(6)
}

fn ensure_pv_starts_with(best_move: Move, pv: Vec<Move>) -> Vec<Move> {
    if pv.first().copied() == Some(best_move) {
        pv
    } else {
        vec![best_move]
    }
}

fn history_bonus(depth: i32) -> i32 {
    (depth * depth * 128).clamp(128, MAX_HISTORY_SCORE / 2)
}

fn update_history_stat(entry: &mut i32, bonus: i32) {
    let bonus = bonus.clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE);
    *entry += bonus - (*entry * bonus.abs() / MAX_HISTORY_SCORE);
    *entry = (*entry).clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE);
}

fn move_from(mv: Move) -> Square {
    mv.from().expect("standard chess move has an origin square")
}

fn piece_index(piece: Piece) -> usize {
    usize::from(piece) - 1
}

fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn terminal_score(position: &NnuePosition, ply: usize) -> Option<i32> {
    match position.known_outcome() {
        Some(KnownOutcome::Decisive { .. }) => Some(-MATE + ply as i32),
        Some(KnownOutcome::Draw) => Some(0),
        None => None,
    }
}

fn syzygy_score(wdl: shakmaty_syzygy::Wdl, ply: usize, use_50_move_rule: bool) -> i32 {
    match wdl {
        shakmaty_syzygy::Wdl::Win => TB_WIN_SCORE - ply as i32,
        shakmaty_syzygy::Wdl::Loss => -TB_WIN_SCORE + ply as i32,
        shakmaty_syzygy::Wdl::CursedWin if !use_50_move_rule => TB_WIN_SCORE - ply as i32,
        shakmaty_syzygy::Wdl::BlessedLoss if !use_50_move_rule => -TB_WIN_SCORE + ply as i32,
        shakmaty_syzygy::Wdl::Draw
        | shakmaty_syzygy::Wdl::CursedWin
        | shakmaty_syzygy::Wdl::BlessedLoss => 0,
    }
}

fn has_non_pawn_material(position: &NnuePosition) -> bool {
    let side = position.side_to_move();
    !((position.board().by_color(side) & position.board().by_role(Piece::Knight))
        | (position.board().by_color(side) & position.board().by_role(Piece::Bishop))
        | (position.board().by_color(side) & position.board().by_role(Piece::Rook))
        | (position.board().by_color(side) & position.board().by_role(Piece::Queen)))
    .is_empty()
}

fn reverse_futility_margin(depth: i32) -> i32 {
    80 + depth * 90
}

fn futility_margin(depth: i32) -> i32 {
    depth * 120
}

fn late_move_pruning_threshold(depth: i32) -> usize {
    match depth {
        i32::MIN..=1 => 4,
        2 => 8,
        3 => 12,
        _ => usize::MAX,
    }
}

fn time_check_interval(limits: &SearchLimits) -> u32 {
    limits
        .nodes
        .map(|nodes| {
            (nodes / NODE_LIMIT_CHECK_DIVISOR).clamp(1, TIME_CHECK_INTERVAL_NODES as u64) as u32
        })
        .unwrap_or(TIME_CHECK_INTERVAL_NODES)
}

fn late_move_reduction(depth: i32, index: usize, history_score: i32, is_pv_node: bool) -> i32 {
    let move_number = index as i32 + 1;
    let mut reduction = 1;

    if depth >= 5 && move_number >= 8 {
        reduction += 1;
    }
    if depth >= 7 && move_number >= 16 {
        reduction += 1;
    }
    if !is_pv_node && depth >= 5 && move_number >= 12 {
        reduction += 1;
    }
    if history_score > 10_000 {
        reduction -= 1;
    } else if history_score < -8_000 && depth >= 4 {
        reduction += 1;
    }

    reduction.clamp(1, (depth - 2).max(1))
}

fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

pub fn format_uci_score(score: i32) -> String {
    if score.abs() > MATE_THRESHOLD && score.abs() <= MATE {
        let plies = MATE - score.abs();
        let moves = (plies + 1) / 2;
        if score > 0 {
            format!("mate {moves}")
        } else {
            format!("mate -{moves}")
        }
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_mate_in_one() {
        let position = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            depth: Some(2),
            ..SearchLimits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher = Searcher::new(position, options, limits, tt, stop, None);
        let outcome = searcher.search();
        assert_eq!(
            outcome.best_move.map(|mv| outcome.root.to_uci(mv)),
            Some("g6g7".to_string())
        );
    }

    #[test]
    fn returns_a_legal_startpos_move() {
        let position = Position::startpos();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            depth: Some(1),
            ..SearchLimits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
        let outcome = searcher.search();
        assert!(
            outcome
                .best_move
                .is_some_and(|mv| position.is_legal(mv))
        );
    }

    #[test]
    fn captures_hanging_queen() {
        let position = Position::from_fen("q3k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            depth: Some(1),
            ..SearchLimits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher = Searcher::new(position, options, limits, tt, stop, None);
        let outcome = searcher.search();
        assert_eq!(
            outcome.best_move.map(|mv| outcome.root.to_uci(mv)),
            Some("a1a8".to_string())
        );
    }

    #[test]
    fn root_search_stores_fail_high_as_lower_bound() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1").unwrap();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let limits = SearchLimits::default();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
        let root_moves = searcher.root_moves();
        let (score, _) = searcher.root_search(1, -10, 10, &root_moves);
        let entry = searcher.tt.probe(position.hash()).unwrap();
        assert!(score >= 10);
        assert_eq!(entry.bound, Bound::Lower);
    }

    #[test]
    fn depth_four_search_reports_stable_pv() {
        let position = Position::startpos();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            depth: Some(4),
            ..SearchLimits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher = Searcher::new(position.clone(), options, limits, tt, stop, None);
        let outcome = searcher.search();
        assert_eq!(outcome.depth, 4);
        assert!(
            outcome
                .best_move
                .is_some_and(|mv| position.is_legal(mv))
        );
        assert!(!outcome.pv.is_empty());
    }

    #[test]
    fn time_check_interval_matches_stockfish_node_policy() {
        assert_eq!(time_check_interval(&SearchLimits::default()), 512);
        assert_eq!(
            time_check_interval(&SearchLimits {
                nodes: Some(100),
                ..SearchLimits::default()
            }),
            1
        );
        assert_eq!(
            time_check_interval(&SearchLimits {
                nodes: Some(2048),
                ..SearchLimits::default()
            }),
            2
        );
        assert_eq!(
            time_check_interval(&SearchLimits {
                nodes: Some(1_000_000),
                ..SearchLimits::default()
            }),
            512
        );
    }

    #[test]
    fn hard_clock_uses_countdown_between_now_calls() {
        let position = Position::startpos();
        let options = SearchOptions {
            hash_mb: 4,
            ..SearchOptions::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(options.hash_mb);
        let mut searcher =
            Searcher::new(position, options, SearchLimits::default(), tt, stop, None);
        let deadline = Instant::now() - Duration::from_millis(1);
        searcher.time_check_interval = 512;
        searcher.time_check_countdown = 2;

        assert!(!searcher.hard_time_expired(deadline));
        assert!(searcher.hard_time_expired(deadline));
        assert_eq!(searcher.time_check_countdown, 512);
    }

    #[test]
    fn pv_is_sanitized_to_start_with_bestmove() {
        let position = Position::startpos();
        let best = position.uci_to_move("e2e4").unwrap();
        let other = position.uci_to_move("d2d4").unwrap();

        assert_eq!(
            ensure_pv_starts_with(best, vec![best, other]),
            vec![best, other]
        );
        assert_eq!(ensure_pv_starts_with(best, vec![other]), vec![best]);
        assert_eq!(ensure_pv_starts_with(best, Vec::new()), vec![best]);
    }

    #[test]
    fn score_format_handles_mate() {
        assert_eq!(format_uci_score(MATE - 1), "mate 1");
        assert_eq!(format_uci_score(-MATE + 2), "mate -1");
    }
}
