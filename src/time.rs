use std::time::{Duration, Instant};

use shakmaty::Color;

const MIN_SEARCH_MS: u64 = 1;
const MAX_MOVES_TO_GO: u32 = 50;
const LOW_TIME_MOVES_TO_GO_NUMERATOR: u64 = 5;
const LOW_TIME_MOVES_TO_GO_DENOMINATOR: u64 = 100;
const MAXIMUM_TIME_NUMERATOR: u64 = 8097;
const MAXIMUM_TIME_DENOMINATOR: u64 = 10_000;

#[derive(Debug, Clone, Default)]
pub struct TimeControl {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub infinite: bool,
    pub white_time: Option<Duration>,
    pub black_time: Option<Duration>,
    pub white_increment: Option<Duration>,
    pub black_increment: Option<Duration>,
    pub moves_to_go: Option<u32>,
    pub move_overhead: Duration,
    pub ply: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchBudget {
    pub soft: Duration,
    pub hard: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SearchDeadlines {
    pub soft: Option<Instant>,
    pub hard: Option<Instant>,
}

impl TimeControl {
    pub fn depth(depth: u32) -> Self {
        Self {
            depth: Some(depth),
            ..Self::default()
        }
    }

    pub fn max_depth(&self) -> u32 {
        self.depth.unwrap_or(64).clamp(1, 128)
    }

    pub fn deadlines(&self, side_to_move: Color, now: Instant) -> SearchDeadlines {
        let Some(budget) = self.budget(side_to_move) else {
            return SearchDeadlines::default();
        };

        SearchDeadlines {
            soft: Some(now + budget.soft),
            hard: Some(now + budget.hard),
        }
    }

    pub fn budget(&self, side_to_move: Color) -> Option<SearchBudget> {
        if self.infinite || self.depth.is_some() || self.nodes.is_some() {
            return None;
        }
        if let Some(movetime) = self.movetime {
            let hard = reserve_overhead(movetime, self.move_overhead);
            return Some(SearchBudget { soft: hard, hard });
        }

        let remaining = match side_to_move {
            Color::White => self.white_time,
            Color::Black => self.black_time,
        }?;
        let increment = match side_to_move {
            Color::White => self.white_increment,
            Color::Black => self.black_increment,
        }
        .unwrap_or_default();

        Some(clock_budget(
            remaining,
            increment,
            self.moves_to_go,
            self.move_overhead,
            self.ply,
        ))
    }
}

fn clock_budget(
    remaining: Duration,
    increment: Duration,
    moves_to_go: Option<u32>,
    overhead: Duration,
    ply: u32,
) -> SearchBudget {
    let remaining_ms = duration_ms(remaining);
    let increment_ms = duration_ms(increment);
    let overhead_ms = duration_ms(overhead);
    let moves = stockfish_moves_to_go(remaining_ms, moves_to_go);
    let time_left_ms = stockfish_time_left(remaining_ms, increment_ms, overhead_ms, moves);
    let (opt_scale, max_scale) =
        stockfish_scales(remaining_ms, time_left_ms, moves_to_go, ply, moves);

    let maximum_current_ms = remaining_ms
        .saturating_mul(MAXIMUM_TIME_NUMERATOR)
        .saturating_div(MAXIMUM_TIME_DENOMINATOR)
        .saturating_sub(overhead_ms)
        .max(MIN_SEARCH_MS);

    let soft_ms = scaled_time(time_left_ms, opt_scale)
        .max(MIN_SEARCH_MS)
        .min(maximum_current_ms);
    let hard_ms = scaled_time(soft_ms, max_scale)
        .max(soft_ms)
        .min(maximum_current_ms);

    SearchBudget {
        soft: Duration::from_millis(soft_ms),
        hard: Duration::from_millis(hard_ms),
    }
}

fn stockfish_moves_to_go(remaining_ms: u64, moves_to_go: Option<u32>) -> u32 {
    let mut moves = moves_to_go.unwrap_or(MAX_MOVES_TO_GO).min(MAX_MOVES_TO_GO);
    if remaining_ms < 1000 {
        moves = ((remaining_ms.saturating_mul(LOW_TIME_MOVES_TO_GO_NUMERATOR)
            / LOW_TIME_MOVES_TO_GO_DENOMINATOR) as u32)
            .max(1);
    }
    moves.max(1)
}

fn stockfish_time_left(
    remaining_ms: u64,
    increment_ms: u64,
    overhead_ms: u64,
    moves_to_go: u32,
) -> u64 {
    let moves = moves_to_go as u64;
    remaining_ms
        .saturating_add(increment_ms.saturating_mul(moves.saturating_sub(1)))
        .saturating_sub(overhead_ms.saturating_mul(2 + moves))
        .max(MIN_SEARCH_MS)
}

fn stockfish_scales(
    remaining_ms: u64,
    time_left_ms: u64,
    moves_to_go: Option<u32>,
    ply: u32,
    moves: u32,
) -> (f64, f64) {
    let remaining = remaining_ms.max(MIN_SEARCH_MS) as f64;
    let time_left = time_left_ms.max(MIN_SEARCH_MS) as f64;
    let ply = ply as f64;

    if moves_to_go.is_none() {
        let original_time_adjust = 0.3272 * time_left.log10() - 0.4141;
        let log_time_in_sec = (remaining / 1000.0).log10();
        let opt_constant = (0.0029869 + 0.00033554 * log_time_in_sec).min(0.004905);
        let max_constant = (3.3744 + 3.0608 * log_time_in_sec).max(3.1441);
        let opt_scale = (0.012112 + (ply + 3.22713).powf(0.46866) * opt_constant)
            .min(0.19404 * remaining / time_left)
            * original_time_adjust;
        let max_scale = (max_constant + ply / 12.352).min(6.873);
        (opt_scale, max_scale)
    } else {
        let moves = moves as f64;
        let opt_scale = ((0.88 + ply / 116.4) / moves).min(0.88 * remaining / time_left);
        let max_scale = 1.3 + 0.11 * moves;
        (opt_scale, max_scale)
    }
}

fn scaled_time(time_ms: u64, scale: f64) -> u64 {
    if !scale.is_finite() || scale <= 0.0 {
        return MIN_SEARCH_MS;
    }
    (time_ms as f64 * scale).max(MIN_SEARCH_MS as f64) as u64
}

fn reserve_overhead(duration: Duration, overhead: Duration) -> Duration {
    duration
        .saturating_sub(overhead)
        .max(Duration::from_millis(MIN_SEARCH_MS))
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movetime_reserves_overhead() {
        let tc = TimeControl {
            movetime: Some(Duration::from_millis(100)),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        assert_eq!(
            tc.budget(Color::White),
            Some(SearchBudget {
                soft: Duration::from_millis(75),
                hard: Duration::from_millis(75),
            })
        );
    }

    #[test]
    fn clock_budget_has_soft_and_hard_limits() {
        let tc = TimeControl {
            white_time: Some(Duration::from_millis(1000)),
            white_increment: Some(Duration::from_millis(50)),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        let budget = tc.budget(Color::White).unwrap();
        assert_eq!(budget.soft, Duration::from_millis(25));
        assert_eq!(budget.hard, Duration::from_millis(84));
        assert!(budget.hard > budget.soft);
        assert!(budget.hard < Duration::from_millis(1000));
    }

    #[test]
    fn explicit_moves_to_go_is_honored() {
        let tc = TimeControl {
            white_time: Some(Duration::from_millis(1000)),
            white_increment: Some(Duration::from_millis(50)),
            moves_to_go: Some(10),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        let budget = tc.budget(Color::White).unwrap();
        assert_eq!(budget.soft, Duration::from_millis(101));
        assert_eq!(budget.hard, Duration::from_millis(242));
        assert!(budget.hard > budget.soft);
        assert!(budget.hard < Duration::from_millis(1000));
    }

    #[test]
    fn subsecond_clock_keeps_stockfish_overhead_reserve() {
        let tc = TimeControl {
            black_time: Some(Duration::from_millis(500)),
            black_increment: Some(Duration::from_millis(0)),
            moves_to_go: Some(1),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        let budget = tc.budget(Color::Black).unwrap();
        assert_eq!(budget.soft, Duration::from_millis(MIN_SEARCH_MS));
        assert_eq!(budget.hard, Duration::from_millis(4));
    }

    #[test]
    fn increment_usage_never_exceeds_overhead_reserved_clock() {
        let tc = TimeControl {
            white_time: Some(Duration::from_millis(5)),
            white_increment: Some(Duration::from_millis(50)),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        let budget = tc.budget(Color::White).unwrap();
        assert_eq!(budget.soft, Duration::from_millis(MIN_SEARCH_MS));
        assert_eq!(budget.hard, Duration::from_millis(MIN_SEARCH_MS));
    }
}
