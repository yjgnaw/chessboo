use std::time::{Duration, Instant};

use cozy_chess::Color;

const MIN_SEARCH_MS: u64 = 1;
const DEFAULT_MOVES_TO_GO: u32 = 28;
const INCREMENT_USAGE_NUMERATOR: u64 = 7;
const INCREMENT_USAGE_DENOMINATOR: u64 = 10;
const HARD_BUDGET_NUMERATOR: u64 = 5;
const HARD_BUDGET_DENOMINATOR: u64 = 2;

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
        ))
    }
}

fn clock_budget(
    remaining: Duration,
    increment: Duration,
    moves_to_go: Option<u32>,
    overhead: Duration,
) -> SearchBudget {
    let safe_ms = duration_ms(reserve_overhead(remaining, overhead));
    let increment_ms = duration_ms(increment);
    let moves = moves_to_go.unwrap_or(DEFAULT_MOVES_TO_GO).clamp(1, 80) as u64;

    let base_ms = safe_ms / moves;
    let increment_part_ms =
        increment_ms.saturating_mul(INCREMENT_USAGE_NUMERATOR) / INCREMENT_USAGE_DENOMINATOR;
    let mut soft_ms = base_ms.saturating_add(increment_part_ms);
    let max_soft_ms = if moves <= 2 {
        safe_ms.saturating_mul(85) / 100
    } else {
        safe_ms.saturating_mul(35) / 100 + increment_ms
    };
    soft_ms = soft_ms.clamp(MIN_SEARCH_MS, max_soft_ms.max(MIN_SEARCH_MS));

    let hard_scaled_ms = soft_ms.saturating_mul(HARD_BUDGET_NUMERATOR) / HARD_BUDGET_DENOMINATOR;
    let hard_candidate_ms =
        hard_scaled_ms.max(soft_ms.saturating_add(increment_ms / 3).saturating_add(20));
    let hard_ms = if moves <= 2 {
        safe_ms
    } else {
        hard_candidate_ms
            .min(safe_ms.saturating_mul(7) / 10)
            .min(safe_ms)
    }
    .max(soft_ms);

    SearchBudget {
        soft: Duration::from_millis(soft_ms),
        hard: Duration::from_millis(hard_ms),
    }
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
        assert_eq!(budget.soft, Duration::from_millis(69));
        assert_eq!(budget.hard, Duration::from_millis(172));
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
        assert_eq!(budget.soft, Duration::from_millis(132));
        assert!(budget.hard > budget.soft);
        assert!(budget.hard < Duration::from_millis(1000));
    }

    #[test]
    fn final_move_can_use_most_of_the_safe_clock() {
        let tc = TimeControl {
            black_time: Some(Duration::from_millis(500)),
            black_increment: Some(Duration::from_millis(0)),
            moves_to_go: Some(1),
            move_overhead: Duration::from_millis(25),
            ..TimeControl::default()
        };

        let budget = tc.budget(Color::Black).unwrap();
        assert_eq!(budget.hard, Duration::from_millis(475));
        assert!(budget.soft < budget.hard);
    }
}
