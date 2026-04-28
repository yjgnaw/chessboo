use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use vampirc_uci::{UciMessage, UciTimeControl, parse_one};

use crate::position::Position;
use crate::search::{
    SearchInfo, SearchLimits, SearchOptions, SearchReporter, Searcher, format_uci_score,
};
use crate::tt::TranspositionTable;
use crate::{ENGINE_AUTHOR, ENGINE_NAME, ENGINE_VERSION};

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
    tt: Option<TranspositionTable>,
    search: Option<SearchHandle>,
}

struct SearchHandle {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<TranspositionTable>,
}

impl UciEngine {
    fn new() -> Self {
        let options = SearchOptions::default();
        Self {
            position: Position::startpos(),
            tt: Some(TranspositionTable::new(options.hash_mb)),
            options,
            search: None,
        }
    }

    fn handle_line(&mut self, line: &str) -> bool {
        self.reap_finished();
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
                self.start_search(time_control, search_control);
            }
            UciMessage::Stop => self.stop_search(),
            UciMessage::Quit => {
                self.stop_search();
                return true;
            }
            UciMessage::SetOption { name, value } => self.set_option(&name, value.as_deref()),
            UciMessage::Debug(_)
            | UciMessage::PonderHit
            | UciMessage::Register { .. }
            | UciMessage::Unknown(_, _) => {}
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
        apply_default_go_behavior(&mut limits);

        let position = self.position.clone();
        let options = self.options.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let tt = self
            .tt
            .take()
            .unwrap_or_else(|| TranspositionTable::new(options.hash_mb));
        let reporter: SearchReporter = Box::new(print_info);

        let handle = thread::spawn(move || {
            let mut searcher =
                Searcher::new(position, options, limits, tt, worker_stop, Some(reporter));
            let outcome = searcher.search();
            let bestmove = outcome
                .best_move
                .map(|mv| outcome.root.to_uci(mv))
                .unwrap_or_else(|| "0000".to_string());
            println!("bestmove {bestmove}");
            flush_stdout();
            searcher.into_tt()
        });

        self.search = Some(SearchHandle { stop, handle });
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

fn print_info(info: SearchInfo) {
    let pv = if info.pv.is_empty() {
        String::new()
    } else {
        format!(" pv {}", info.pv.join(" "))
    };
    println!(
        "info depth {} seldepth {} score {} nodes {} nps {} time {} hashfull {}{}",
        info.depth,
        info.seldepth,
        format_uci_score(info.score),
        info.nodes,
        info.nps,
        info.elapsed_ms,
        info.hashfull,
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

    #[test]
    fn parses_position_with_moves() {
        let mut engine = UciEngine::new();
        engine.handle_line("position startpos moves e2e4 e7e5");
        assert_eq!(
            engine.position.board().side_to_move(),
            cozy_chess::Color::White
        );
    }

    #[test]
    fn hash_option_resizes_table() {
        let mut engine = UciEngine::new();
        engine.handle_line("setoption name Hash value 4");
        assert_eq!(engine.options.hash_mb, 4);
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
}
