use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use chessboo::perft::perft;
use chessboo::position::Position;
use chessboo::search::{SearchLimits, SearchOptions, Searcher};
use chessboo::tt::TranspositionTable;

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

fn value_after(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == key)
        .map(|pair| pair[1].clone())
}

fn print_help() {
    println!(
        "Chessboo {version}

Usage:
  chessboo uci
  chessboo perft --fen <fen> --depth <n>
  chessboo bench [--depth <n>]
  chessboo selfplay [--games <n>] [--depth <n>] [--nodes <n>] [--plies <n>]",
        version = chessboo::ENGINE_VERSION
    );
}
