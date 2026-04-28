pub mod eval;
pub mod perft;
pub mod position;
pub mod search;
pub mod see;
pub mod time;
pub mod tt;
pub mod uci;

pub const ENGINE_NAME: &str = "Chessboo";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const ENGINE_AUTHOR: &str = "JYWang + Codex";
