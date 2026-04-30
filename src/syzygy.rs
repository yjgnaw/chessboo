use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shakmaty::{Chess, Move};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase, Wdl};

use crate::position::Position;

pub const EMPTY_SYZYGY_PATH: &str = "<empty>";
pub const DEFAULT_SYZYGY_PATH: &str = EMPTY_SYZYGY_PATH;
pub const DEFAULT_SYZYGY_PROBE_DEPTH: u32 = 1;
pub const DEFAULT_SYZYGY_PROBE_LIMIT: usize = 7;

#[derive(Debug)]
pub struct SyzygyTablebase {
    tables: Tablebase<Chess>,
    max_pieces: usize,
    loaded_files: usize,
    directories: Vec<PathBuf>,
}

impl SyzygyTablebase {
    pub fn load(path: &str) -> Result<Arc<Self>, String> {
        let directories = tablebase_directories(path)?;
        if directories.is_empty() {
            return Err(format!("no Syzygy tablebase directories found in `{path}`"));
        }

        let mut tables = Tablebase::<Chess>::new();
        let mut loaded_files = 0;
        for directory in &directories {
            loaded_files += tables.add_directory(directory).map_err(|err| {
                format!(
                    "could not load Syzygy directory `{}`: {err}",
                    directory.display()
                )
            })?;
        }

        if loaded_files == 0 {
            return Err(format!("no Syzygy tablebase files found in `{path}`"));
        }

        let max_pieces = tables.max_pieces();
        Ok(Arc::new(Self {
            tables,
            max_pieces,
            loaded_files,
            directories,
        }))
    }

    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    pub fn loaded_files(&self) -> usize {
        self.loaded_files
    }

    pub fn directory_count(&self) -> usize {
        self.directories.len()
    }

    pub fn can_probe(&self, position: &Position, probe_limit: usize) -> bool {
        let piece_limit = self.max_pieces.min(probe_limit);
        probe_limit > 0 && position.piece_count() <= piece_limit && !position.has_castling_rights()
    }

    pub fn can_probe_at_depth(
        &self,
        position: &Position,
        probe_limit: usize,
        probe_depth: u32,
        depth: u32,
    ) -> bool {
        let piece_limit = self.max_pieces.min(probe_limit);
        if probe_limit == 0
            || position.piece_count() > piece_limit
            || position.has_castling_rights()
        {
            return false;
        }

        let effective_probe_depth = if probe_limit > self.max_pieces {
            0
        } else {
            probe_depth
        };
        position.piece_count() < piece_limit || depth >= effective_probe_depth
    }

    pub fn probe_wdl(&self, position: &Position, use_50_move_rule: bool) -> Option<Wdl> {
        let position = position.to_shakmaty();
        if use_50_move_rule {
            self.tables
                .probe_wdl(&position)
                .ok()
                .and_then(wdl_from_ambiguous)
        } else {
            self.tables.probe_wdl_after_zeroing(&position).ok()
        }
    }

    pub fn probe_wdl_after_zeroing(&self, position: &Position) -> Option<Wdl> {
        let position = position.to_shakmaty();
        self.tables.probe_wdl_after_zeroing(&position).ok()
    }

    pub fn best_move(&self, position: &Position, use_50_move_rule: bool) -> Option<(Move, Wdl)> {
        let shakmaty_position = position.to_shakmaty();
        let (mv, _) = self.tables.best_move(&shakmaty_position).ok()??;
        let wdl = self.probe_wdl(position, use_50_move_rule)?;
        Some((mv, wdl))
    }
}

fn wdl_from_ambiguous(wdl: AmbiguousWdl) -> Option<Wdl> {
    match wdl {
        AmbiguousWdl::Win => Some(Wdl::Win),
        AmbiguousWdl::Loss => Some(Wdl::Loss),
        AmbiguousWdl::Draw
        | AmbiguousWdl::BlessedLoss
        | AmbiguousWdl::CursedWin
        | AmbiguousWdl::MaybeWin
        | AmbiguousWdl::MaybeLoss => Some(Wdl::Draw),
    }
}

fn tablebase_directories(path: &str) -> Result<Vec<PathBuf>, String> {
    if path.trim().is_empty() || path.trim().eq_ignore_ascii_case(EMPTY_SYZYGY_PATH) {
        return Ok(Vec::new());
    }

    let mut directories = Vec::new();
    for path in split_syzygy_path(path) {
        let root = PathBuf::from(path);
        if !root.exists() {
            return Err(format!("Syzygy path `{}` does not exist", root.display()));
        }
        if has_table_files(&root)? {
            directories.push(root);
            continue;
        }
        for entry in fs::read_dir(&root)
            .map_err(|err| format!("could not read Syzygy path `{}`: {err}", root.display()))?
        {
            let entry = entry
                .map_err(|err| format!("could not read Syzygy path `{}`: {err}", root.display()))?;
            let child = entry.path();
            if child.is_dir() && has_table_files(&child)? {
                directories.push(child);
            }
        }
    }
    directories.sort();
    directories.dedup();
    Ok(directories)
}

#[cfg(windows)]
fn split_syzygy_path(path: &str) -> Vec<&str> {
    path.split(';')
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .collect()
}

#[cfg(not(windows))]
fn split_syzygy_path(path: &str) -> Vec<&str> {
    path.split(':')
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .collect()
}

fn has_table_files(path: &Path) -> Result<bool, String> {
    if !path.is_dir() {
        return Ok(false);
    }

    for entry in fs::read_dir(path)
        .map_err(|err| format!("could not read Syzygy path `{}`: {err}", path.display()))?
    {
        let entry = entry
            .map_err(|err| format!("could not read Syzygy path `{}`: {err}", path.display()))?;
        let extension = entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        if matches!(extension.as_deref(), Some("rtbw" | "rtbz")) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL_TEST_SYZYGY_PATH: &str = r"D:\char10_t\chess\syzygy";

    #[test]
    fn empty_syzygy_path_disables_loading() {
        assert!(tablebase_directories(EMPTY_SYZYGY_PATH).unwrap().is_empty());
        assert!(tablebase_directories("").unwrap().is_empty());
    }

    #[test]
    fn probe_depth_uses_stockfish_cardinality_rule() {
        let tables = SyzygyTablebase {
            tables: Tablebase::new(),
            max_pieces: 5,
            loaded_files: 0,
            directories: Vec::new(),
        };
        let three_piece = Position::from_fen("6k1/8/8/3P4/4K3/8/8/8 w - - 0 1").unwrap();
        let five_piece = Position::from_fen("6k1/7p/8/3P4/4K3/4P3/8/8 w - - 0 1").unwrap();

        assert!(tables.can_probe_at_depth(&three_piece, 5, 4, 0));
        assert!(!tables.can_probe_at_depth(&five_piece, 5, 4, 0));
        assert!(tables.can_probe_at_depth(&five_piece, 5, 4, 4));
        assert!(tables.can_probe_at_depth(&five_piece, 7, 4, 0));
    }

    #[test]
    fn local_tablebase_can_probe_kpvk_when_available() {
        if !Path::new(LOCAL_TEST_SYZYGY_PATH).exists() {
            return;
        }

        let tables = SyzygyTablebase::load(LOCAL_TEST_SYZYGY_PATH).unwrap();
        let position = Position::from_fen("6k1/8/8/3P4/4K3/8/8/8 w - - 0 1").unwrap();

        assert_eq!(tables.max_pieces(), 5);
        assert_eq!(tables.probe_wdl(&position, true), Some(Wdl::Win));
        assert!(tables.best_move(&position, true).is_some());
    }
}
