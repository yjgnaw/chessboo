use std::fmt;
use std::sync::Arc;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{
    Bitboard, Board, CastlingMode, Chess, Color, EnPassantMode, KnownOutcome, Move, MoveList,
    Outcome, Position as ShakmatyPosition, Role, Square,
};

#[derive(Debug, Clone)]
pub struct Position {
    board: Chess,
    history: Arc<HistoryNode>,
    hash: u64,
}

#[derive(Debug)]
struct HistoryNode {
    hash: u64,
    previous: Option<Arc<HistoryNode>>,
    len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionError(String);

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PositionError {}

impl HistoryNode {
    fn root(hash: u64) -> Arc<Self> {
        Arc::new(Self {
            hash,
            previous: None,
            len: 1,
        })
    }

    fn push(previous: &Arc<Self>, hash: u64) -> Arc<Self> {
        Arc::new(Self {
            hash,
            previous: Some(Arc::clone(previous)),
            len: previous.len + 1,
        })
    }
}

fn compute_hash(board: &Chess) -> u64 {
    let Zobrist64(hash) = board.zobrist_hash(EnPassantMode::Legal);
    hash
}

impl Position {
    pub const STARTPOS_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub fn startpos() -> Self {
        Self::from_chess(Chess::new())
    }

    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {
        let parsed = fen
            .parse::<Fen>()
            .map_err(|err| PositionError(format!("invalid FEN `{fen}`: {err}")))?;
        let board = parsed
            .into_position(CastlingMode::Standard)
            .or_else(|err| err.ignore_invalid_ep_square())
            .or_else(|err| err.ignore_invalid_castling_rights())
            .map_err(|err| PositionError(format!("invalid FEN `{fen}`: {err}")))?;
        Ok(Self::from_chess(board))
    }

    pub fn from_chess(board: Chess) -> Self {
        let hash = compute_hash(&board);
        Self {
            board,
            history: HistoryNode::root(hash),
            hash,
        }
    }

    pub fn board(&self) -> &Board {
        self.board.board()
    }

    pub fn to_shakmaty(&self) -> Chess {
        self.board.clone()
    }

    pub fn side_to_move(&self) -> Color {
        self.board.turn()
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn repetition_hash(&self) -> u64 {
        self.hash()
    }

    pub fn ply(&self) -> usize {
        self.history.len.saturating_sub(1)
    }

    pub fn legal_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    pub fn is_legal(&self, mv: Move) -> bool {
        self.board.is_legal(mv)
    }

    pub fn checkers(&self) -> Bitboard {
        self.board.checkers()
    }

    pub fn piece_on(&self, square: Square) -> Option<Role> {
        self.board().role_at(square)
    }

    pub fn color_on(&self, square: Square) -> Option<Color> {
        self.board().color_at(square)
    }

    pub fn piece_count(&self) -> usize {
        self.board().occupied().count()
    }

    pub fn has_castling_rights(&self) -> bool {
        self.board.castles().castling_rights().any()
    }

    pub fn halfmove_clock(&self) -> u8 {
        self.board.halfmoves().min(u32::from(u8::MAX)) as u8
    }

    pub fn play(&mut self, mv: Move) -> Result<(), PositionError> {
        if !self.is_legal(mv) {
            return Err(PositionError(format!("illegal move {}", self.to_uci(mv))));
        }
        self.board.play_unchecked(mv);
        self.hash = compute_hash(&self.board);
        self.push_history();
        Ok(())
    }

    pub fn after_move(&self, mv: Move) -> Self {
        let mut next = self.clone();
        next.board.play_unchecked(mv);
        next.hash = compute_hash(&next.board);
        next.push_history();
        next
    }

    pub fn null_move(&self) -> Option<Self> {
        let board = self.board.clone().swap_turn().ok()?;
        let hash = compute_hash(&board);
        Some(Self {
            board,
            history: Arc::clone(&self.history),
            hash,
        })
    }

    pub fn play_uci(&mut self, text: &str) -> Result<Move, PositionError> {
        let mv = self.uci_to_move(text)?;
        self.play(mv)?;
        Ok(mv)
    }

    pub fn uci_to_move(&self, text: &str) -> Result<Move, PositionError> {
        let uci = text
            .trim()
            .parse::<UciMove>()
            .map_err(|err| PositionError(format!("invalid UCI move `{text}`: {err}")))?;
        uci.to_move(&self.board)
            .map_err(|err| PositionError(format!("illegal UCI move `{text}`: {err}")))
    }

    pub fn to_uci(&self, mv: Move) -> String {
        UciMove::from_move(mv, self.board.castles().mode()).to_string()
    }

    pub fn is_draw(&self) -> bool {
        match self.board.outcome() {
            Outcome::Known(KnownOutcome::Decisive { .. }) => false,
            Outcome::Known(KnownOutcome::Draw) => true,
            Outcome::Unknown => self.is_rule_draw_without_board_outcome(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        match self.board.outcome() {
            Outcome::Known(_) => true,
            Outcome::Unknown => self.is_rule_draw_without_board_outcome(),
        }
    }

    pub fn known_outcome(&self) -> Option<KnownOutcome> {
        self.board.outcome().known()
    }

    pub fn result_string(&self) -> &'static str {
        if self.is_draw() {
            "1/2-1/2"
        } else {
            match self.known_outcome() {
                Some(outcome) => outcome.as_str(),
                None => "*",
            }
        }
    }

    pub fn is_capture(&self, mv: Move) -> bool {
        mv.is_capture()
    }

    pub fn is_tactical(&self, mv: Move) -> bool {
        self.is_capture(mv) || mv.promotion().is_some()
    }

    pub fn moved_piece(&self, mv: Move) -> Option<Role> {
        Some(mv.role())
    }

    pub fn captured_piece(&self, mv: Move) -> Option<Role> {
        mv.capture()
    }

    pub fn is_quiet(&self, mv: Move) -> bool {
        !self.is_capture(mv) && mv.promotion().is_none()
    }

    pub fn is_rule_draw(&self) -> bool {
        self.is_rule_draw_without_board_outcome() || self.board.is_insufficient_material()
    }

    fn push_history(&mut self) {
        self.history = HistoryNode::push(&self.history, self.repetition_hash());
    }

    fn is_threefold_repetition(&self) -> bool {
        let key = self.repetition_hash();
        let mut count = 0;
        let mut node = Some(self.history.as_ref());
        while let Some(history) = node {
            if history.hash == key {
                count += 1;
                if count >= 3 {
                    return true;
                }
            }
            node = history.previous.as_deref();
        }
        false
    }

    fn is_rule_draw_without_board_outcome(&self) -> bool {
        self.is_fifty_move_rule_draw() || self.is_threefold_repetition()
    }

    fn is_fifty_move_rule_draw(&self) -> bool {
        self.board.halfmoves() >= 100
    }
}

#[cfg(test)]
fn square_color(square: Square) -> bool {
    (square.file().to_u32() + square.rank().to_u32()).is_multiple_of(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_has_twenty_legal_moves() {
        assert_eq!(Position::startpos().legal_moves().len(), 20);
    }

    #[test]
    fn translates_standard_uci_castling() {
        let position = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mv = position.uci_to_move("e1g1").unwrap();
        assert!(mv.is_castle());
        assert_eq!(position.to_uci(mv), "e1g1");
    }

    #[test]
    fn rejects_illegal_uci_move() {
        let position = Position::startpos();
        assert!(position.uci_to_move("e1e8").is_err());
    }

    #[test]
    fn bishops_on_same_color_are_insufficient_material() {
        let position = Position::from_fen("4k3/8/1b6/8/8/8/8/2B1K3 w - - 0 1").unwrap();
        let white_bishop = (position.board().by_color(Color::White)
            & position.board().by_role(Role::Bishop))
        .first()
        .unwrap();
        let black_bishop = (position.board().by_color(Color::Black)
            & position.board().by_role(Role::Bishop))
        .first()
        .unwrap();
        assert_eq!(square_color(white_bishop), square_color(black_bishop));
        assert!(position.is_draw());
    }

    #[test]
    fn fifty_move_rule_is_terminal_draw() {
        let position = Position::from_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 1").unwrap();
        assert!(position.is_draw());
        assert!(position.is_terminal());
        assert_eq!(position.result_string(), "1/2-1/2");
    }

    #[test]
    fn checkmate_is_not_overridden_by_fifty_move_rule() {
        let position = Position::from_fen("7k/6Q1/7K/8/8/8/8/8 b - - 100 1").unwrap();
        assert!(!position.is_draw());
        assert!(position.is_terminal());
        assert_eq!(position.result_string(), "1-0");
    }
}
