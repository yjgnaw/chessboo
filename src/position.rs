use std::fmt;

use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};

#[derive(Debug, Clone)]
pub struct Position {
    board: Board,
    history: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionError(String);

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PositionError {}

impl Position {
    pub const STARTPOS_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub fn startpos() -> Self {
        Self::from_board(Board::startpos())
    }

    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {
        let board = fen
            .parse::<Board>()
            .map_err(|err| PositionError(format!("invalid FEN `{fen}`: {err}")))?;
        Ok(Self::from_board(board))
    }

    pub fn from_board(board: Board) -> Self {
        let mut position = Self {
            board,
            history: Vec::with_capacity(128),
        };
        position.push_history();
        position
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    pub fn hash(&self) -> u64 {
        self.board.hash()
    }

    pub fn repetition_hash(&self) -> u64 {
        self.board.hash_without_ep()
    }

    pub fn history(&self) -> &[u64] {
        &self.history
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        self.board.generate_moves(|piece_moves| {
            moves.extend(piece_moves);
            false
        });
        moves
    }

    pub fn play(&mut self, mv: Move) -> Result<(), PositionError> {
        if !self.board.is_legal(mv) {
            return Err(PositionError(format!("illegal move {}", self.to_uci(mv))));
        }
        self.board.play_unchecked(mv);
        self.push_history();
        Ok(())
    }

    pub fn after_move(&self, mv: Move) -> Self {
        let mut next = self.clone();
        next.board.play_unchecked(mv);
        next.push_history();
        next
    }

    pub fn null_move(&self) -> Option<Self> {
        let board = self.board.null_move()?;
        Some(Self {
            board,
            history: self.history.clone(),
        })
    }

    pub fn play_uci(&mut self, text: &str) -> Result<Move, PositionError> {
        let mv = self.uci_to_move(text)?;
        self.play(mv)?;
        Ok(mv)
    }

    pub fn uci_to_move(&self, text: &str) -> Result<Move, PositionError> {
        let normalized = text.trim().to_ascii_lowercase();
        if normalized.len() < 4 {
            return Err(PositionError(format!("invalid UCI move `{text}`")));
        }

        let direct = normalized
            .parse::<Move>()
            .map_err(|_| PositionError(format!("invalid UCI move `{text}`")))?;
        if self.board.is_legal(direct) {
            return Ok(direct);
        }

        if let Some(castle) = self.uci_castle_to_internal(&normalized)
            && self.board.is_legal(castle)
        {
            return Ok(castle);
        }

        Err(PositionError(format!("illegal UCI move `{text}`")))
    }

    pub fn to_uci(&self, mv: Move) -> String {
        if self.is_internal_castle(mv) {
            let rank = mv.from.rank();
            let target_file = if mv.to.file() > mv.from.file() {
                File::G
            } else {
                File::C
            };
            let target = Square::new(target_file, rank);
            format!("{}{}{}", mv.from, target, promotion_suffix(mv.promotion))
        } else {
            format!("{}{}{}", mv.from, mv.to, promotion_suffix(mv.promotion))
        }
    }

    pub fn is_draw(&self) -> bool {
        self.board.status() == GameStatus::Drawn
            || self.is_threefold_repetition()
            || self.has_insufficient_material()
    }

    pub fn is_terminal(&self) -> bool {
        self.board.status() != GameStatus::Ongoing || self.is_draw()
    }

    pub fn result_string(&self) -> &'static str {
        if self.is_draw() {
            "1/2-1/2"
        } else if self.board.status() == GameStatus::Won {
            match self.board.side_to_move() {
                Color::White => "0-1",
                Color::Black => "1-0",
            }
        } else {
            "*"
        }
    }

    pub fn is_capture(&self, mv: Move) -> bool {
        if self.board.color_on(mv.to) == Some(!self.board.side_to_move()) {
            return true;
        }
        self.is_en_passant_capture(mv)
    }

    pub fn is_tactical(&self, mv: Move) -> bool {
        self.is_capture(mv) || mv.promotion.is_some()
    }

    pub fn moved_piece(&self, mv: Move) -> Option<Piece> {
        self.board.piece_on(mv.from)
    }

    pub fn captured_piece(&self, mv: Move) -> Option<Piece> {
        if self.is_en_passant_capture(mv) {
            Some(Piece::Pawn)
        } else {
            self.board.piece_on(mv.to)
        }
    }

    pub fn is_quiet(&self, mv: Move) -> bool {
        !self.is_capture(mv) && mv.promotion.is_none()
    }

    fn push_history(&mut self) {
        self.history.push(self.repetition_hash());
    }

    fn is_threefold_repetition(&self) -> bool {
        let key = self.repetition_hash();
        self.history
            .iter()
            .rev()
            .filter(|&&hash| hash == key)
            .count()
            >= 3
    }

    fn has_insufficient_material(&self) -> bool {
        if !(self.board.pieces(Piece::Pawn)
            | self.board.pieces(Piece::Rook)
            | self.board.pieces(Piece::Queen))
        .is_empty()
        {
            return false;
        }

        let white_minors = (self.board.colored_pieces(Color::White, Piece::Knight)
            | self.board.colored_pieces(Color::White, Piece::Bishop))
        .len();
        let black_minors = (self.board.colored_pieces(Color::Black, Piece::Knight)
            | self.board.colored_pieces(Color::Black, Piece::Bishop))
        .len();
        let total_minors = white_minors + black_minors;

        if total_minors <= 1 {
            return true;
        }

        if total_minors == 2
            && white_minors == 1
            && black_minors == 1
            && self.board.pieces(Piece::Knight).is_empty()
        {
            let white_bishop = self
                .board
                .colored_pieces(Color::White, Piece::Bishop)
                .next_square();
            let black_bishop = self
                .board
                .colored_pieces(Color::Black, Piece::Bishop)
                .next_square();
            if let (Some(wb), Some(bb)) = (white_bishop, black_bishop) {
                return square_color(wb) == square_color(bb);
            }
        }

        false
    }

    fn is_en_passant_capture(&self, mv: Move) -> bool {
        if self.board.piece_on(mv.from) != Some(Piece::Pawn) || self.board.piece_on(mv.to).is_some()
        {
            return false;
        }
        let Some(ep_file) = self.board.en_passant() else {
            return false;
        };
        mv.to.file() == ep_file && mv.from.file() != mv.to.file()
    }

    fn is_internal_castle(&self, mv: Move) -> bool {
        self.board.piece_on(mv.from) == Some(Piece::King)
            && self.board.color_on(mv.to) == Some(self.board.side_to_move())
    }

    fn uci_castle_to_internal(&self, text: &str) -> Option<Move> {
        let from = parse_square(text.get(0..2)?)?;
        let to = parse_square(text.get(2..4)?)?;
        if self.board.piece_on(from) != Some(Piece::King) {
            return None;
        }
        let rank = Rank::First.relative_to(self.board.side_to_move());
        if from.rank() != rank || to.rank() != rank {
            return None;
        }

        let rights = self.board.castle_rights(self.board.side_to_move());
        let rook_file = match to.file() {
            File::G => rights.short?,
            File::C => rights.long?,
            _ => return None,
        };
        Some(Move {
            from,
            to: Square::new(rook_file, rank),
            promotion: None,
        })
    }
}

fn parse_square(text: &str) -> Option<Square> {
    text.parse::<Square>().ok()
}

fn promotion_suffix(piece: Option<Piece>) -> &'static str {
    match piece {
        Some(Piece::Knight) => "n",
        Some(Piece::Bishop) => "b",
        Some(Piece::Rook) => "r",
        Some(Piece::Queen) => "q",
        _ => "",
    }
}

fn square_color(square: Square) -> bool {
    (square.file() as u8 + square.rank() as u8).is_multiple_of(2)
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
        assert_eq!(format!("{mv}"), "e1h1");
        assert_eq!(position.to_uci(mv), "e1g1");
    }

    #[test]
    fn rejects_illegal_uci_move() {
        let position = Position::startpos();
        assert!(position.uci_to_move("e1e8").is_err());
    }
}
