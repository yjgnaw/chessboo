use cozy_chess::{
    BitBoard, Board, Color, File, Piece, Square, get_bishop_moves, get_king_moves,
    get_knight_moves, get_pawn_attacks, get_rook_moves,
};

use crate::position::Position;

const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];
const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

#[derive(Debug, Clone, Copy, Default)]
struct Score {
    mg: i32,
    eg: i32,
}

impl Score {
    fn add(&mut self, other: Score, sign: i32) {
        self.mg += other.mg * sign;
        self.eg += other.eg * sign;
    }
}

pub fn evaluate(position: &Position) -> i32 {
    let white_score = evaluate_board(position.board());
    match position.side_to_move() {
        Color::White => white_score,
        Color::Black => -white_score,
    }
}

pub fn evaluate_board(board: &Board) -> i32 {
    let mut score = Score::default();
    let mut phase = 0;

    for &color in &Color::ALL {
        let sign = if color == Color::White { 1 } else { -1 };
        let mut side = Score::default();

        for &piece in &Piece::ALL {
            let count = board.colored_pieces(color, piece).len() as i32;
            phase += PHASE[piece_index(piece)] * count;
            for square in board.colored_pieces(color, piece) {
                let idx = piece_index(piece);
                side.mg += MG_VALUE[idx] + piece_square(piece, color, square).mg;
                side.eg += EG_VALUE[idx] + piece_square(piece, color, square).eg;
            }
        }

        side.add(pawn_structure(board, color), 1);
        side.add(piece_activity(board, color), 1);
        side.add(king_safety(board, color), 1);
        side.add(mop_up(board, color), 1);
        score.add(side, sign);
    }

    phase = phase.clamp(0, MAX_PHASE);
    (score.mg * phase + score.eg * (MAX_PHASE - phase)) / MAX_PHASE
}

pub fn piece_value(piece: Piece) -> i32 {
    MG_VALUE[piece_index(piece)]
}

fn piece_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

fn piece_square(piece: Piece, color: Color, square: Square) -> Score {
    let file = square.file() as i32;
    let rank = relative_rank(color, square);
    let file_center = 3 - (file - 3).abs().min((file - 4).abs());
    let center = file_center + (3 - (rank - 3).abs().min((rank - 4).abs()));
    let edge = file.min(7 - file) + rank.min(7 - rank);

    match piece {
        Piece::Pawn => Score {
            mg: rank * 9 + file_center * 3,
            eg: rank * 18 + file_center * 2,
        },
        Piece::Knight => Score {
            mg: center * 18 + edge * 5 - 35,
            eg: center * 10 + edge * 4 - 20,
        },
        Piece::Bishop => Score {
            mg: center * 10 + edge * 2 - 18,
            eg: center * 8 + edge * 2 - 10,
        },
        Piece::Rook => Score {
            mg: rank * 3,
            eg: rank * 6,
        },
        Piece::Queen => Score {
            mg: center * 3,
            eg: center * 5,
        },
        Piece::King => Score {
            mg: -(center * 16) - rank * 12,
            eg: center * 18 + rank * 8,
        },
    }
}

fn pawn_structure(board: &Board, color: Color) -> Score {
    let mut score = Score::default();
    let pawns = board.colored_pieces(color, Piece::Pawn);
    let enemy_pawns = board.colored_pieces(!color, Piece::Pawn);

    for file_idx in 0..8 {
        let file = File::try_index(file_idx).expect("valid file");
        let count = (pawns & file.bitboard()).len() as i32;
        if count > 1 {
            score.mg -= 12 * (count - 1);
            score.eg -= 18 * (count - 1);
        }
    }

    for pawn in pawns {
        let file = pawn.file() as i32;
        let rank = relative_rank(color, pawn);

        if !has_friendly_pawn_on_adjacent_file(board, color, file) {
            score.mg -= 10;
            score.eg -= 14;
        }

        if is_passed_pawn(color, pawn, enemy_pawns) {
            let bonus = [0, 8, 16, 28, 50, 85, 140, 0][rank as usize];
            score.mg += bonus / 2;
            score.eg += bonus;
        }
    }

    score
}

fn piece_activity(board: &Board, color: Color) -> Score {
    let own = board.colors(color);
    let blockers = board.occupied();
    let mut score = Score::default();

    let bishops = board.colored_pieces(color, Piece::Bishop).len();
    if bishops >= 2 {
        score.mg += 32;
        score.eg += 48;
    }

    for knight in board.colored_pieces(color, Piece::Knight) {
        let mobility = (get_knight_moves(knight) & !own).len() as i32;
        score.mg += mobility * 4;
        score.eg += mobility * 3;
    }
    for bishop in board.colored_pieces(color, Piece::Bishop) {
        let mobility = (get_bishop_moves(bishop, blockers) & !own).len() as i32;
        score.mg += mobility * 5;
        score.eg += mobility * 4;
    }
    for rook in board.colored_pieces(color, Piece::Rook) {
        let mobility = (get_rook_moves(rook, blockers) & !own).len() as i32;
        score.mg += mobility * 2;
        score.eg += mobility * 3;
        let file = rook.file();
        let friendly_pawns = board.colored_pieces(color, Piece::Pawn) & file.bitboard();
        let enemy_pawns = board.colored_pieces(!color, Piece::Pawn) & file.bitboard();
        if friendly_pawns.is_empty() && enemy_pawns.is_empty() {
            score.mg += 22;
            score.eg += 12;
        } else if friendly_pawns.is_empty() {
            score.mg += 12;
            score.eg += 6;
        }
    }
    for queen in board.colored_pieces(color, Piece::Queen) {
        let mobility = ((get_rook_moves(queen, blockers) | get_bishop_moves(queen, blockers))
            & !own)
            .len() as i32;
        score.mg += mobility;
        score.eg += mobility * 2;
    }

    score.add(threats(board, color), 1);
    score
}

fn threats(board: &Board, color: Color) -> Score {
    let attacks = attacks_by(board, color);
    let enemy = board.colors(!color);
    let attacked = attacks & enemy;
    let mut score = Score::default();
    for square in attacked {
        if let Some(piece) = board.piece_on(square)
            && piece != Piece::King
        {
            score.mg += piece_value(piece) / 18;
            score.eg += piece_value(piece) / 24;
        }
    }
    score
}

fn king_safety(board: &Board, color: Color) -> Score {
    let king = board.king(color);
    let enemy_attacks = attacks_by(board, !color);
    let king_zone = get_king_moves(king) | king.bitboard();
    let attacked_zone = (enemy_attacks & king_zone).len() as i32;
    let mut shield = 0;
    let forward_rank = match color {
        Color::White => king.rank() as i32 + 1,
        Color::Black => king.rank() as i32 - 1,
    };
    if (0..8).contains(&forward_rank) {
        let king_file = king.file() as i32;
        for file in (king_file - 1)..=(king_file + 1) {
            if (0..8).contains(&file) {
                let square = Square::new(
                    File::try_index(file as usize).expect("valid file"),
                    cozy_chess::Rank::try_index(forward_rank as usize).expect("valid rank"),
                );
                if board.colored_pieces(color, Piece::Pawn).has(square) {
                    shield += 1;
                }
            }
        }
    }

    Score {
        mg: shield * 12 - attacked_zone * 18,
        eg: -attacked_zone * 4,
    }
}

fn mop_up(board: &Board, color: Color) -> Score {
    let own_material = non_king_material(board, color);
    let enemy_material = non_king_material(board, !color);
    if enemy_material > 0 || own_material - enemy_material < 450 {
        return Score::default();
    }

    let own_king = board.king(color);
    let enemy_king = board.king(!color);
    let edge_distance = distance_to_edge(enemy_king);
    let edge_bonus = (6 - edge_distance).max(0);
    let king_distance = square_distance(own_king, enemy_king);
    let king_proximity = (14 - king_distance).max(0);

    Score {
        mg: edge_bonus * 2,
        eg: edge_bonus * 18 + king_proximity * 4,
    }
}

fn non_king_material(board: &Board, color: Color) -> i32 {
    Piece::ALL
        .iter()
        .copied()
        .filter(|&piece| piece != Piece::King)
        .map(|piece| EG_VALUE[piece_index(piece)] * board.colored_pieces(color, piece).len() as i32)
        .sum()
}

fn distance_to_edge(square: Square) -> i32 {
    let file = square.file() as i32;
    let rank = square.rank() as i32;
    file.min(7 - file) + rank.min(7 - rank)
}

fn square_distance(a: Square, b: Square) -> i32 {
    (a.file() as i32 - b.file() as i32).abs() + (a.rank() as i32 - b.rank() as i32).abs()
}

fn attacks_by(board: &Board, color: Color) -> BitBoard {
    let blockers = board.occupied();
    let mut attacks = BitBoard::EMPTY;
    for pawn in board.colored_pieces(color, Piece::Pawn) {
        attacks |= get_pawn_attacks(pawn, color);
    }
    for knight in board.colored_pieces(color, Piece::Knight) {
        attacks |= get_knight_moves(knight);
    }
    for bishop in board.colored_pieces(color, Piece::Bishop) {
        attacks |= get_bishop_moves(bishop, blockers);
    }
    for rook in board.colored_pieces(color, Piece::Rook) {
        attacks |= get_rook_moves(rook, blockers);
    }
    for queen in board.colored_pieces(color, Piece::Queen) {
        attacks |= get_rook_moves(queen, blockers) | get_bishop_moves(queen, blockers);
    }
    attacks | get_king_moves(board.king(color))
}

fn has_friendly_pawn_on_adjacent_file(board: &Board, color: Color, file: i32) -> bool {
    let pawns = board.colored_pieces(color, Piece::Pawn);
    for adjacent in [file - 1, file + 1] {
        if (0..8).contains(&adjacent) {
            let file = File::try_index(adjacent as usize).expect("valid file");
            if !(pawns & file.bitboard()).is_empty() {
                return true;
            }
        }
    }
    false
}

fn is_passed_pawn(color: Color, pawn: Square, enemy_pawns: BitBoard) -> bool {
    let file = pawn.file() as i32;
    let rank = pawn.rank() as i32;
    for enemy in enemy_pawns {
        let enemy_file = enemy.file() as i32;
        if (enemy_file - file).abs() > 1 {
            continue;
        }
        let enemy_rank = enemy.rank() as i32;
        match color {
            Color::White if enemy_rank > rank => return false,
            Color::Black if enemy_rank < rank => return false,
            _ => {}
        }
    }
    true
}

fn relative_rank(color: Color, square: Square) -> i32 {
    let rank = square.rank() as i32;
    match color {
        Color::White => rank,
        Color::Black => 7 - rank,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_is_roughly_equal() {
        let position = Position::startpos();
        assert!(evaluate(&position).abs() < 20);
    }

    #[test]
    fn queen_advantage_is_large() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1").unwrap();
        assert!(evaluate(&position) > 800);
    }

    #[test]
    fn won_endgames_prefer_enemy_king_on_edge() {
        let edge = Position::from_fen("7k/8/8/8/8/8/8/KQ6 w - - 0 1").unwrap();
        let center = Position::from_fen("8/8/8/3k4/8/8/8/KQ6 w - - 0 1").unwrap();
        assert!(evaluate(&edge) > evaluate(&center));
    }
}
