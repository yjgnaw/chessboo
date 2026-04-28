use cozy_chess::{
    BitBoard, Board, Color, Move, Piece, Rank, Square, get_bishop_moves, get_king_moves,
    get_knight_moves, get_pawn_attacks, get_rook_moves,
};

use crate::eval::piece_value;
use crate::position::Position;

pub fn static_exchange_eval(position: &Position, mv: Move) -> i32 {
    if !position.is_tactical(mv) {
        return 0;
    }

    let board = position.board();
    let side = board.side_to_move();
    let Some(moved) = board.piece_on(mv.from) else {
        return 0;
    };

    let captured = position
        .captured_piece(mv)
        .map(piece_value)
        .unwrap_or_default();
    let promoted = mv.promotion.unwrap_or(moved);
    let promotion_gain = promotion_gain(moved, promoted);
    let initial_gain = captured + promotion_gain;
    let Some((pieces, occupied)) = position_after_initial_capture(position, mv, moved, promoted)
    else {
        return initial_gain;
    };

    initial_gain - exchange_reply(!side, mv.to, pieces, occupied, piece_value(promoted))
}

fn position_after_initial_capture(
    position: &Position,
    mv: Move,
    moved: Piece,
    promoted: Piece,
) -> Option<([[BitBoard; Piece::NUM]; Color::NUM], BitBoard)> {
    let board = position.board();
    let side = board.side_to_move();
    let them = !side;
    let target = mv.to.bitboard();
    let from = mv.from.bitboard();
    let captured_square = captured_square(position, mv);
    let captured_piece = position.captured_piece(mv);

    let mut pieces = board_pieces(board);
    let mut occupied = board.occupied();
    pieces[color_index(side)][piece_index(moved)] -= from;
    occupied -= from;

    if let (Some(square), Some(piece)) = (captured_square, captured_piece) {
        let captured = square.bitboard();
        pieces[color_index(them)][piece_index(piece)] -= captured;
        occupied -= captured;
    }

    pieces[color_index(side)][piece_index(promoted)] |= target;
    occupied |= target;
    Some((pieces, occupied))
}

fn exchange_reply(
    side: Color,
    target: Square,
    pieces: [[BitBoard; Piece::NUM]; Color::NUM],
    occupied: BitBoard,
    target_value: i32,
) -> i32 {
    let Some(attacker) = least_valuable_legal_attacker(side, target, pieces, occupied) else {
        return 0;
    };

    let (next_pieces, next_occupied, promoted) =
        make_exchange_move(side, target, attacker, pieces, occupied);
    let gain = target_value + promotion_gain(attacker.piece, promoted);
    (gain
        - exchange_reply(
            !side,
            target,
            next_pieces,
            next_occupied,
            piece_value(promoted),
        ))
    .max(0)
}

#[derive(Debug, Clone, Copy)]
struct Attacker {
    piece: Piece,
    from: Square,
}

fn least_valuable_legal_attacker(
    side: Color,
    target: Square,
    pieces: [[BitBoard; Piece::NUM]; Color::NUM],
    occupied: BitBoard,
) -> Option<Attacker> {
    for piece in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
        let attackers = attackers_for_piece(side, piece, target, pieces, occupied);
        for from in attackers {
            let attacker = Attacker { piece, from };
            if exchange_move_is_legal(side, target, attacker, pieces, occupied) {
                return Some(attacker);
            }
        }
    }
    None
}

fn attackers_for_piece(
    side: Color,
    piece: Piece,
    target: Square,
    pieces: [[BitBoard; Piece::NUM]; Color::NUM],
    occupied: BitBoard,
) -> BitBoard {
    let side_pieces = pieces[color_index(side)];
    match piece {
        Piece::Pawn => get_pawn_attacks(target, !side) & side_pieces[piece_index(Piece::Pawn)],
        Piece::Knight => get_knight_moves(target) & side_pieces[piece_index(Piece::Knight)],
        Piece::Bishop => {
            get_bishop_moves(target, occupied) & side_pieces[piece_index(Piece::Bishop)]
        }
        Piece::Rook => get_rook_moves(target, occupied) & side_pieces[piece_index(Piece::Rook)],
        Piece::Queen => {
            (get_bishop_moves(target, occupied) | get_rook_moves(target, occupied))
                & side_pieces[piece_index(Piece::Queen)]
        }
        Piece::King => get_king_moves(target) & side_pieces[piece_index(Piece::King)],
    }
}

fn exchange_move_is_legal(
    side: Color,
    target: Square,
    attacker: Attacker,
    pieces: [[BitBoard; Piece::NUM]; Color::NUM],
    occupied: BitBoard,
) -> bool {
    let (next_pieces, next_occupied, _) =
        make_exchange_move(side, target, attacker, pieces, occupied);
    let king = next_pieces[color_index(side)][piece_index(Piece::King)]
        .next_square()
        .unwrap_or(target);
    !square_attacked_by(&next_pieces, next_occupied, king, !side, true)
}

fn make_exchange_move(
    side: Color,
    target: Square,
    attacker: Attacker,
    mut pieces: [[BitBoard; Piece::NUM]; Color::NUM],
    mut occupied: BitBoard,
) -> ([[BitBoard; Piece::NUM]; Color::NUM], BitBoard, Piece) {
    let us = color_index(side);
    let them = color_index(!side);
    let from = attacker.from.bitboard();
    let target_bb = target.bitboard();
    let promoted = promotion_piece(attacker.piece, side, target).unwrap_or(attacker.piece);

    pieces[us][piece_index(attacker.piece)] -= from;
    occupied -= from;
    for piece in Piece::ALL {
        pieces[them][piece_index(piece)] -= target_bb;
    }
    pieces[us][piece_index(promoted)] |= target_bb;
    occupied |= target_bb;

    (pieces, occupied, promoted)
}

fn square_attacked_by(
    pieces: &[[BitBoard; Piece::NUM]; Color::NUM],
    occupied: BitBoard,
    square: Square,
    side: Color,
    include_king: bool,
) -> bool {
    let side_pieces = pieces[color_index(side)];
    !(get_pawn_attacks(square, !side) & side_pieces[piece_index(Piece::Pawn)]).is_empty()
        || !(get_knight_moves(square) & side_pieces[piece_index(Piece::Knight)]).is_empty()
        || !(get_bishop_moves(square, occupied)
            & (side_pieces[piece_index(Piece::Bishop)] | side_pieces[piece_index(Piece::Queen)]))
        .is_empty()
        || !(get_rook_moves(square, occupied)
            & (side_pieces[piece_index(Piece::Rook)] | side_pieces[piece_index(Piece::Queen)]))
        .is_empty()
        || (include_king
            && !(get_king_moves(square) & side_pieces[piece_index(Piece::King)]).is_empty())
}

fn board_pieces(board: &Board) -> [[BitBoard; Piece::NUM]; Color::NUM] {
    let mut pieces = [[BitBoard::EMPTY; Piece::NUM]; Color::NUM];
    for color in Color::ALL {
        for piece in Piece::ALL {
            pieces[color_index(color)][piece_index(piece)] = board.colored_pieces(color, piece);
        }
    }
    pieces
}

fn captured_square(position: &Position, mv: Move) -> Option<Square> {
    if !position.is_capture(mv) {
        return None;
    }
    if position.board().piece_on(mv.to).is_some() {
        Some(mv.to)
    } else {
        Some(Square::new(mv.to.file(), mv.from.rank()))
    }
}

fn promotion_piece(piece: Piece, side: Color, target: Square) -> Option<Piece> {
    (piece == Piece::Pawn && target.rank() == Rank::Eighth.relative_to(side))
        .then_some(Piece::Queen)
}

fn promotion_gain(from: Piece, to: Piece) -> i32 {
    if from == Piece::Pawn && to != Piece::Pawn {
        piece_value(to) - piece_value(Piece::Pawn)
    } else {
        0
    }
}

fn piece_index(piece: Piece) -> usize {
    piece as usize
}

fn color_index(color: Color) -> usize {
    color as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winning_capture_is_positive() {
        let position = Position::from_fen("4k3/8/8/3p4/5N2/8/8/4K3 w - - 0 1").unwrap();
        let mv = position.uci_to_move("f4d5").unwrap();
        assert!(static_exchange_eval(&position, mv) > 0);
    }

    #[test]
    fn queen_capture_on_defended_pawn_is_bad() {
        let position = Position::from_fen("4k3/8/5n2/3p4/8/8/8/3QK3 w - - 0 1").unwrap();
        let mv = position.uci_to_move("d1d5").unwrap();
        assert!(static_exchange_eval(&position, mv) < -500);
    }

    #[test]
    fn en_passant_scores_captured_pawn() {
        let position = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let mv = position.uci_to_move("e5d6").unwrap();
        assert_eq!(
            static_exchange_eval(&position, mv),
            piece_value(Piece::Pawn)
        );
    }

    #[test]
    fn quiet_promotion_has_promotion_gain() {
        let position = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = position.uci_to_move("a7a8q").unwrap();
        assert_eq!(
            static_exchange_eval(&position, mv),
            piece_value(Piece::Queen) - piece_value(Piece::Pawn)
        );
    }

    #[test]
    fn xray_recapture_is_seen() {
        let position = Position::from_fen("4k3/8/8/3r4/3p4/8/3Q4/4K3 w - - 0 1").unwrap();
        let mv = position.uci_to_move("d2d4").unwrap();
        assert!(static_exchange_eval(&position, mv) < 0);
    }

    #[test]
    fn pinned_piece_is_not_used_as_recapturer() {
        let position = Position::from_fen("4r1k1/8/8/8/4q3/8/4B3/4K3 b - - 0 1").unwrap();
        let mv = position.uci_to_move("e4e2").unwrap();
        assert_eq!(
            static_exchange_eval(&position, mv),
            piece_value(Piece::Bishop)
        );
    }

    #[test]
    fn king_recapture_on_defended_square_is_excluded() {
        let position = Position::from_fen("8/8/4k3/4b2Q/8/8/8/K3R3 w - - 0 1").unwrap();
        let mv = position.uci_to_move("h5e5").unwrap();
        assert_eq!(
            static_exchange_eval(&position, mv),
            piece_value(Piece::Bishop)
        );
    }

    #[test]
    fn safe_king_recapture_is_counted() {
        let position = Position::from_fen("8/8/4k3/4b2Q/8/8/8/K7 w - - 0 1").unwrap();
        let mv = position.uci_to_move("h5e5").unwrap();
        assert!(static_exchange_eval(&position, mv) < -500);
    }
}
