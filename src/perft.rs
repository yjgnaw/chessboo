use crate::position::Position;

pub fn perft(position: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = position.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0_u64;
    for mv in moves {
        let child = position.after_move(mv);
        nodes = nodes.saturating_add(perft(&child, depth - 1));
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_perft() {
        let position = Position::startpos();
        assert_eq!(perft(&position, 1), 20);
        assert_eq!(perft(&position, 2), 400);
        assert_eq!(perft(&position, 3), 8902);
    }

    #[test]
    fn castling_perft() {
        let position = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(perft(&position, 1), 26);
        assert_eq!(perft(&position, 2), 568);
    }
}
