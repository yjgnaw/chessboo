use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use shakmaty::{
    Bitboard, Board, Color, File, KnownOutcome, Move, MoveList, Rank, Role as Piece, Square,
};

use crate::eval::evaluate;
use crate::position::Position;

const PIECE_COUNT: usize = 6;

pub const FEATURE_COUNT: usize = 2 * PIECE_COUNT * 64;
pub const HIDDEN_SIZE: usize = 128;
pub const LAYER1_SIZE: usize = 2 * HIDDEN_SIZE;
pub const LAYER2_SIZE: usize = 0;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const INTERNAL_EVAL_FILE: &str = "<internal>";

const TEXT_MAGIC: &str = "CHESSBOO_NNUE_BOOTSTRAP_V1";
const BINARY_MAGIC: &[u8; 8] = b"CBNNUE01";
const EMBEDDED_NET: &[u8] = include_bytes!("../nets/chessboo-v1.nnue");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NnueError(String);

impl fmt::Display for NnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NnueError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnueSource {
    Embedded,
    External,
}

#[derive(Debug)]
pub struct NnueNet {
    source: NnueSource,
    checksum: u64,
    weights: NnueWeights,
}

#[derive(Debug)]
enum NnueWeights {
    Bootstrap,
    Dense(Box<DenseWeights>),
}

#[derive(Debug)]
struct DenseWeights {
    l0_weights: Vec<i16>,
    l0_bias: [i16; HIDDEN_SIZE],
    output_weights: Vec<i16>,
    output_bias: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accumulator {
    values: [i16; HIDDEN_SIZE],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatorPair {
    white: Accumulator,
    black: Accumulator,
}

#[derive(Debug, Clone)]
pub struct NnuePosition {
    position: Position,
    accumulators: Option<AccumulatorPair>,
    net: Option<Arc<NnueNet>>,
}

impl NnueNet {
    pub fn embedded() -> Result<Arc<Self>, NnueError> {
        static NET: OnceLock<Result<Arc<NnueNet>, NnueError>> = OnceLock::new();
        NET.get_or_init(|| Self::from_bytes(EMBEDDED_NET, NnueSource::Embedded).map(Arc::new))
            .clone()
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Arc<Self>, NnueError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| {
            NnueError(format!(
                "could not read NNUE file `{}`: {err}",
                path.display()
            ))
        })?;
        Self::from_bytes(&bytes, NnueSource::External).map(Arc::new)
    }

    pub fn from_bytes(bytes: &[u8], source: NnueSource) -> Result<Self, NnueError> {
        if bytes.starts_with(TEXT_MAGIC.as_bytes()) {
            return Ok(Self {
                source,
                checksum: checksum_bytes(bytes),
                weights: NnueWeights::Bootstrap,
            });
        }
        Self::from_binary(bytes, source)
    }

    pub fn source(&self) -> NnueSource {
        self.source
    }

    pub fn checksum(&self) -> u64 {
        self.checksum
    }

    pub fn is_bootstrap(&self) -> bool {
        matches!(self.weights, NnueWeights::Bootstrap)
    }

    pub fn validate(&self) -> Result<(), NnueError> {
        match &self.weights {
            NnueWeights::Bootstrap => Ok(()),
            NnueWeights::Dense(weights) => {
                if weights.l0_weights.len() != FEATURE_COUNT * HIDDEN_SIZE {
                    return Err(NnueError("invalid l0 weight count".to_string()));
                }
                if weights.output_weights.len() != LAYER1_SIZE {
                    return Err(NnueError("invalid output weight count".to_string()));
                }
                Ok(())
            }
        }
    }

    pub fn accumulator_pair(&self, position: &Position) -> AccumulatorPair {
        match &self.weights {
            NnueWeights::Bootstrap => AccumulatorPair::bootstrap(position),
            NnueWeights::Dense(weights) => AccumulatorPair::dense(position, weights),
        }
    }

    pub fn evaluate_position(&self, position: &Position) -> i32 {
        let accumulators = self.accumulator_pair(position);
        self.evaluate_accumulators(position.side_to_move(), &accumulators, position)
    }

    fn evaluate_accumulators(
        &self,
        side_to_move: Color,
        accumulators: &AccumulatorPair,
        position: &Position,
    ) -> i32 {
        match &self.weights {
            NnueWeights::Bootstrap => evaluate(position),
            NnueWeights::Dense(weights) => weights.evaluate(side_to_move, accumulators),
        }
    }

    fn add_feature(&self, accumulator: &mut Accumulator, feature: usize) {
        match &self.weights {
            NnueWeights::Bootstrap => add_bootstrap_feature(accumulator, feature),
            NnueWeights::Dense(weights) => weights.add_feature(accumulator, feature),
        }
    }

    fn remove_feature(&self, accumulator: &mut Accumulator, feature: usize) {
        match &self.weights {
            NnueWeights::Bootstrap => remove_bootstrap_feature(accumulator, feature),
            NnueWeights::Dense(weights) => weights.remove_feature(accumulator, feature),
        }
    }

    fn from_binary(bytes: &[u8], source: NnueSource) -> Result<Self, NnueError> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader.read_exact(8)?;
        if magic != BINARY_MAGIC {
            return Err(NnueError("invalid NNUE magic".to_string()));
        }
        let version = reader.read_u32()?;
        let feature_count = reader.read_u32()? as usize;
        let hidden = reader.read_u32()? as usize;
        let layer1 = reader.read_u32()? as usize;
        let layer2 = reader.read_u32()? as usize;
        let scale = reader.read_i32()?;
        let qa = reader.read_i32()?;
        let qb = reader.read_i32()?;
        let stored_checksum = reader.read_u64()?;

        if version != 1 {
            return Err(NnueError(format!("unsupported NNUE version {version}")));
        }
        if feature_count != FEATURE_COUNT
            || hidden != HIDDEN_SIZE
            || layer1 != LAYER1_SIZE
            || layer2 != LAYER2_SIZE
            || scale != SCALE
            || qa != QA
            || qb != QB
        {
            return Err(NnueError(
                "NNUE architecture does not match Chessboo v1".to_string(),
            ));
        }

        let checksum_start = reader.position();
        let l0_weights = reader.read_i16_vec(FEATURE_COUNT * HIDDEN_SIZE)?;
        let l0_bias = reader.read_i16_array::<HIDDEN_SIZE>()?;
        let output_weights = reader.read_i16_vec(LAYER1_SIZE)?;
        let output_bias = reader.read_i16()?;
        if !reader.is_finished() {
            return Err(NnueError("NNUE file has trailing bytes".to_string()));
        }
        let actual_checksum = checksum_bytes(&bytes[checksum_start..]);
        if stored_checksum != actual_checksum {
            return Err(NnueError("NNUE checksum mismatch".to_string()));
        }

        let net = Self {
            source,
            checksum: actual_checksum,
            weights: NnueWeights::Dense(Box::new(DenseWeights {
                l0_weights,
                l0_bias,
                output_weights,
                output_bias,
            })),
        };
        net.validate()?;
        Ok(net)
    }
}

impl DenseWeights {
    #[inline(always)]
    fn add_feature(&self, accumulator: &mut Accumulator, feature: usize) {
        let offset = feature * HIDDEN_SIZE;
        for index in 0..HIDDEN_SIZE {
            accumulator.values[index] += self.l0_weights[offset + index];
        }
    }

    #[inline(always)]
    fn remove_feature(&self, accumulator: &mut Accumulator, feature: usize) {
        let offset = feature * HIDDEN_SIZE;
        for index in 0..HIDDEN_SIZE {
            accumulator.values[index] -= self.l0_weights[offset + index];
        }
    }

    #[inline(always)]
    fn evaluate(&self, side_to_move: Color, accumulators: &AccumulatorPair) -> i32 {
        let (us, them) = accumulators.by_side(side_to_move);
        let mut output = 0_i64;
        for hidden in 0..HIDDEN_SIZE {
            let us_value = screlu_i16(us.values[hidden]);
            output += i64::from(us_value) * i64::from(self.output_weights[hidden]);
        }
        for hidden in 0..HIDDEN_SIZE {
            let them_value = screlu_i16(them.values[hidden]);
            output += i64::from(them_value) * i64::from(self.output_weights[HIDDEN_SIZE + hidden]);
        }
        output /= i64::from(QA);
        output += i64::from(self.output_bias);
        (output * i64::from(SCALE) / i64::from(QA * QB))
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
}

impl Accumulator {
    fn zeroed() -> Self {
        Self {
            values: [0; HIDDEN_SIZE],
        }
    }
}

impl AccumulatorPair {
    fn bootstrap(position: &Position) -> Self {
        let mut pair = Self {
            white: Accumulator::zeroed(),
            black: Accumulator::zeroed(),
        };
        for feature in active_features(position.board()) {
            add_bootstrap_feature(pair.for_perspective_mut(Color::White), feature.white);
            add_bootstrap_feature(pair.for_perspective_mut(Color::Black), feature.black);
        }
        pair
    }

    fn dense(position: &Position, weights: &DenseWeights) -> Self {
        let mut pair = Self {
            white: Accumulator {
                values: weights.l0_bias,
            },
            black: Accumulator {
                values: weights.l0_bias,
            },
        };
        for feature in active_features(position.board()) {
            weights.add_feature(pair.for_perspective_mut(Color::White), feature.white);
            weights.add_feature(pair.for_perspective_mut(Color::Black), feature.black);
        }
        pair
    }

    fn by_side(&self, side: Color) -> (&Accumulator, &Accumulator) {
        match side {
            Color::White => (&self.white, &self.black),
            Color::Black => (&self.black, &self.white),
        }
    }

    fn for_perspective_mut(&mut self, perspective: Color) -> &mut Accumulator {
        match perspective {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        }
    }
}

impl NnuePosition {
    pub fn new(position: Position, net: Option<Arc<NnueNet>>) -> Self {
        let accumulators = net.as_ref().map(|net| net.accumulator_pair(&position));
        Self {
            position,
            accumulators,
            net,
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn board(&self) -> &Board {
        self.position.board()
    }

    pub fn side_to_move(&self) -> Color {
        self.position.side_to_move()
    }

    pub fn hash(&self) -> u64 {
        self.position.hash()
    }

    pub fn legal_moves(&self) -> MoveList {
        self.position.legal_moves()
    }

    pub fn is_legal(&self, mv: Move) -> bool {
        self.position.is_legal(mv)
    }

    pub fn checkers(&self) -> Bitboard {
        self.position.checkers()
    }

    pub fn known_outcome(&self) -> Option<KnownOutcome> {
        self.position.known_outcome()
    }

    pub fn after_move(&self, mv: Move) -> Self {
        let Some(net) = &self.net else {
            return Self::new(self.position.after_move(mv), None);
        };
        let Some(accumulators) = &self.accumulators else {
            return Self::new(self.position.after_move(mv), Some(Arc::clone(net)));
        };

        let before = &self.position;
        let mut next = Self {
            position: before.after_move(mv),
            accumulators: Some(accumulators.clone()),
            net: Some(Arc::clone(net)),
        };
        next.update_accumulators_for_move(before, mv, net);
        next
    }

    pub fn null_move(&self) -> Option<Self> {
        let position = self.position.null_move()?;
        Some(Self {
            position,
            accumulators: self.accumulators.clone(),
            net: self.net.clone(),
        })
    }

    pub fn evaluate(&self) -> i32 {
        let Some(net) = &self.net else {
            return evaluate(&self.position);
        };
        let Some(accumulators) = &self.accumulators else {
            return net.evaluate_position(&self.position);
        };
        net.evaluate_accumulators(self.position.side_to_move(), accumulators, &self.position)
    }

    pub fn is_draw(&self) -> bool {
        self.position.is_draw()
    }

    pub fn is_rule_draw(&self) -> bool {
        self.position.is_rule_draw()
    }

    pub fn is_terminal(&self) -> bool {
        self.position.is_terminal()
    }

    pub fn is_capture(&self, mv: Move) -> bool {
        self.position.is_capture(mv)
    }

    pub fn is_tactical(&self, mv: Move) -> bool {
        self.position.is_tactical(mv)
    }

    pub fn moved_piece(&self, mv: Move) -> Option<Piece> {
        self.position.moved_piece(mv)
    }

    pub fn captured_piece(&self, mv: Move) -> Option<Piece> {
        self.position.captured_piece(mv)
    }

    pub fn is_quiet(&self, mv: Move) -> bool {
        self.position.is_quiet(mv)
    }

    pub fn to_uci(&self, mv: Move) -> String {
        self.position.to_uci(mv)
    }

    pub fn refresh_matches(&self) -> bool {
        let Some(net) = &self.net else {
            return true;
        };
        self.accumulators
            .as_ref()
            .is_some_and(|accumulators| *accumulators == net.accumulator_pair(&self.position))
    }

    fn update_accumulators_for_move(&mut self, before: &Position, mv: Move, net: &NnueNet) {
        let Some(accumulators) = &mut self.accumulators else {
            return;
        };
        let moving_piece = before.moved_piece(mv).unwrap_or(Piece::Pawn);
        let moving_color = before.side_to_move();

        if before.is_internal_castle_move(mv) {
            let (king_to, rook_to) = castle_target_squares(mv, moving_color);
            remove_piece_features(net, accumulators, moving_color, Piece::King, move_from(mv));
            remove_piece_features(net, accumulators, moving_color, Piece::Rook, mv.to());
            add_piece_features(net, accumulators, moving_color, Piece::King, king_to);
            add_piece_features(net, accumulators, moving_color, Piece::Rook, rook_to);
            debug_assert!(self.refresh_matches());
            return;
        }

        remove_piece_features(net, accumulators, moving_color, moving_piece, move_from(mv));
        if let Some(captured) = before.captured_piece(mv) {
            let captured_square = if before.is_en_passant_move(mv) {
                en_passant_captured_square(mv, moving_color)
            } else {
                mv.to()
            };
            remove_piece_features(net, accumulators, !moving_color, captured, captured_square);
        }
        let placed_piece = mv.promotion().unwrap_or(moving_piece);
        add_piece_features(net, accumulators, moving_color, placed_piece, mv.to());

        debug_assert!(self.refresh_matches());
    }
}

#[derive(Debug, Clone, Copy)]
struct FeaturePair {
    white: usize,
    black: usize,
}

#[inline(always)]
pub fn feature_index(
    perspective: Color,
    piece_color: Color,
    piece: Piece,
    square: Square,
) -> usize {
    let piece = piece_index(piece);
    let slot = if piece_color == perspective {
        piece
    } else {
        PIECE_COUNT + piece
    };
    let square = relative_square(square, perspective).to_usize();
    slot * 64 + square
}

fn active_features(board: &Board) -> Vec<FeaturePair> {
    let mut features = Vec::with_capacity(32);
    for &color in &Color::ALL {
        for &piece in &Piece::ALL {
            for square in board.by_color(color) & board.by_role(piece) {
                let white = feature_index(Color::White, color, piece, square);
                let black = feature_index(Color::Black, color, piece, square);
                features.push(FeaturePair { white, black });
            }
        }
    }
    features
}

#[inline(always)]
fn add_piece_features(
    net: &NnueNet,
    accumulators: &mut AccumulatorPair,
    color: Color,
    piece: Piece,
    square: Square,
) {
    let white = feature_index(Color::White, color, piece, square);
    let black = feature_index(Color::Black, color, piece, square);
    net.add_feature(accumulators.for_perspective_mut(Color::White), white);
    net.add_feature(accumulators.for_perspective_mut(Color::Black), black);
}

#[inline(always)]
fn remove_piece_features(
    net: &NnueNet,
    accumulators: &mut AccumulatorPair,
    color: Color,
    piece: Piece,
    square: Square,
) {
    let white = feature_index(Color::White, color, piece, square);
    let black = feature_index(Color::Black, color, piece, square);
    net.remove_feature(accumulators.for_perspective_mut(Color::White), white);
    net.remove_feature(accumulators.for_perspective_mut(Color::Black), black);
}

#[inline(always)]
fn piece_index(piece: Piece) -> usize {
    usize::from(piece) - 1
}

#[inline(always)]
fn move_from(mv: Move) -> Square {
    mv.from().expect("standard chess move has an origin square")
}

#[inline(always)]
fn relative_square(square: Square, perspective: Color) -> Square {
    match perspective {
        Color::White => square,
        Color::Black => square.flip_vertical(),
    }
}

fn add_bootstrap_feature(accumulator: &mut Accumulator, feature: usize) {
    let slot = (feature / 64) % (2 * PIECE_COUNT);
    let hidden = slot.min(HIDDEN_SIZE - 1);
    accumulator.values[hidden] = accumulator.values[hidden].saturating_add(1);
}

fn remove_bootstrap_feature(accumulator: &mut Accumulator, feature: usize) {
    let slot = (feature / 64) % (2 * PIECE_COUNT);
    let hidden = slot.min(HIDDEN_SIZE - 1);
    accumulator.values[hidden] = accumulator.values[hidden].saturating_sub(1);
}

#[inline(always)]
fn screlu_i16(value: i16) -> i32 {
    let value = i32::from(value).clamp(0, QA);
    value * value
}

fn en_passant_captured_square(mv: Move, moving_color: Color) -> Square {
    let offset = if moving_color == Color::White { -8 } else { 8 };
    mv.to()
        .offset(offset)
        .expect("en passant capture stays on board")
}

fn castle_target_squares(mv: Move, moving_color: Color) -> (Square, Square) {
    let rank = moving_color.relative_rank(Rank::First);
    if mv.to().file() > move_from(mv).file() {
        (
            Square::from_coords(File::G, rank),
            Square::from_coords(File::F, rank),
        )
    } else {
        (
            Square::from_coords(File::C, rank),
            Square::from_coords(File::D, rank),
        )
    }
}

pub fn dense_section_lengths() -> [(&'static str, usize); 4] {
    [
        ("l0w.bin", FEATURE_COUNT * HIDDEN_SIZE * 2),
        ("l0b.bin", HIDDEN_SIZE * 2),
        ("l1w.bin", LAYER1_SIZE * 2),
        ("l1b.bin", 2),
    ]
}

pub fn write_dense_payload(payload: &[u8], out: impl AsRef<Path>) -> Result<(), NnueError> {
    let expected: usize = dense_section_lengths()
        .into_iter()
        .map(|(_, len)| len)
        .sum();
    if payload.len() != expected {
        return Err(NnueError(format!(
            "dense payload has {} bytes, expected {expected}",
            payload.len()
        )));
    }

    let mut bytes = Vec::with_capacity(8 + 8 * 4 + 8 + payload.len());
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
    bytes.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
    bytes.extend_from_slice(&(LAYER1_SIZE as u32).to_le_bytes());
    bytes.extend_from_slice(&(LAYER2_SIZE as u32).to_le_bytes());
    bytes.extend_from_slice(&SCALE.to_le_bytes());
    bytes.extend_from_slice(&QA.to_le_bytes());
    bytes.extend_from_slice(&QB.to_le_bytes());
    bytes.extend_from_slice(&checksum_bytes(payload).to_le_bytes());
    bytes.extend_from_slice(payload);
    fs::write(out.as_ref(), bytes).map_err(|err| {
        NnueError(format!(
            "could not write NNUE file `{}`: {err}",
            out.as_ref().display()
        ))
    })
}

pub fn checksum_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn position(&self) -> usize {
        self.offset
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], NnueError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| NnueError("NNUE file is too large".to_string()))?;
        let Some(bytes) = self.bytes.get(self.offset..end) else {
            return Err(NnueError("truncated NNUE file".to_string()));
        };
        self.offset = end;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32, NnueError> {
        let bytes: [u8; 4] = self.read_exact(4)?.try_into().expect("four bytes");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, NnueError> {
        let bytes: [u8; 4] = self.read_exact(4)?.try_into().expect("four bytes");
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16, NnueError> {
        let bytes: [u8; 2] = self.read_exact(2)?.try_into().expect("two bytes");
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, NnueError> {
        let bytes: [u8; 8] = self.read_exact(8)?.try_into().expect("eight bytes");
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i16_vec(&mut self, len: usize) -> Result<Vec<i16>, NnueError> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let bytes: [u8; 2] = self.read_exact(2)?.try_into().expect("two bytes");
            values.push(i16::from_le_bytes(bytes));
        }
        Ok(values)
    }

    fn read_i16_array<const N: usize>(&mut self) -> Result<[i16; N], NnueError> {
        let mut values = [0_i16; N];
        for value in &mut values {
            let bytes: [u8; 2] = self.read_exact(2)?.try_into().expect("two bytes");
            *value = i16::from_le_bytes(bytes);
        }
        Ok(values)
    }
}

trait PositionNnueExt {
    fn is_en_passant_move(&self, mv: Move) -> bool;
    fn is_internal_castle_move(&self, mv: Move) -> bool;
}

impl PositionNnueExt for Position {
    fn is_en_passant_move(&self, mv: Move) -> bool {
        let _ = self;
        mv.is_en_passant()
    }

    fn is_internal_castle_move(&self, mv: Move) -> bool {
        let _ = self;
        mv.is_castle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_index_is_perspective_relative() {
        let white = feature_index(Color::White, Color::White, Piece::Knight, Square::G1);
        let black = feature_index(Color::Black, Color::Black, Piece::Knight, Square::G8);
        assert_eq!(white, black);

        let white_king = feature_index(Color::White, Color::White, Piece::King, Square::E1);
        let black_king = feature_index(Color::Black, Color::Black, Piece::King, Square::E8);
        assert_eq!(white_king, black_king);
        assert!(white_king < FEATURE_COUNT);
    }

    #[test]
    fn embedded_net_validates() {
        let net = NnueNet::embedded().unwrap();
        net.validate().unwrap();
        assert_eq!(net.source(), NnueSource::Embedded);
        assert!(!net.is_bootstrap());
        assert_ne!(net.checksum(), 0);
    }

    #[test]
    fn parser_rejects_bad_magic() {
        assert!(NnueNet::from_bytes(b"bad", NnueSource::External).is_err());
    }

    #[test]
    fn parser_rejects_truncated_binary() {
        let mut bytes = Vec::from(BINARY_MAGIC.as_slice());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        assert!(NnueNet::from_bytes(&bytes, NnueSource::External).is_err());
    }

    #[test]
    fn parser_rejects_wrong_architecture() {
        let mut bytes = Vec::from(BINARY_MAGIC.as_slice());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER1_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER2_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&SCALE.to_le_bytes());
        bytes.extend_from_slice(&QA.to_le_bytes());
        bytes.extend_from_slice(&QB.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        let err = NnueNet::from_bytes(&bytes, NnueSource::External).unwrap_err();
        assert!(err.to_string().contains("architecture"));
    }

    #[test]
    fn parser_rejects_old_halfkp_architecture() {
        let mut bytes = Vec::from(BINARY_MAGIC.as_slice());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(64_u32 * 10 * 64).to_le_bytes());
        bytes.extend_from_slice(&256_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        bytes.extend_from_slice(&SCALE.to_le_bytes());
        bytes.extend_from_slice(&QA.to_le_bytes());
        bytes.extend_from_slice(&QB.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        let err = NnueNet::from_bytes(&bytes, NnueSource::External).unwrap_err();
        assert!(err.to_string().contains("architecture"));
    }

    #[test]
    fn parser_rejects_old_p768_h1024_architecture() {
        let mut bytes = Vec::from(BINARY_MAGIC.as_slice());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
        bytes.extend_from_slice(&1024_u32.to_le_bytes());
        bytes.extend_from_slice(&(LAYER1_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER2_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&SCALE.to_le_bytes());
        bytes.extend_from_slice(&QA.to_le_bytes());
        bytes.extend_from_slice(&QB.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        let err = NnueNet::from_bytes(&bytes, NnueSource::External).unwrap_err();
        assert!(err.to_string().contains("architecture"));
    }

    #[test]
    fn parser_rejects_checksum_mismatch() {
        let payload_len: usize = dense_section_lengths()
            .into_iter()
            .map(|(_, len)| len)
            .sum();
        let payload = vec![0_u8; payload_len];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BINARY_MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
        bytes.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER1_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER2_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&SCALE.to_le_bytes());
        bytes.extend_from_slice(&QA.to_le_bytes());
        bytes.extend_from_slice(&QB.to_le_bytes());
        bytes.extend_from_slice(&(checksum_bytes(&payload) ^ 1).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let err = NnueNet::from_bytes(&bytes, NnueSource::External).unwrap_err();
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn dense_binary_evaluates_both_perspectives() {
        let mut payload = Vec::new();
        payload.extend(std::iter::repeat_n(0_u8, FEATURE_COUNT * HIDDEN_SIZE * 2));
        for _ in 0..HIDDEN_SIZE {
            payload.extend_from_slice(&10_i16.to_le_bytes());
        }
        for _ in 0..HIDDEN_SIZE {
            payload.extend_from_slice(&2_i16.to_le_bytes());
        }
        for _ in 0..HIDDEN_SIZE {
            payload.extend_from_slice(&3_i16.to_le_bytes());
        }
        payload.extend_from_slice(&5_i16.to_le_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(BINARY_MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
        bytes.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER1_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(LAYER2_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&SCALE.to_le_bytes());
        bytes.extend_from_slice(&QA.to_le_bytes());
        bytes.extend_from_slice(&QB.to_le_bytes());
        bytes.extend_from_slice(&checksum_bytes(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let net = Arc::new(NnueNet::from_bytes(&bytes, NnueSource::External).unwrap());
        let position = Position::startpos();
        let score = NnuePosition::new(position, Some(net)).evaluate();

        let hidden = i64::from(screlu_i16(10));
        let output = (HIDDEN_SIZE as i64 * hidden * i64::from(2 + 3) / i64::from(QA)) + 5;
        let expected = (output * i64::from(SCALE) / i64::from(QA * QB)) as i32;
        assert_eq!(score, expected);
    }

    #[test]
    fn incremental_quiet_move_matches_refresh() {
        let net = NnueNet::embedded().unwrap();
        let position = Position::startpos();
        let mv = position.uci_to_move("g1f3").unwrap();
        let nnue_position = NnuePosition::new(position, Some(net));
        let child = nnue_position.after_move(mv);
        assert!(child.refresh_matches());
    }

    #[test]
    fn incremental_special_moves_match_refresh() {
        let net = NnueNet::embedded().unwrap();
        let cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
            ("4k3/8/8/8/8/8/q7/R3K3 w Q - 0 1", "a1a2"),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1d1"),
        ];
        for (fen, uci) in cases {
            let position = Position::from_fen(fen).unwrap();
            let mv = position.uci_to_move(uci).unwrap();
            let nnue_position = NnuePosition::new(position, Some(Arc::clone(&net)));
            let child = nnue_position.after_move(mv);
            assert!(child.refresh_matches(), "{fen} {uci}");
        }
    }

    #[test]
    fn use_nnue_false_equivalent_is_classical() {
        let position = Position::startpos();
        let nnue_position = NnuePosition::new(position.clone(), None);
        assert_eq!(nnue_position.evaluate(), evaluate(&position));
    }
}
