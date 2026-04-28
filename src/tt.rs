use cozy_chess::Move;

const CLUSTER_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub key: u64,
    pub depth: i16,
    pub score: i32,
    pub bound: Bound,
    pub best_move: Option<Move>,
}

#[derive(Debug, Clone, Copy)]
struct StoredEntry {
    entry: Entry,
    generation: u8,
}

#[derive(Debug, Clone, Copy, Default)]
struct Cluster {
    entries: [Option<StoredEntry>; CLUSTER_SIZE],
}

#[derive(Debug, Clone)]
pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    used: usize,
    generation: u8,
}

impl TranspositionTable {
    pub fn new(hash_mb: usize) -> Self {
        let bytes = hash_mb.max(1).saturating_mul(1024 * 1024);
        let cluster_size = std::mem::size_of::<Cluster>().max(1);
        let len = (bytes / cluster_size).max(256);
        Self {
            clusters: vec![Cluster::default(); len],
            used: 0,
            generation: 0,
        }
    }

    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        self.clusters.fill(Cluster::default());
        self.used = 0;
        self.generation = 0;
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        self.clusters[self.index(key)]
            .entries
            .iter()
            .flatten()
            .find_map(|stored| (stored.entry.key == key).then_some(stored.entry))
    }

    pub fn store(&mut self, entry: Entry) {
        let index = self.index(entry.key);
        let generation = self.generation;
        let cluster = &mut self.clusters[index];

        if let Some(slot) = cluster
            .entries
            .iter_mut()
            .find(|slot| slot.is_some_and(|stored| stored.entry.key == entry.key))
        {
            let old = slot.expect("matched occupied slot");
            if entry.depth >= old.entry.depth
                || entry.bound == Bound::Exact
                || old.generation != generation
            {
                *slot = Some(StoredEntry { entry, generation });
            }
            return;
        }

        if let Some(slot) = cluster.entries.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(StoredEntry { entry, generation });
            self.used += 1;
            return;
        }

        let replace_index = cluster
            .entries
            .iter()
            .enumerate()
            .max_by_key(|(_, slot)| replacement_priority(slot.expect("occupied slot"), generation))
            .map(|(slot, _)| slot)
            .expect("cluster has slots");

        cluster.entries[replace_index] = Some(StoredEntry { entry, generation });
    }

    pub fn hashfull(&self) -> u64 {
        if self.clusters.is_empty() {
            return 0;
        }
        let total_slots = self.clusters.len() * CLUSTER_SIZE;
        ((self.used.min(total_slots) as u64) * 1000) / total_slots as u64
    }

    fn index(&self, key: u64) -> usize {
        key as usize % self.clusters.len()
    }
}

fn replacement_priority(entry: StoredEntry, generation: u8) -> i32 {
    let stale = (entry.generation != generation) as i32;
    let non_exact = (entry.entry.bound != Bound::Exact) as i32;
    stale * 1_000_000 + non_exact * 10_000 - i32::from(entry.entry.depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::{Move, Square};

    fn entry(key: u64, depth: i16, bound: Bound) -> Entry {
        Entry {
            key,
            depth,
            score: i32::from(depth),
            bound,
            best_move: Some(Move {
                from: Square::A1,
                to: Square::A2,
                promotion: None,
            }),
        }
    }

    #[test]
    fn cluster_probe_finds_colliding_entries() {
        let mut tt = TranspositionTable::new(1);
        let stride = tt.clusters.len() as u64;
        for i in 0..CLUSTER_SIZE {
            tt.store(entry(1 + stride * i as u64, i as i16 + 1, Bound::Lower));
        }

        for i in 0..CLUSTER_SIZE {
            let key = 1 + stride * i as u64;
            assert_eq!(tt.probe(key).map(|entry| entry.depth), Some(i as i16 + 1));
        }
        assert_eq!(tt.used, CLUSTER_SIZE);
    }

    #[test]
    fn replacement_prefers_stale_shallow_non_exact_entries() {
        let mut tt = TranspositionTable::new(1);
        let stride = tt.clusters.len() as u64;
        tt.store(entry(2, 8, Bound::Exact));
        tt.store(entry(2 + stride, 2, Bound::Upper));
        tt.store(entry(2 + stride * 2, 7, Bound::Lower));
        tt.store(entry(2 + stride * 3, 6, Bound::Lower));

        tt.new_search();
        tt.store(entry(2 + stride * 4, 4, Bound::Lower));

        assert!(tt.probe(2).is_some());
        assert!(tt.probe(2 + stride).is_none());
        assert!(tt.probe(2 + stride * 4).is_some());
    }
}
