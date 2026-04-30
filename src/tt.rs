use shakmaty::Move;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const CLUSTER_SIZE: usize = 4;
const HASHFULL_SAMPLE_CLUSTERS: usize = 1000;

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

pub struct TranspositionTable {
    clusters: Arc<Vec<Mutex<Cluster>>>,
    used: Arc<AtomicUsize>,
    generation: Arc<AtomicU8>,
}

impl TranspositionTable {
    pub fn new(hash_mb: usize) -> Self {
        let bytes = hash_mb.max(1).saturating_mul(1024 * 1024);
        let cluster_size = std::mem::size_of::<Mutex<Cluster>>().max(1);
        let len = (bytes / cluster_size).max(256);
        Self {
            clusters: Arc::new((0..len).map(|_| Mutex::new(Cluster::default())).collect()),
            used: Arc::new(AtomicUsize::new(0)),
            generation: Arc::new(AtomicU8::new(0)),
        }
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for cluster in self.clusters.iter() {
            *cluster.lock().expect("tt cluster lock poisoned") = Cluster::default();
        }
        self.used.store(0, Ordering::Relaxed);
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let cluster = self.clusters[self.index(key)]
            .lock()
            .expect("tt cluster lock poisoned");
        cluster
            .entries
            .iter()
            .flatten()
            .find_map(|stored| (stored.entry.key == key).then_some(stored.entry))
    }

    pub fn store(&self, entry: Entry) {
        let index = self.index(entry.key);
        let generation = self.generation.load(Ordering::Relaxed);
        let mut cluster = self.clusters[index]
            .lock()
            .expect("tt cluster lock poisoned");

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
            self.used.fetch_add(1, Ordering::Relaxed);
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
        let sample_clusters = self.clusters.len().min(HASHFULL_SAMPLE_CLUSTERS);
        if sample_clusters == 0 {
            return 0;
        }

        let generation = self.generation.load(Ordering::Relaxed);
        let current_generation_entries = self
            .clusters
            .iter()
            .take(sample_clusters)
            .map(|cluster| {
                cluster
                    .lock()
                    .expect("tt cluster lock poisoned")
                    .entries
                    .iter()
                    .flatten()
                    .filter(|stored| stored.generation == generation)
                    .count()
            })
            .sum::<usize>();
        let sampled_slots = sample_clusters * CLUSTER_SIZE;
        (current_generation_entries as u64 * 1000) / sampled_slots as u64
    }

    fn index(&self, key: u64) -> usize {
        key as usize % self.clusters.len()
    }
}

impl Clone for TranspositionTable {
    fn clone(&self) -> Self {
        Self {
            clusters: Arc::clone(&self.clusters),
            used: Arc::clone(&self.used),
            generation: Arc::clone(&self.generation),
        }
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
    use shakmaty::{Move, Role, Square};

    fn entry(key: u64, depth: i16, bound: Bound) -> Entry {
        Entry {
            key,
            depth,
            score: i32::from(depth),
            bound,
            best_move: Some(Move::Normal {
                role: Role::King,
                from: Square::A1,
                capture: None,
                to: Square::A2,
                promotion: None,
            }),
        }
    }

    #[test]
    fn cluster_probe_finds_colliding_entries() {
        let tt = TranspositionTable::new(1);
        let stride = tt.clusters.len() as u64;
        for i in 0..CLUSTER_SIZE {
            tt.store(entry(1 + stride * i as u64, i as i16 + 1, Bound::Lower));
        }

        for i in 0..CLUSTER_SIZE {
            let key = 1 + stride * i as u64;
            assert_eq!(tt.probe(key).map(|entry| entry.depth), Some(i as i16 + 1));
        }
        assert_eq!(tt.used.load(Ordering::Relaxed), CLUSTER_SIZE);
    }

    #[test]
    fn replacement_prefers_stale_shallow_non_exact_entries() {
        let tt = TranspositionTable::new(1);
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

    #[test]
    fn hashfull_counts_only_current_generation_entries() {
        let tt = TranspositionTable::new(1);
        let stride = tt.clusters.len() as u64;
        for i in 0..CLUSTER_SIZE {
            tt.store(entry(stride * i as u64, i as i16 + 1, Bound::Lower));
        }

        assert!(tt.hashfull() > 0);
        assert_eq!(tt.used.load(Ordering::Relaxed), CLUSTER_SIZE);

        tt.new_search();

        assert_eq!(tt.hashfull(), 0);
        assert_eq!(tt.used.load(Ordering::Relaxed), CLUSTER_SIZE);

        for i in 0..CLUSTER_SIZE {
            tt.store(entry(stride * (CLUSTER_SIZE + i) as u64, 8, Bound::Exact));
        }

        assert!(tt.hashfull() > 0);
    }
}
