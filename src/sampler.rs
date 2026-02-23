use crate::vocab::Vocab;
use rand::distributions::Uniform;
use rand::Rng;
use std::collections::HashMap;
use std::collections::HashSet;

pub struct UnigramSampler {
    vocab: Vocab,
    power: f32,
    table: Vec<u32>,
}

impl UnigramSampler {
    pub fn new(vocab: Vocab, power: f32, table_size: usize) -> Self {
        let mut table = Vec::with_capacity(table_size);
        let vocab_size = vocab.size();

        let mut weights: Vec<f32> = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let word = vocab.get_word(i as u32);
            let count = vocab.word_counts.get(word).copied().unwrap_or(1) as f32;
            weights.push(count.powf(power));
        }

        let total_weight: f32 = weights.iter().sum();
        let mut cumulative = 0.0f32;

        for i in 0..table_size {
            let target = (i as f32) / (table_size as f32);
            while cumulative < target && cumulative < 1.0 {
                for (j, weight) in weights.iter().enumerate() {
                    cumulative += weight / total_weight;
                    if cumulative >= target {
                        table.push(j as u32);
                        break;
                    }
                }
            }
        }

        Self {
            vocab,
            power,
            table,
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R, positive: &[u32], count: usize) -> Vec<u32> {
        let mut negative = Vec::with_capacity(count);
        let mut seen: HashSet<u32> = positive.iter().cloned().collect();

        while negative.len() < count {
            let idx = rng.gen_range(0..self.table.len());
            let word_id = self.table[idx];

            if !seen.contains(&word_id) {
                seen.insert(word_id);
                negative.push(word_id);
            }
        }

        negative
    }

    pub fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
}

pub fn compute_subsampling_probs(vocab: &Vocab, threshold: f32) -> HashMap<u32, f32> {
    let total_words = vocab.total_words() as f32;
    let mut probs = HashMap::new();

    for (word, count) in &vocab.word_counts {
        let id = vocab.get_id(word);
        let freq = (*count as f32) / total_words;
        let prob = (freq.sqrt() / freq + threshold).max(0.0).min(1.0);
        probs.insert(id, prob);
    }

    probs
}

pub fn should_subsample<R: Rng>(word_id: u32, probs: &HashMap<u32, f32>, rng: &mut R) -> bool {
    if let Some(&prob) = probs.get(&word_id) {
        let uniform = Uniform::new_inclusive(0.0f32, 1.0f32);
        rng.sample(uniform) > prob
    } else {
        false
    }
}
