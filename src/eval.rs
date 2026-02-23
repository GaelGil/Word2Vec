use crate::train::cosine_similarity;
use crate::vocab::Vocab;
use candle_core::{Device, Result, Tensor};
use candle_nn::Embedding;
use std::collections::HashMap;

pub struct Evaluator {
    vocab: Vocab,
    embeddings: Embedding,
}

impl Evaluator {
    pub fn new(vocab: Vocab, embeddings: Embedding) -> Self {
        Self { vocab, embeddings }
    }

    pub fn find_similar(&self, word: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        let word_id = self.vocab.get_id(word);

        let word_embed = self.get_embedding(word_id)?;

        let mut similarities: Vec<(String, f32)> = Vec::new();

        for i in 0..self.vocab.size() {
            if i != word_id as usize {
                let other_embed = self.get_embedding(i as u32)?;
                let sim = cosine_similarity(&word_embed, &other_embed);
                similarities.push((self.vocab.get_word(i as u32).to_string(), sim));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities.into_iter().take(top_k).collect())
    }

    pub fn get_embedding(&self, word_id: u32) -> Result<Vec<f32>> {
        let ids = Tensor::new(&[word_id as u32], &Device::Cpu)?;
        let embed = candle_nn::Module::forward(&self.embeddings, &ids)?;
        Ok(embed.to_vec2()?[0].clone())
    }

    pub fn analogy(&self, a: &str, b: &str, c: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        let a_embed = self.get_embedding(self.vocab.get_id(a))?;
        let b_embed = self.get_embedding(self.vocab.get_id(b))?;
        let c_embed = self.get_embedding(self.vocab.get_id(c))?;

        let target: Vec<f32> = a_embed
            .iter()
            .zip(b_embed.iter())
            .zip(c_embed.iter())
            .map(|((a, b), c)| b - a + c)
            .collect();

        let mut similarities: Vec<(String, f32)> = Vec::new();

        for i in 0..self.vocab.size() {
            let word = self.vocab.get_word(i as u32);
            if word != a && word != b && word != c {
                let other_embed = self.get_embedding(i as u32)?;
                let sim = cosine_similarity(&target, &other_embed);
                similarities.push((word.to_string(), sim));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities.into_iter().take(top_k).collect())
    }

    pub fn compute_similarity(&self, word1: &str, word2: &str) -> Result<f32> {
        let embed1 = self.get_embedding(self.vocab.get_id(word1))?;
        let embed2 = self.get_embedding(self.vocab.get_id(word2))?;
        Ok(cosine_similarity(&embed1, &embed2))
    }

    pub fn most_similar_batch(
        &self,
        words: &[&str],
        top_k: usize,
    ) -> Result<HashMap<String, Vec<(String, f32)>>> {
        let mut results = HashMap::new();

        for &word in words {
            let similar = self.find_similar(word, top_k)?;
            results.insert(word.to_string(), similar);
        }

        Ok(results)
    }
}

pub fn evaluate_analogy_accuracy(
    evaluator: &Evaluator,
    analogies: &[(&str, &str, &str, &str)],
) -> Result<f32> {
    let mut correct = 0;

    for (a, b, c, expected) in analogies {
        let results = evaluator.analogy(a, b, c, 1)?;

        if let Some((predicted, _)) = results.first() {
            if predicted == *expected {
                correct += 1;
            }
        }
    }

    if analogies.is_empty() {
        Ok(0.0)
    } else {
        Ok(correct as f32 / analogies.len() as f32)
    }
}
