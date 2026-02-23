use std::collections::HashMap;
use std::path::Path;

#[derive(Clone)]
pub struct Vocab {
    pub word_to_id: HashMap<String, u32>,
    pub id_to_word: Vec<String>,
    pub word_counts: HashMap<String, u64>,
}

impl Vocab {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            word_counts: HashMap::new(),
        }
    }

    pub fn build_from_text<P: AsRef<Path>>(
        path: P,
        min_count: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let tokens = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let mut word_counts: HashMap<String, u64> = HashMap::new();
        for token in &tokens {
            *word_counts.entry(token.clone()).or_insert(0) += 1;
        }

        word_counts.retain(|_, count| *count >= min_count as u64);

        let mut vocab = Self::new();
        vocab.word_to_id.insert("<unk>".to_string(), 0);
        vocab.id_to_word.push("<unk>".to_string());
        vocab.word_counts.insert("<unk>".to_string(), 0);

        for (word, count) in word_counts {
            let id = vocab.id_to_word.len() as u32;
            vocab.word_to_id.insert(word.clone(), id);
            vocab.id_to_word.push(word.clone());
            vocab.word_counts.insert(word.clone(), count);
        }

        Ok(vocab)
    }

    pub fn build_from_tokens(tokens: &[String], min_count: u32) -> Self {
        let mut word_counts: HashMap<String, u64> = HashMap::new();
        for token in tokens {
            *word_counts.entry(token.clone()).or_insert(0) += 1;
        }

        word_counts.retain(|_, count| *count >= min_count as u64);

        let mut vocab = Self::new();
        vocab.word_to_id.insert("<unk>".to_string(), 0);
        vocab.id_to_word.push("<unk>".to_string());
        vocab.word_counts.insert("<unk>".to_string(), 0);

        for (word, count) in word_counts {
            let id = vocab.id_to_word.len() as u32;
            vocab.word_to_id.insert(word.clone(), id);
            vocab.id_to_word.push(word.clone());
            vocab.word_counts.insert(word.clone(), count);
        }

        vocab
    }

    pub fn get_id(&self, word: &str) -> u32 {
        self.word_to_id.get(word).copied().unwrap_or(0)
    }

    pub fn get_word(&self, id: u32) -> &str {
        self.id_to_word
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn total_words(&self) -> u64 {
        self.word_counts.values().sum()
    }
}
