mod eval;
mod model;
mod sampler;
mod train;
mod vocab;

use eval::Evaluator;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;
use train::Trainer;
use vocab::Vocab;

fn load_text<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path.as_ref())?;
    let tokens = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok(tokens)
}

fn load_dataset<P: AsRef<Path>>(path: P) -> Result<(Vocab, Vec<u32>), Box<dyn std::error::Error>> {
    let tokens = load_text(path)?;
    let vocab = Vocab::build_from_tokens(&tokens, 1);
    let ids: Vec<u32> = tokens.iter().map(|t| vocab.get_id(t)).collect();
    Ok((vocab, ids))
}

fn init_model(
    vocab: Vocab,
    embedding_dim: usize,
    window_size: usize,
    neg_samples: usize,
    learning_rate: f32,
) -> Result<Trainer, Box<dyn std::error::Error>> {
    let trainer = Trainer::new(
        vocab,
        embedding_dim,
        window_size,
        neg_samples,
        learning_rate,
    )?;
    Ok(trainer)
}

fn train_model(
    mut trainer: Trainer,
    tokens: &[u32],
    epochs: usize,
    batch_size: usize,
) -> Result<Trainer, Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::from_entropy();

    for epoch in 0..epochs {
        let loss = trainer.train_epoch(tokens, &mut rng, batch_size)?;
        println!("Epoch {}: Loss = {:.4}", epoch + 1, loss);
    }

    Ok(trainer)
}

fn evaluate_model(
    vocab: &Vocab,
    trainer: &Trainer,
    test_words: Vec<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let embeddings = trainer.get_embeddings().clone();
    let evaluator = Evaluator::new(vocab.clone(), embeddings);

    println!("\n=== Evaluation ===");

    for word in test_words {
        println!("\nWords similar to '{}':", word);
        match evaluator.find_similar(word, 5) {
            Ok(similar) => {
                for (similar_word, score) in similar {
                    println!("  {}: {:.4}", similar_word, score);
                }
            }
            Err(e) => println!("  Error: {:?}", e),
        }
    }

    println!("\n=== Word Analogies ===");
    let analogies = vec![("king", "man", "queen")];

    for (a, b, c) in analogies {
        println!("\n{} is to {} as {} is to:", a, b, c);
        match evaluator.analogy(a, b, c, 5) {
            Ok(results) => {
                for (word, score) in results {
                    println!("  {}: {:.4}", word, score);
                }
            }
            Err(e) => println!("  Error: {:?}", e),
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = "data.txt";
    let embedding_dim = 100;
    let window_size = 2;
    let neg_samples = 5;
    let learning_rate = 0.01;
    let epochs = 10;
    let batch_size = 512;

    println!("=== Loading Dataset ===");
    let (vocab, tokens) = load_dataset(data_path)?;
    println!("Loaded {} tokens", tokens.len());

    println!("\nVocabulary size: {}", vocab.size());

    println!("\n=== Initializing Model ===");
    let mut trainer = init_model(
        vocab.clone(),
        embedding_dim,
        window_size,
        neg_samples,
        learning_rate,
    )?;

    println!("\n=== Training ===");
    trainer = train_model(trainer, &tokens, epochs, batch_size)?;

    println!("\n=== Saving Model ===");
    trainer.save("model.safetensors")?;

    println!("\n=== Evaluation ===");
    evaluate_model(&vocab, &trainer, vec!["the", "and", "king"])?;

    Ok(())
}
