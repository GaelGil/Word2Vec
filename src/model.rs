use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};

pub struct CbowModel {
    // Continuous Bag of Words Model
    // target_embeddigs: embeddings for the target/center words
    // context_embeddings: embeddings for the surrounding words
    // vocab_size: The size of the vocab
    // embedding_dim: The embedding dimension
    target_embeddings: Embedding,
    context_embeddings: Embedding,
    vocab_size: usize,
    embedding_dim: usize,
}

impl CbowModel {
    pub fn new(vocab_size: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        // create the target embeddings and context embeddings and pass them to the CBOWModel
        // to create it
        let target_embeddings = embedding(vocab_size, embedding_dim, vb.pp("target_embeddings"))?;
        let context_embeddings = embedding(vocab_size, embedding_dim, vb.pp("context_embeddings"))?;

        return Ok(Self {
            target_embeddings,
            context_embeddings,
            vocab_size,
            embedding_dim,
        });
    }

    pub fn get_target_embeddings(&self, target_ids: &Tensor) -> Result<Tensor> {
        return self.target_embeddings.forward(target_ids);
    }

    pub fn get_context_embeddings(&self, context_ids: &Tensor) -> Result<Tensor> {
        return self.context_embeddings.forward(context_ids);
    }

    pub fn context_mean(&self, context_ids: &Tensor) -> Result<Tensor> {
        let context_embeds = self.get_context_embeddings(context_ids)?;
        return context_embeds.mean_keepdim(0);
    }

    pub fn score(&self, target_ids: &Tensor, context_ids: &Tensor) -> Result<Tensor> {
        let target_embed = self.get_target_embeddings(target_ids)?;
        let context_mean = self.get_context_mean(context_ids)?;
        let multiplied = (&target_embed * &context_mean)?;
        let score = multiplied.sum_keepdim(1)?;
        return Ok(score);
    }

    pub fn vocab_size(&self) -> usize {
        return self.vocab_size;
    }

    pub fn embedding_dim(&self) -> usize {
        return self.embedding_dim;
    }
}
