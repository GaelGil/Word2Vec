use candle_core::Result;
use candle_nn::{embedding, Embedding, VarBuilder};

pub struct CbowModel {
    target_embeddings: Embedding,
    context_embeddings: Embedding,
    vocab_size: usize,
    embedding_dim: usize,
}

impl CbowModel {
    pub fn new(vocab_size: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let target_embeddings = embedding(vocab_size, embedding_dim, vb.pp("target_embeddings"))?;
        let context_embeddings = embedding(vocab_size, embedding_dim, vb.pp("context_embeddings"))?;

        Ok(Self {
            target_embeddings,
            context_embeddings,
            vocab_size,
            embedding_dim,
        })
    }

    pub fn get_target_embeddings(&self) -> &Embedding {
        &self.target_embeddings
    }

    pub fn get_context_embeddings(&self) -> &Embedding {
        &self.context_embeddings
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}
