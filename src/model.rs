use candle_core::Result;
use candle_nn::{Embedding, VarBuilder, embedding};

pub struct CbowModel {
    // Continous Bag of Words Model
    // target_embeddigs: TODO: implement this comment
    // context_embeddings: TODO: implement this comment
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

    pub fn get_target_embeddings(&self) -> &Embedding {
        return &self.target_embeddings;
    }

    pub fn get_context_embeddings(&self) -> &Embedding {
        return &self.context_embeddings;
    }

    pub fn vocab_size(&self) -> usize {
        return self.vocab_size;
    }

    pub fn embedding_dim(&self) -> usize {
        return self.embedding_dim;
    }
}
