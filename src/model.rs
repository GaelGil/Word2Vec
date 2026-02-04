use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

pub struct Model {
    embeddings: Embedding,
    linear1: Linear,
    linear2: Linear,
}

impl Model {
    pub fn new(vocab_size: usize, window_size: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = candle_nn::embedding(vocab_size, 100, vb.pp("embeddings"))?;
        let linear1 = candle_nn::linear(100, window_size, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(window_size, vocab_size, vb.pp("linear2"))?;
        Ok(Self {
            embeddings,
            linear1,
            linear2,
        })
    }
}

impl Module for Model {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(x)?;
        let x = self.linear1.forward(&x)?;
        self.linear2.forward(&x)
    }
}
