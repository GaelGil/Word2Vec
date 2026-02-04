use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

pub struct MyModel {
    embeddings: Embedding,
    linear1: Linear,
    linear2: Linear,
}

impl MyModel {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let embeddings = candle_nn::embedding(100, 100, vb.pp("embeddings"))?;
        let linear1 = candle_nn::linear(100, 3, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(3, 100, vb.pp("linear2"))?;
        Ok(Self {
            embeddings,
            linear1,
            linear2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // let x = self.linear1.forward(x)?;
        // let x = x.relu()?;
        // self.linear2.forward(&x)
        let x = self.embeddings.forward(x)?;
        let x = self.linear1.forward(&x)?;
        self.linear2.forward(&x)
    }
}
