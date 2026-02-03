use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct MyModel {
    linear1: Linear,
    linear2: Linear,
}

impl MyModel {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(10, 32, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(32, 2, vb.pp("linear2"))?;

        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}
