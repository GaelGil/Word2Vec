mod model;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use model::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let vb = VarBuilder::from_dtype(DType::F32, &device);
    let model = Model::new(vb)?;

    let x = Tensor::randn(0f32, 1f32, (4, 10), &device)?;

    let y = model.forward(&x)?;
    println!("{y}");
    Ok(())
}
