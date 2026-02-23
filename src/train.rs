use crate::vocab::Vocab;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, AdamW, Embedding, Module, Optimizer, VarBuilder, VarMap};
use rand::Rng;

pub struct Trainer {
    vocab: Vocab,
    target_embeddings: Embedding,
    context_embeddings: Embedding,
    var_map: VarMap,
    optimizer: AdamW,
    window_size: usize,
    neg_samples: usize,
    device: Device,
}

impl Trainer {
    pub fn new(
        vocab: Vocab,
        embedding_dim: usize,
        window_size: usize,
        neg_samples: usize,
        learning_rate: f32,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let target_embeddings = embedding(vocab.size(), embedding_dim, vb.pp("target_embeddings"))?;
        let context_embeddings =
            embedding(vocab.size(), embedding_dim, vb.pp("context_embeddings"))?;

        let optimizer = AdamW::new_lr(var_map.all_vars(), learning_rate as f64)?;

        Ok(Self {
            vocab,
            target_embeddings,
            context_embeddings,
            var_map,
            optimizer,
            window_size,
            neg_samples,
            device,
        })
    }

    pub fn train_epoch<R: Rng>(
        &mut self,
        tokens: &[u32],
        rng: &mut R,
        _batch_size: usize,
    ) -> Result<f32> {
        if tokens.len() <= 2 * self.window_size {
            return Ok(0.0);
        }

        let mut total_loss = 0.0f32;
        let mut num_batches = 0;

        for i in self.window_size..tokens.len() - self.window_size {
            let target = tokens[i];

            let mut context = Vec::with_capacity(2 * self.window_size);
            for j in 1..=self.window_size {
                context.push(tokens[i - j]);
                context.push(tokens[i + j]);
            }

            let target_tensor = Tensor::new(&[target], &self.device)?;
            let context_tensor = Tensor::new(context.as_slice(), &self.device)?;

            let target_embed = self.target_embeddings.forward(&target_tensor)?;
            let context_embeds = self.context_embeddings.forward(&context_tensor)?;

            let context_mean = context_embeds.mean_keepdim(0)?;

            let target_vec = target_embed.to_vec2::<f32>()?;
            let context_vec = context_mean.to_vec2::<f32>()?;

            let dot = target_vec[0]
                .iter()
                .zip(context_vec[0].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();

            let dot_tensor = Tensor::new(&[dot], &self.device)?;

            let pos_loss = candle_nn::loss::binary_cross_entropy_with_logit(
                &dot_tensor,
                &Tensor::ones(1, DType::F32, &self.device)?,
            )?;

            let mut neg_loss_accum = 0.0f32;

            for _ in 0..self.neg_samples {
                let rand_idx = rng.gen_range(0..tokens.len());
                let neg_target = tokens[rand_idx];

                let neg_target_tensor = Tensor::new(&[neg_target], &self.device)?;
                let neg_target_embed = self.target_embeddings.forward(&neg_target_tensor)?;

                let neg_target_vec = neg_target_embed.to_vec2::<f32>()?;

                let neg_dot = neg_target_vec[0]
                    .iter()
                    .zip(context_vec[0].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();

                let neg_dot_tensor = Tensor::new(&[neg_dot], &self.device)?;

                let n_loss = candle_nn::loss::binary_cross_entropy_with_logit(
                    &neg_dot_tensor,
                    &Tensor::zeros(1, DType::F32, &self.device)?,
                )?;

                neg_loss_accum += n_loss.to_vec0::<f32>()?;
            }

            let neg_loss = Tensor::new(neg_loss_accum / self.neg_samples as f32, &self.device)?;

            let loss = (&pos_loss + &neg_loss)?;

            self.optimizer.backward_step(&loss)?;

            total_loss += loss.to_vec0::<f32>()?;
            num_batches += 1;
        }

        if num_batches > 0 {
            Ok(total_loss / num_batches as f32)
        } else {
            Ok(0.0)
        }
    }

    pub fn get_embeddings(&self) -> &Embedding {
        &self.target_embeddings
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.var_map.save(path)
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        self.var_map.load(path)
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

pub fn generate_training_pairs(tokens: &[u32], window_size: usize) -> Vec<(u32, Vec<u32>)> {
    let mut pairs = Vec::new();

    for i in window_size..tokens.len() - window_size {
        let target = tokens[i];
        let mut context = Vec::with_capacity(2 * window_size);

        for j in 1..=window_size {
            context.push(tokens[i - j]);
            context.push(tokens[i + j]);
        }

        pairs.push((target, context));
    }

    pairs
}

pub fn tokens_to_tensor(tokens: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(tokens, device)?.reshape((tokens.len(),))
}

pub fn cosine_similarity(embed1: &[f32], embed2: &[f32]) -> f32 {
    let dot: f32 = embed1.iter().zip(embed2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = embed1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embed2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 > 0.0 && norm2 > 0.0 {
        dot / (norm1 * norm2)
    } else {
        0.0
    }
}
