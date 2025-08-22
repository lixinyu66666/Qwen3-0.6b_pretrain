# Qwen3-0.6b Pretraining

A large language model pretraining framework ### 3. Data Preparation

Use the multi-corpus data processing script with the trained tokenizer:

```bash
# Process multiple corpora and generate training data
python scripts/prepare_multi_corpus.py \
    --config configs/data/multi_corpus.yaml \
    --tokenizer_path tokenizers/qwen3_tokenizer \
    --output_dir data/processed
```

### 4. Train Modelrch and Hugging Face Transformers, supporting distributed training, mixed precision training, and efficient data processing.

## Project Overview

This project implements a complete language model pretraining pipeline with the following core features:

- **Multiple Training Modes**: Support for debug mode training and distributed accelerated training
- **Efficient Data Processing**: Multi-corpus streaming processing with Arrow format storage
- **Performance Optimization**: Mixed precision training, gradient checkpointing, DeepSpeed integration
- **Evaluation Benchmarks**: Perplexity calculation and generation performance benchmarking
- **Flexible Configuration**: Modular configuration system supporting different model scales

## Project Structure

```
Qwen3-0.6b_pretrain/
├── scripts/           # Training and evaluation scripts
├── configs/           # Configuration files
├── src/              # Core source code
├── data/             # Dataset directory
├── environment.yml   # Conda environment configuration
└── README.md         # Project documentation
```

## Environment Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate qwen3pretrain
```

### 2. Install Dependencies

The environment includes the following main dependencies:
- **PyTorch 2.5.0** (CUDA 12.1 support)
- **Transformers** (Model architecture and pretraining modules)
- **Accelerate** (Distributed training framework)
- **DeepSpeed** (Memory optimization and distributed training)
- **Datasets** (Data processing)
- **tiktoken/sentencepiece** (Tokenizers)

## Quick Start

### 1. Prepare Corpus

First, prepare the corpus from raw data sources:

```bash
# Prepare temporary corpus from various data sources
python scripts/prepare_tmp_corpus.py \
    --out data/raw/tmp_corpus.txt \
```

### 2. Train Tokenizer

Train a tokenizer on the prepared corpus:

```bash
# Train tokenizer from prepared corpus
python scripts/train_tokenizer.py \
    --input data/raw/tmp_corpus.txt \
    --vocab_size 32000 \
    --output_dir tokenizer
```

### 3. Data Preparation

Use the multi-corpus data processing script with the trained tokenizer:

```bash
# Process multiple corpora and generate training data
python scripts/prepare_multi_corpus.py \
    --target_tokens 9e9 \
    --shard_tokens 1e8 \
    --out_dir data/processed \
    --tokenizer_dir tokenizer \
    --mix "wiki_en=0.55,rpjv2_small_en=0.2,wiki_zh=0.15,wiki_tw=0.05,stackoverflow_en=0.05"
```

```bash
# Packed training data
python scripts/make_packed_dataset.py \
    --arrow_dir data/processed/rpjv2_3B_arrow \
    --out_dir   data/packed/rpjv2_s2048 \
    --seq_len   2048 \
    --shard_tokens 50000000
```

### 4. Train Model

#### Distributed Training (Multi-GPU)

```bash
# Launch distributed training with Accelerate
scripts/run_deepspeed.sh
```

### 5. Plot training curve

```bash
python scripts/plot_loss.py \
    --log logs/train_qwen3_0.6b.log \
    --csv results/loss.csv
```


### 6. Model Evaluation
#### Quick inference

```bash
python scripts/quick_infer.py \
    --ckpt checkpoints/qwen3_0p6b/step-00030600 \
    --tokenizer tokenizer \
    --prompt "Write a short poem about stars." \
    --max_new_tokens 128
```

#### Perplexity Evaluation

```bash
python scripts/eval_ppl.py \
    --ckpt checkpoints/qwen3_0p6b/step-00030600 \
    --dataset wikitext \
    --subset wikitext-2-raw-v1 \
```

#### Generate sanity

```bash
python scripts/sanity_generate.py \
    --ckpt checkpoints/qwen3_0p6b/step-00030600 \
    --out results/sanity_outputs.jsonl
```

#### Inference performance and video memory monitoring
```bash
python scripts/bench_generate.py \
    --ckpt checkpoints/qwen3_0p6b/step-00030600 \
    --prompt "Hello, world!"
```

## Core Components

### Training Scripts

| Script | Purpose | Features |
|--------|---------|----------|
| `train_debug.py` | Debug training | Manual mixed precision, detailed logging |
| `train_accel.py` | Distributed training | Accelerate framework, automatic device management |

### Data Processing

- **streaming_dataset.py**: Streaming dataset processing, supporting large-scale corpora
- **packed_ds.py**: Load the packaged NPZ file
- **prepare_multi_corpus.py**: Multi-corpus preprocessing and token budget control

### Evaluation Tools

- **eval_ppl.py**: Calculate model perplexity on test sets.
- **bench_generate.py**: Monitor the inference performance and video memory.
- **sanity_generate.py**: Check whether the output is coherent and grammatically correct. Assess whether there is frequent repetition (a higher distinct-1/2 score indicates better diversity). Examine whether there is any abnormal punctuation or garbled text, especially when mixing Chinese and English.

### Configuration System

```
configs/
├── model/           # Model architecture configuration
├── train/           # Training parameter configuration  
├── deepspeed/       # DeepSpeed optimization configuration
└── accelerate/      # Accelerate distributed configuration
```

## Configuration Examples

### Model Configuration Example (model_debug.json)

```json
{
    "model_type": "llama",
    "vocab_size": 32000,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 2048
}
```

### Training Configuration Example (debug.yaml)

```yaml
# Data configuration
data_dir: "data/processed"
train_micro_batch_size_per_gpu: 4
gradient_accumulation_steps: 8

# Optimizer configuration
learning_rate: 2e-4
weight_decay: 0.01
betas: [0.9, 0.95]

# Training configuration
max_steps: 10000
warmup_steps: 1000
bf16: true
gradient_checkpointing: true

# Save and evaluation
save_interval: 1000
eval_interval: 500
logging_interval: 10
```

## Performance Optimization

### Memory Optimization Strategies

1. **Mixed Precision Training**: Use BF16 to reduce memory usage
2. **Gradient Checkpointing**: Trade computation for memory
3. **DeepSpeed ZeRO**: Distributed memory optimization
4. **Gradient Accumulation**: Simulate large batch training on small batches

### Training Acceleration Techniques

1. **TF32 Optimization**: Automatically enable CUDA matrix operation acceleration
2. **DataLoader Optimization**: Multi-process data loading and prefetching
3. **Compilation Optimization**: Use torch.compile for model acceleration
4. **Efficient Tokenization**: Preprocess data to avoid online tokenization

## Monitoring and Debugging

### Training Monitoring

- Real-time loss curves and learning rate scheduling
- GPU memory usage monitoring
- Training throughput statistics
- Gradient norm checking

### Debugging Features

- Detailed training log output
- Model weights and gradient inspection
- Data loading performance analysis
- Memory leak detection

## FAQ

### Q: How to adjust model size?
A: Modify configuration files in `configs/model/`, main parameters include `hidden_size`, `num_hidden_layers`, `num_attention_heads`.

### Q: How to handle OOM issues?
A: Try the following strategies:
1. Reduce `train_micro_batch_size_per_gpu`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Use DeepSpeed ZeRO

### Q: How to add new datasets?
A: Add new data source configurations in `scripts/prepare_multi_corpus.py`, supporting JSONL, Parquet and other formats.

## Contributing

1. Fork the project repository
2. Create a feature branch
3. Commit code changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or suggestions, please submit an Issue or contact the project maintainers.