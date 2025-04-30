# RWKV-X

RWKV-X is a Linear Complexity Hybrid Language Model based on the RWKV architecture, integrating Sparse Attention to improve the model's long sequence processing capabilities.

## Project Overview

RWKV-X is an extended version of the RWKV-7 language model, introducing several improvements while maintaining RWKV's efficient RNN-like characteristics. This project includes complete code implementations for pre-training, fine-tuning, evaluation, and packaging for pip.

## Key Features

- **Long Sequence Support**: Innovative architecture design supporting context lengths up to 64K
- **Sparse Attention Mechanism**: Efficient sparse attention mechanism to enhance long text understanding
- **Comprehensive Toolchain**: Complete workflow from pre-training to fine-tuning to evaluation

## Project Structure

```
RWKV-X/
├── sft/                  # Supervised fine-tuning module
│   ├── src/              # Fine-tuning source code
│   ├── tokenizer/        # Tokenizer
│   ├── scripts/          # Fine-tuning scripts
│   └── train.py          # Fine-tuning main program
├── pretrain/             # Pre-training module
│   ├── src/              # Pre-training source code
│   ├── tokenizer/        # Tokenizer
│   └── train.py          # Pre-training main program
├── package/              # Python package distribution module
│   ├── src/              # Package source code
│   └── rwkv_x/           # RWKV-X core implementation
├── evaluation/           # Evaluation module
│   ├── src/              # Evaluation source code
│   ├── lm_eval/          # Language model evaluation
│   └── postprocess/      # Post-processing tools
└── tree.py               # Directory structure generation script
```

## Usage

### Installation
```bash
# Install from pypi
pip install rwkv-x
# Install from source
cd RWKV-X/package
pip install -e .
```

### Inference Example

```python
from rwkv_x.model import RWKV_X
from rwkv_x.utils import PIPELINE, PIPELINE_ARGS

# Load model
# you can get the model weights from https://huggingface.co/howard-hou/RWKV-X/
model = RWKV_X(model_path='RWKV-X-0.2B-64k-Base.pth', strategy='cuda fp16')
pipeline = PIPELINE(model)

# Set generation parameters
args = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.7,
    top_k=100,
    alpha_frequency=0.25,
    alpha_presence=0.25,
    token_ban=[],
    token_stop=[],
    chunk_len=256
)

# Generate text
ctx = "This is a sample prompt."
output = pipeline.generate(ctx, token_count=200, args=args)
print(output)
```

### Fine-tuning the Model

```bash
cd RWKV-X/sft
python train.py --load_model ../RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth \
    --wandb "rwkv1b5-sft" --proj_dir out/rwkv1b5-sft \
    --data_file ../data.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 10 \
    --micro_bsz 32 --accumulate_grad_batches 4 \
    --lr_init 6e-5 --lr_final 1.5e-5 \
    --accelerator gpu --devices 2 --precision bf16
```

### Model Evaluation

```bash
cd RWKV-X/evaluation
python lm_eval_rwkvx_pip.py path/to/model.pth --task_group english
```

## Performance Benchmarks

RWKV-X performs excellently on multiple standard evaluation benchmarks:

- Classification and reading comprehension tasks such as MMLU, LAMBADA, HellaSwag
- Long text processing capability evaluation (RULER, LongBench)
- Multilingual capability evaluation

## License

RWKV-X is released under the MIT License. See the LICENSE file for details.
