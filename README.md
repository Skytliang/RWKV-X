# RWKV-X 🚀

**RWKV-X** is a Linear Complexity Hybrid Language Model based on the RWKV architecture, integrating **Sparse Attention** to enhance long-sequence processing capabilities. 📚⚡

---

## 🧠 Project Overview

RWKV-X is an extended version of the RWKV-7 language model, introducing several improvements while maintaining RWKV's efficient, RNN-like characteristics. This project includes complete implementations for **pre-training**, **fine-tuning**, **evaluation**, and **packaging for pip**. 🔧📦

---

## ✨ Key Features

- **🧵 Long Sequence Support**: Innovative architecture supporting context lengths up to **64K**
- **🧠 Sparse Attention Mechanism**: Efficient sparse attention to enhance long-text understanding
- **🛠️ Comprehensive Toolchain**: End-to-end workflow from pre-training to fine-tuning to evaluation

---

## 🗂️ Project Structure

```
RWKV-X/
├── sft/                  # Supervised fine-tuning module
│   ├── src/              # Fine-tuning source code
│   ├── tokenizer/        # Tokenizer
│   ├── scripts/          # Fine-tuning scripts
│   └── train.py          # Main fine-tuning script
├── pretrain/             # Pre-training module
│   ├── src/              # Pre-training source code
│   ├── tokenizer/        # Tokenizer
│   └── train.py          # Main pre-training script
├── package/              # Python package module
│   ├── src/              
│   └── rwkv_x/           # Core implementation of RWKV-X
├── evaluation/           # Evaluation module
│   ├── src/              # Evaluation source code
│   ├── lm_eval/          # Language model evaluation
│   └── postprocess/      # Post-processing tools
└── tree.py               # Directory structure generation script
```

---

## ⚙️ Usage

### 📦 Installation

```bash
# Install from PyPI
pip install rwkv-x

# Or install from source
cd RWKV-X/package
pip install -e .
```

---

### 🧪 Inference Example

```python
# !!! set these before import RWKV !!!
import os
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv_x.model import RWKV_X
from rwkv_x.utils import PIPELINE, PIPELINE_ARGS

# Load model
# You can get the model weights from: https://huggingface.co/howard-hou/RWKV-X/
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

---

### 🔧 Fine-tuning the Model

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

---

### 📊 Model Evaluation

```bash
cd RWKV-X/evaluation
python lm_eval_rwkvx_pip.py path/to/model.pth --task_group english
```

---

## 📈 Performance Benchmarks

RWKV-X delivers outstanding results on multiple standard benchmarks:

- 🏆 Classification & Reading: MMLU, LAMBADA, HellaSwag
- 🧾 Long Text Understanding: RULER, LongBench
- 🌍 Multilingual Tasks

---

## 📄 License

RWKV-X is released under the **MIT License**. See the `LICENSE` file for details. ✅
