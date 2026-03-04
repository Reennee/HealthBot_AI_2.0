# HealthBot AI 2.0

A domain-specific **healthcare chatbot** built by fine-tuning **TinyLlama-1.1B-Chat** on a curated medical Q&A dataset using **LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning method that enables training on free-tier Google Colab GPUs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reennee/HealthBot_AI_2.0/blob/main/Healthbot-Finetuning.ipynb)

> **Demo Video**: *https://drive.google.com/file/d/1eqxEdM_px920NlpG42GQrXLtFYCk2E5C/view?usp=sharing*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Fine-Tuning Methodology](#fine-tuning-methodology)
6. [Hyperparameter Experiments](#hyperparameter-experiments)
7. [Performance Metrics](#performance-metrics)
8. [Example Conversations](#example-conversations)
9. [How to Run](#how-to-run)
10. [Tech Stack](#tech-stack)

---

## Project Overview

HealthBot AI 2.0 fine-tunes a generative language model for the **healthcare domain** so it can accurately answer medical questions about symptoms, diagnoses, treatments, pharmacology, and physiology. The project uses:

- **Generative QA approach** — the model generates free-text answers rather than extracting spans
- **LoRA (PEFT)** — only 1.13% of parameters are trained, making fine-tuning feasible on a T4 GPU
- **4-bit quantization** — reduces the 1.1B parameter model from ~4.4 GB to ~1.1 GB in memory
- **Gradio interface** — users can interact with the model through a browser-based chat UI

---

## Project Structure

```
HealthBot_AI_2.0/
│   ├── HealthBot_FineTuning.ipynb  # Complete fine-tuning pipeline (Colab-ready)
│   ├── requirements.txt            # Python dependencies
│   └── healthbot_finetuned/        # LoRA adapter weights (post-training)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md
```

---

## Dataset

**Source**: [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

This dataset contains medical flashcard-style Q&A pairs covering a wide range of clinical topics including pharmacology, physiology, pathology, anatomy, and clinical medicine.

| Statistic | Value |
|-----------|-------|
| Total available samples | 33,955 |
| Training samples used | 2,000 |
| Validation samples used | 200 |
| Evaluation samples (ROUGE/BLEU) | 50 |
| Format | Question / Answer pairs |
| Median question length | ~72 characters |
| Median answer length | ~112 characters |

**Why 2,000 samples?** A subset was used to train efficiently within Colab's free GPU quota (~12 min training vs ~60 min for the full dataset), while still achieving strong evaluation scores. The dataset is diverse, covering pharmacology, physiology, diagnostics, and clinical medicine — ensuring broad generalization.

**Sample entry:**
```json
{
  "input": "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?",
  "output": "Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.",
  "instruction": "Answer this question truthfully"
}
```

---

## Data Preprocessing

Each raw Q&A pair goes through the following preprocessing pipeline:

### 1. Normalization
Strip whitespace from questions and answers; filter out any empty samples.

### 2. Instruction-Response Template Formatting
Each sample is wrapped in TinyLlama's chat template with a domain-specific system prompt:

```
<|system|>
You are HealthBot AI, a specialized medical assistant. You ONLY answer questions
related to health, medicine, anatomy, symptoms, treatments, medications, and medical
concepts. Always recommend consulting a healthcare professional for personalized advice.
<|user|>
{question}
<|assistant|>
{answer}
```

### 3. Tokenization & Length Management
- Tokenized using TinyLlama's SentencePiece tokenizer
- Sequences truncated to `max_length=512` tokens to fit within the model's context window
- Padding token set to EOS token (`</s>`) with right-side padding

### 4. Train / Validation Split
- 90% train / 10% validation using a fixed `seed=42` for reproducibility
- Final split: **2,000 train** / **200 validation**

---

## Fine-Tuning Methodology

### Model Selection
**TinyLlama-1.1B-Chat-v1.0** was selected because:
- Small enough to train on a free Colab T4 GPU (15 GB VRAM)
- Pre-trained on chat/instruction data, making it well-suited for Q&A fine-tuning
- Strong baseline performance relative to its size

### Quantization
4-bit NF4 quantization via **BitsAndBytes** reduces memory usage while preserving model quality:

| Setting | Value | Purpose |
|---------|-------|---------|
| `load_in_4bit` | True | Compress weights from 16-bit to 4-bit |
| `bnb_4bit_quant_type` | `nf4` | Normal Float 4 — optimal for LLM weights |
| `bnb_4bit_use_double_quant` | True | Quantize the quantization constants too |
| `bnb_4bit_compute_dtype` | `float16` | Fast matrix multiply on T4 |

### LoRA (Parameter-Efficient Fine-Tuning)
LoRA injects trainable low-rank decomposition matrices into the model's layers while keeping the original weights frozen. This reduces trainable parameters from **1.1B → 12.6M (1.13%)**.

| LoRA Setting | Value | Reason |
|---|---|---|
| `r` (rank) | 16 | Balances expressiveness vs. memory |
| `lora_alpha` | 32 | Scaling factor; effective scale = alpha/r = 2.0 |
| `lora_dropout` | 0.05 | Light regularization |
| `target_modules` | all attention + MLP projections | Maximum domain adaptation |

**Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Training Setup

| Setting | Value |
|---------|-------|
| Trainer | HuggingFace TRL `SFTTrainer` |
| Optimizer | `paged_adamw_8bit` (memory-efficient) |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine decay |
| Warmup ratio | 0.03 |
| Max gradient norm | 0.3 |
| Mixed precision | Disabled (fp16=False, bf16=False) |

---

## Hyperparameter Experiments

The following experiments were conducted to identify the optimal configuration. GPU memory and training time were tracked for each run.

| Exp | LR | Batch | Grad Accum | Epochs | LoRA r | Train Loss | Val Loss | ROUGE-L | GPU Mem | Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2e-4 | 2 | 8 | 1 | 16 | ~1.42 | ~1.48 | 0.21 | ~5 GB | ~4 min |
| 2 | 1e-4 | 2 | 8 | 2 | 16 | ~1.28 | ~1.31 | 0.26 | ~5 GB | ~8 min |
| 3 | 2e-4 | 4 | 4 | 3 | 32 | ~1.20 | ~1.25 | 0.27 | ~8 GB | ~14 min |
| **4 ✓** | **2e-4** | **4** | **4** | **3** | **16** | **~1.18** | **~1.22** | **0.35** | **~7 GB** | **~12 min** |

**Key findings:**
- Increasing epochs from 1 → 3 gave the biggest improvement in ROUGE-L (+0.14)
- Larger batch (4 vs 2) with proportionally reduced gradient accumulation improved convergence stability
- Higher LoRA rank (r=32) did not improve over r=16 and used more memory
- Learning rate 2e-4 consistently outperformed 1e-4 and 5e-5 for this dataset size

---

## Performance Metrics

Evaluation was performed on 50 held-out validation samples using the HuggingFace `evaluate` library.

### Automatic Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **ROUGE-1** | **0.4242** | Unigram overlap with reference answers |
| **ROUGE-2** | **0.2706** | Bigram overlap with reference answers |
| **ROUGE-L** | **0.3518** | Longest common subsequence overlap |
| **BLEU** | **0.1281** | Precision-based n-gram overlap |

These scores indicate strong lexical alignment between model outputs and reference medical answers, reflecting successful domain adaptation.

### Comparison: Base Model vs Fine-Tuned

The base TinyLlama model was not benchmarked quantitatively (CPU-only inference was prohibitively slow). Qualitatively, the fine-tuned model shows clear improvement:

| Aspect | Base Model | Fine-Tuned |
|--------|-----------|------------|
| Medical terminology | Generic / absent | Accurate (polyuria, polydipsia, etc.) |
| Answer structure | Verbose, off-topic | Concise, clinically focused |
| Domain focus | General | Healthcare-specific |
| Disclaimer inclusion | None | Recommends professional consultation |
| Dataset-specific facts | Absent | Present (e.g., Mg2+/PTH/Ca2+ relationship) |

---

## How to Run

1. Click the **Open in Colab** badge at the top of this README
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells top-to-bottom — training takes ~12 minutes
4. The **Gradio chat interface** launches automatically in Step 10 with a public link you can open in any browser and interact with directly

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base LLM | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning | HuggingFace PEFT (LoRA) + TRL (SFTTrainer) |
| Quantization | BitsAndBytes 4-bit NF4 |
| Dataset | HuggingFace `datasets` |
| Evaluation | HuggingFace `evaluate` (ROUGE, BLEU) |
| Chat Interface | Gradio |
| Training Platform | Google Colab (T4 GPU) |

---

> **Disclaimer**: HealthBot AI is an academic project for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
