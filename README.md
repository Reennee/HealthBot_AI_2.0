# 🏥 HealthBot AI 2.0

A domain-specific healthcare assistant built by fine-tuning a Large Language Model (LLM) using **LoRA (Low-Rank Adaptation)** for efficient, parameter-efficient training on consumer GPUs.

## 🚀 Project Overview

This project fine-tunes **TinyLlama-1.1B-Chat** on the [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) dataset to create a healthcare-focused chatbot that provides relevant medical information.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/HealthBot_AI_2.0/blob/main/Backend/HealthBot_FineTuning.ipynb)

## 📁 Project Structure

```
HealthBot_AI_2.0/
├── Backend/
│   ├── HealthBot_FineTuning.ipynb  # Complete fine-tuning pipeline (Colab-ready)
│   ├── app.py                      # FastAPI inference server
│   ├── requirements.txt            # Python dependencies
│   ├── datasets/                   # Processed training data
│   └── models/                     # Saved model adapters
├── Frontend/                       # Next.js chat interface
└── README.md
```

## 🔧 Tech Stack

- **Model**: TinyLlama-1.1B-Chat (4-bit quantized)
- **Fine-Tuning**: LoRA via HuggingFace PEFT + TRL
- **Dataset**: Medical Meadow Flashcards (~33k Q&A pairs)
- **Backend**: FastAPI
- **Frontend**: Next.js
- **Evaluation**: ROUGE, BLEU, qualitative comparison

## 🏃 Quick Start

### Google Colab (Recommended)
Click the Colab badge above to run the full fine-tuning pipeline.

### Local Setup
```bash
# Backend
cd Backend
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend
cd Frontend
npm install
npm run dev
```

## 📊 Results

*Results will be added after training (Steps 4-5).*

## 📹 Demo Video

*Link to demo video will be added here.*

---

> **Disclaimer**: This is an academic project. The chatbot is not a substitute for professional medical advice.