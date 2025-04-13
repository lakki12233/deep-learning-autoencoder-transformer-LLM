
# Deep Learning Assignment (CSE 676-B, Spring 2025)

This repository contains the complete implementation of Assignment 2 for the course **Deep Learning (CSE 676-B)** at University at Buffalo, Spring 2025. The assignment explores practical and theoretical aspects of Autoencoders, Transformers, Large Language Models (LLMs), and Vision Transformers (ViT).

---

## 📁 Contents

### Part I: Theoretical Analysis
- 📄 `a2_part_1_dshrisai_sumanman.pdf`  
  Includes:
  - Trainable parameter count in Autoencoders
  - L2 regularization backprop derivation
  - Self-attention mathematical explanation
  - Computational graphs and challenges

---

### Part II: Autoencoders for Anomaly Detection
- 📓 `a2_part_2_dshrisai_sumanman.ipynb`  
  - Dataset: NAB (Numenta Anomaly Benchmark)
  - Models: Dense, LSTM, and Conv1D Autoencoders
  - Evaluation: Achieved >80% accuracy
  - Metrics: MSE, Precision, Recall, F1-Score
  - Visualization: Reconstruction error analysis
  - Saved weights provided via UBBox

---

### Part III: Transformer from Scratch (PyTorch)
- 📓 `a2_part_3_dshrisai_sumanman.ipynb`  
  - Complete PyTorch implementation of Transformer encoder
  - Dataset analysis, tokenization, embeddings, multi-head self-attention
  - Evaluation with accuracy/loss plots
  - Regularization and optimization techniques applied

---

### Part IV: LLM Summarization with BART
- 📓 `a2_part_4_sumanman_dshrisai.ipynb`  
  - Model: `facebook/bart-base` from HuggingFace
  - Dataset: BillSum or Multi-News
  - Evaluation: ROUGE, BLEU, BERTScore
  - Performance > threshold (ROUGE-1 > 40, BLEU > 12, BERTScore > 75)

---

### 🎁 Bonus: ViT and LLM Probing
- 📓 `a2_bonus_vit_dshrisai_sumanman.ipynb`  
  - Vision Transformer applied to Cats vs Dogs Dataset
  - Includes model training and evaluation
  - Deployment video included in UBBox

- 📓 `a2_bonus_classification_dshrisai_sumanman.ipynb`  
  - DistilBERT and TinyBERT used for spam classification
  - Achieved accuracy and F1 > 85%
  - Frozen base model + classifier head

---

## 🧠 Technologies Used
- Python, PyTorch
- HuggingFace Transformers
- Scikit-learn, matplotlib, seaborn
- Torchinfo for model summaries
- TensorBoard / WandB (optional)

---

## 📦 File Overview

| File Name | Description |
|-----------|-------------|
| `a2_part_1_dshrisai_sumanman.pdf` | Theoretical analysis |
| `a2_part_2_dshrisai_sumanman.ipynb` | Autoencoders for anomaly detection |
| `a2_part_3_dshrisai_sumanman.ipynb` | Transformer from scratch |
| `a2_part_4_sumanman_dshrisai.ipynb` | BART summarization |
| `a2_bonus_vit_dshrisai_sumanman.ipynb` | ViT classification |
| `a2_bonus_classification_dshrisai_sumanman.ipynb` | LLM linear probing |
| `a2_weights_dshrisai_sumanman..txt` | Link to saved weights on UBBox |

---

## 👩‍💻 Authors
- **Suman Mandava**
- **Sai D Shrivatsav**

---

## 📜 License
This is a university course assignment. The code is for academic and educational purposes only.
