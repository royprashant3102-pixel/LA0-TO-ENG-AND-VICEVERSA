# 🌐 English ↔ Lao Neural Machine Translation (NMT)

A fine-tuned Neural Machine Translation system for bidirectional translation between English and Lao (ລາວ)** using MarianMT transformer models.

---

## 📌 Overview

This project fine-tunes a pre-trained MarianMT model on English–Lao sentence pairs to build a bidirectional translation system. It covers the full NMT pipeline — from data cleaning and preprocessing to fine-tuning, BLEU evaluation, and inference.

| Direction | BLEU Score |
|-----------|------------|
| Lao → English | 14.90 |
| English → Lao | 21    |

---

## 🗂️ Project Structure

```
English_Lao_NMT/
│
├── English_Lao_NMT.ipynb       # Main Colab notebook
├── README.md                   # Project documentation
└── data/                       # Sentence pair datasets (upload to Drive)
```

---

## 🚀 Pipeline Steps

| Step | Description |
|------|-------------|
| **Step 1** | Install all dependencies |
| **Step 2** | Upload dataset (sentence pairs) |
| **Step 3** | Load & inspect data |
| **Step 4** | Data cleaning & preprocessing |
| **Step 5** | Train/Validation split |
| **Step 6** | Load pre-trained MarianMT model |
| **Step 7** | Quick baseline BLEU evaluation |
| **Step 8** | Fine-tune model on Lao–English pairs |
| **Step 9** | Evaluate BLEU scores (both directions) |
| **Step 10** | Plot training curves |
| **Step 11** | Inference — translate new sentences |
| **Step 12** | Save & download fine-tuned model |

---

## 🧠 Model Details

- **Base Model:** [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP)
- **Architecture:** MarianMT (Transformer-based seq2seq)
- **Framework:** PyTorch + HuggingFace Transformers
- **Language Pair:** English ↔ Lao (`en` ↔ `lo`)

---

## ⚙️ Decoding Configuration

```python
model.generate(
    **inputs,
    num_beams=8,
    max_length=256,
    min_length=5,
    early_stopping=False,
    no_repeat_ngram_size=3,
    length_penalty=1.5
)
```

---

## 📦 Requirements

```bash
pip install transformers
pip install sacrebleu
pip install sentencepiece
pip install torch
pip install datasets
```

Or run **Step 1** of the notebook which installs all dependencies automatically.

---

## 🏃 How to Run

1. Open the notebook in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p_vXpzYbtemtcKZko-PuwYmRGMyOalWy)

2. Upload your English–Lao sentence pairs dataset

3. Run all cells sequentially (**Runtime → Run all**)

4. Download the fine-tuned model from **Step 12**

---

## 📊 BLEU Evaluation

BLEU scores are computed using [SacreBLEU](https://github.com/mjpost/sacrebleu):

```python
from sacrebleu.metrics import BLEU
bleu = BLEU()
score = bleu.corpus_score(predictions, [references])
```

---

## 📈 Results

```
Lao → English BLEU = 14.90
  28.8/15.8/11.8/9.2
  BP = 1.000  |  ratio = 1.047
  hyp_len = 3090  |  ref_len = 2952
```

---

## 🔧 Tips to Further Improve BLEU

- Increase `num_beams` to 12 for better beam search
- Use `no_repeat_ngram_size=4` to reduce repetition
- Add `repetition_penalty=1.3` to penalize repeated tokens
- Experiment with `length_penalty=0.8` for shorter, more accurate outputs
- Train for more epochs with a lower learning rate
- Use back-translation for data augmentation

---

## 📝 Dataset

The model is trained on parallel English–Lao sentence pairs. You can find open Lao NLP datasets at:
- [FLORES-200](https://github.com/facebookresearch/flores)
- [ALT Corpus](http://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/)
- [Opus Corpora](https://opus.nlpl.eu/)

---

## 👤 Author

**Prashant Roy**  
GitHub: [@royprashant3102-pixel](https://github.com/royprashant3102-pixel)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for pre-trained MarianMT models
- [SacreBLEU](https://github.com/mjpost/sacrebleu) for BLEU evaluation
