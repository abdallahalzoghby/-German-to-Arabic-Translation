# 🧠 German-to-Arabic Translation Quality Evaluation with BERT & BLEURT

This NLP project focuses on evaluating the quality of translated sentences from **German to Arabic** using advanced deep learning techniques including **BERT** and **BLEURT**.

---

## 🎯 Objective

To automatically assess or classify the quality of translations using semantic embeddings and scoring models. The project combines linguistic preprocessing with transformer-based evaluation models to judge how close a translation is to its source meaning.

---

## 🗃️ Dataset

- File: `e-f-dataset.csv`
- Contains sentences (likely source–translation pairs)
- Features may include original German sentence, Arabic translation, and human ratings (if available)

---

## 📌 Project Workflow

1. **Data Loading and Exploration**
   - Read dataset and analyze basic stats
   - Print sample rows to understand structure

2. **Text Preprocessing**
   - Tokenization using `nltk`
   - Stopword removal
   - Encoding labels (if classification used)

3. **Feature Extraction**
   - Use **BERT embeddings** to encode sentence meaning
   - Apply **BLEURT** for semantic similarity scoring

4. **Model Training**
   - A `Sequential` neural network model (Keras) trained on sentence embeddings
   - Classifies or scores translation quality

5. **Evaluation**
   - Metrics: Accuracy (for classification) or BLEURT scores (for similarity)
   - Optional: Use `transformers` for fine-tuned BERT model (`BertForSequenceClassification`)

---

## 🔧 Technologies Used

| Library          | Purpose                                  |
|------------------|------------------------------------------|
| `pandas`         | Data handling and preprocessing          |
| `nltk`           | Tokenization and stopword removal        |
| `transformers`   | Using BERT and tokenizer                 |
| `BLEURT`         | Scoring translation quality              |
| `Keras / TensorFlow` | Neural network classification model |
| `sklearn`        | Label encoding, model evaluation         |

---

## 💡 Sample Code Snippets

```python
from bleurt import score
scorer = score.BleurtScorer()
scores = scorer.score(references=["Das ist gut"], candidates=["هذا جيد"])
