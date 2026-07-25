# NLP-Consumer-Complaint-Classifier
# 🧠 Consumer Complaint Classification using Deep Learning & DistilBERT

An end-to-end Natural Language Processing (NLP) project that classifies consumer complaints into predefined categories using Deep Learning and Transformer-based models.

The project compares the performance of **SimpleRNN, LSTM, GRU, and DistilBERT**, then deploys the best-performing model with **Streamlit**.

---

# 📌 Project Overview

Consumer complaint classification is an important NLP task that helps financial institutions automatically route and analyze customer complaints.

In this project, different neural network architectures were implemented and compared to determine the most effective model for complaint classification.

---

# 📊 Dataset

**Dataset:** Consumer Financial Protection Bureau (CFPB) Consumer Complaint Dataset

The dataset contains consumer complaints related to financial products and services.

### Categories

- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages & Loans
- Retail Banking

---

# 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Remove missing values
- Remove duplicate records
- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Remove numbers
- Remove extra spaces
- Tokenization
- Padding sequences (Deep Learning models)
- Label Encoding
- Train/Test Split

---

# 🤖 Models Implemented

The project compares four different NLP models.

## 1️⃣ SimpleRNN

A baseline recurrent neural network model.

---

## 2️⃣ LSTM

Captures long-term dependencies better than traditional RNN.

---

## 3️⃣ GRU

A lightweight alternative to LSTM with fewer parameters and faster training.

---

## 4️⃣ DistilBERT

A pretrained Transformer model from Hugging Face.

It provides contextual understanding of language while being smaller and faster than BERT.

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|--------:|---------:|
| 🥇 DistilBERT | **89.52%** | **89.02%** | **89.52%** | **89.04%** |
| 🥈 GRU | **88.24%** | **88.08%** | **88.24%** | **88.10%** |
| 🥉 LSTM | **86.79%** | **87.67%** | **86.79%** | **87.05%** |
| SimpleRNN | **69.02%** | **73.73%** | **69.02%** | **70.29%** |

---

# 🏆 Best Model

**DistilBERT** achieved the best performance.

✔ Highest Accuracy

✔ Highest F1 Score

✔ Better contextual understanding

Therefore, DistilBERT was selected for deployment.

---

# 🌐 Deployment

The application was deployed using **Streamlit**.

Users can:

- Enter a complaint
- Click Predict
- Receive the predicted complaint category instantly

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Hugging Face Transformers
- PyTorch
- Streamlit
- Matplotlib
- Seaborn

---


<img width="1536" height="1024" alt="img" src="https://github.com/user-attachments/assets/e8c644f2-a3cd-438d-9021-69f5e7b00a00" />




---

⭐ If you found this project useful, consider giving it a star!
