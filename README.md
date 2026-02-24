# 🛡️ NLP-Spam-Classifier-Project

> An end-to-end NLP project that classifies SMS/Email messages as Spam or Ham. Built with Python, NLTK &amp; Scikit-learn using Tokenization, Stemming, TF-IDF Vectorization and Multinomial Naive Bayes. Achieves 97% accuracy &amp; 100% precision on 5,572 messages. Deployed as an interactive web app using Streamlit.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [NLP Pipeline](#nlp-pipeline)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Results](#results)

---

## 📖 Overview

This is a complete end-to-end machine learning project that detects whether an SMS or Email message is **Spam** or **Ham (Not Spam)**. The project covers the entire ML workflow — from raw data cleaning and exploratory data analysis, to text preprocessing, model training, model comparison, and deployment as a web application.

The model was trained on the **SMS Spam Collection Dataset** containing 5,572 real-world messages and uses a **Multinomial Naive Bayes** classifier with **TF-IDF Vectorization** to achieve state-of-the-art results.

---

## 🎯 Features

- ✅ Real-time spam detection with confidence score
- ✅ Complete NLP preprocessing pipeline
- ✅ Comparison of 10+ ML algorithms
- ✅ Interactive dark-themed Streamlit web app
- ✅ Prediction history tracking
- ✅ Example spam and ham messages to test
- ✅ Word cloud visualizations
- ✅ Live character and word counter

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Name | SMS Spam Collection Dataset |
| Source | [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Total Messages | 5,572 |
| Spam Messages | 747 (13.4%) |
| Ham Messages | 4,825 (86.6%) |
| Features | Message text |
| Target | spam / ham |

> The dataset is imbalanced — 86.6% ham vs 13.4% spam. This makes **precision** a more important metric than accuracy to avoid false positives.

---
