# SteamSense: AI-Powered Steam Review Analysis

An NLP-based pipeline that transforms large volumes of Steam player reviews into sentiment insights and actionable recommendations for game developers.

<p align="center">
  <img src="assets/pipeline.svg" alt="SteamSense five-stage data pipeline" width="100%" />
</p>

## Overview

SteamSense helps developers understand what players like and dislike without manually reading thousands of reviews. It combines sentiment classification with topic modeling to identify both overall player opinion and the reasons behind it.

## Business Value

- **Faster feedback analysis:** reduces the effort required to process large volumes of player reviews.
- **Data-driven updates:** highlights recurring complaints and positive feedback to support patch and product decisions.
- **Accessible market intelligence:** gives smaller studios a structured way to learn from player feedback.

## Dataset

The project uses the [Steam Reviews 2021 dataset on Kaggle](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021), which contains approximately 21 million reviews.

A stratified sample of 100,000 English-language reviews was selected. After empty reviews were removed, the report states that the processed dataset contained 99,816 entries.

<p align="center">
  <img src="assets/sentiment-distribution.svg" alt="Positive and negative sentiment distribution in the sampled reviews" width="90%" />
</p>

## Data Pipeline

1. **Data ingestion:** English-language reviews are filtered and a random sample of 100,000 is selected.
2. **Text preprocessing:** URLs, HTML and Steam BBCode tags are removed; text is lowercased, tokenized with NLTK, filtered for stopwords, and lemmatized. Sentiment-bearing words are retained.
3. **Sentiment classification:** the pipeline compares VADER, TF-IDF-based Multinomial Naive Bayes and Logistic Regression, and fine-tuned BERT (`bert-base-uncased`). BERT is fine-tuned for one epoch on a stratified 10,000-review sample using a T4 GPU.
4. **Topic modeling:** separate LDA models extract themes from positive and negative reviews. Positive reviews cover gameplay, core loop, graphics and story; negative reviews focus on bugs, crashes, optimization and price.
5. **Evaluation:** models are assessed using accuracy, precision, recall and F1-score with a stratified train/test split.

## Topic Modeling Insights

The LDA analysis extracts recurring terms from positive and negative reviews, helping developers move beyond a simple positive/negative label.

<p align="center">
  <img src="assets/topic-modeling.png" alt="Most frequent words in positive and negative Steam reviews" width="100%" />
</p>

## Model Performance

Reported results from the evaluation table:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VADER (baseline) | 0.70 | 0.98 | 0.71 | 0.82 |
| Logistic Regression + TF-IDF | 0.90 | 0.98 | 0.91 | 0.94 |
| Fine-tuned BERT | 0.96 | 0.98 | 1.00 | 0.99* |

<p align="center">
  <img src="assets/model-performance.png" alt="Comparison of accuracy, precision, recall and F1-score across the three models" width="100%" />
</p>

The report's conclusion gives a different BERT F1-score (0.9079); the value in the table and chart above is taken from its evaluation section.

### Confusion Matrices

<p align="center">
  <img src="assets/confusion-matrices.png" alt="Confusion matrices for VADER, Logistic Regression and fine-tuned BERT" width="100%" />
</p>

## Key Recommendations

- Use **Logistic Regression** for online sentiment analysis where lower computation time is important.
- Run **BERT** periodically for deeper batch analysis.
- Use LDA topics to prioritize recurring technical and optimization complaints in patches and updates.
- Consider class-weight balancing or SMOTE to address the imbalance between positive and negative reviews.

## Tools & Methods

NLTK · TF-IDF · VADER · Multinomial Naive Bayes · Logistic Regression · BERT · Gensim (LDA) · Google Colab
