# Yelp Review Usefulness Analysis

## Overview
This project analyzes **60,000 Yelp reviews** to determine the factors that make a review “useful” to readers. Using sentiment analysis, social network metrics, and machine learning classification, the study explores how content style, review length, and a user’s social connections influence usefulness votes.

## Objectives
1. Identify patterns in **review writing style** that correlate with usefulness.
2. Determine whether **review length** impacts perceived quality.
3. Assess how a reviewer’s **social network** affects usefulness.
4. Predict usefulness using machine learning models.

## Dataset
- **Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Size:** Original dataset (~8GB), reduced to **60,000 reviews** for local processing.
- **Data Files Used:**
  - `review.json` – Contains review text, ratings, and vote counts.
  - `user.json` – Contains user profile info, including friend lists and review counts.

## Methodology

### 1. Data Preprocessing
- Merged review and user datasets using `user_id`.
- Cleaned review text using **Python `re`** for punctuation and whitespace removal.
- Created new features:
  - **Sentiment scores** using [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment).
  - **Review length** (word count).
  - **Network degree, centrality, closeness** using [NetworkX](https://networkx.org/).

### 2. Exploratory Data Analysis
- **Sentiment Distribution:** Neutral or balanced sentiment was more likely to be voted useful.
- **Length Analysis:** Longer reviews received more usefulness votes.
- **Social Network Influence:** Users with more friends tended to have more useful reviews.

### 3. Machine Learning
- **Features:** Sentiment scores, review length, user network metrics, review count, average stars.
- **Models:**
  - Random Forest Classifier
  - Logistic Regression
  - Linear SVC
- **Evaluation:** Accuracy, Precision, Recall, and F1 Score.
- **Results:** ~64% accuracy across models; Random Forest had the best recall and F1 score.

## Tools & Libraries
- **Python** – Core programming language.
- **Pandas**, **NumPy** – Data manipulation.
- **NLTK (VADER)** – Sentiment analysis.
- **NetworkX** – Social network analysis.
- **Scikit-learn** – Model training and evaluation.
- **Seaborn**, **Matplotlib** – Visualization.

## Key Findings
- **Neutral/balanced sentiment**, **longer reviews**, and **larger friend networks** increase likelihood of usefulness.
- Overly positive sentiment often correlated with lower usefulness votes.
- Predictive models performed moderately, suggesting potential for improvement with **LLMs**.

## Future Improvements
- Leverage **transformer-based models** (e.g., BERT, GPT) for richer semantic understanding.
- Use larger subsets of the Yelp dataset for training.
- Explore **ensemble models** and advanced feature engineering.

## Author
**Alip Yalikun**  
M.Eng. in Computer Science – Virginia Tech  
Email: ayalikun@vt.edu
