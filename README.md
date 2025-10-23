# 🛍️ Amazon Product Review Sentiment Analysis

## 📘 Overview
This project performs **Sentiment Analysis** on Amazon product reviews to determine whether a review expresses a **positive**, **negative**, or **neutral** sentiment.  
We collected, preprocessed, analyzed, and modeled product reviews to build a machine learning model that predicts sentiment automatically.

---

## 🚀 Implementation Steps

### 1. Data Collection
We collected product review data directly from the **Amazon website** using **web scraping**.  
Since labeling was done manually, the dataset was trimmed to a smaller but high-quality subset.

**Steps:**
- Scraped review data for multiple products.
- Cleaned raw text and removed duplicates.
- Manually labeled data for supervised learning.

---

### 2. Exploratory Data Analysis (EDA)
**EDA (Exploratory Data Analysis)** helps understand data distribution and detect anomalies before modeling.

#### Goals of EDA:
- Identify errors or inconsistencies in data.
- Detect outliers or anomalous patterns.
- Understand variable relationships.
- Confirm data is ready for modeling.

#### Common Insights:
- Distribution of sentiment polarity.
- Frequency of positive vs negative reviews.
- Common keywords in each sentiment class.

**Example of summary statistics using pandas:**
Generated Metrics:

count: Total number of reviews analyzed.

min: Most negative sentiment polarity (≈ -1).

max: Most positive sentiment polarity (≈ +1).

mean: Average polarity across all reviews — typically positive.

### 3. Data Preprocessing

Before modeling, we standardized the text data for consistent analysis.

🔠 Lower Casing

Convert all text to lowercase so words like "Text" and "text" are treated the same.

❌ Removal of Punctuations

Removes punctuation marks (! ? , .) to ensure uniform word tokens.

🧹 Removal of Stopwords

Eliminates frequently occurring words such as "the", "a", "is", which add little semantic value.

⚖️ Removal of Frequent and Rare Words

Removes overly frequent or rare terms that don’t contribute to the model’s performance.
(Note: If TF-IDF is used, this step is inherently handled.)

### 4. Data Modeling

We applied several machine learning algorithms to classify sentiments:

Algorithm	Description	Accuracy
Random Forest	Ensemble-based model combining multiple decision trees.	78%
SVM (Support Vector Machine)	Multi-class classifier for text sentiment.	79%
Decision Tree	Tree-based classification algorithm.	75%
Naïve Bayes	Probabilistic model based on word frequencies.	81% (Best)

✅ The Naïve Bayes model performed the best and was selected for the final application with 81% accuracy.
### 5. Model Testing

To test the trained sentiment analysis model interactively:

print("")
review = input("Please enter the review you want to analyse: ")
review = preprocess_review(review)

r_type = model.predict(vec.transform([review]))
r_rating = model1.predict(vec.transform([review]))

print("This type is:", end=" ")
print(r_type)

print("This tweet is about:", end=" ")
print(r_rating)
print("")

This script:

Accepts a custom review input.

Applies the same preprocessing used during training.

Predicts both sentiment type (positive/negative/neutral) and topic rating.

Prints results in a clean, human-readable format.

## 📊 Visualizing Sentiment Scores

After prediction, we visualized sentiment scores to interpret overall customer sentiment.

Example: Using matplotlib or seaborn to visualize sentiment polarity distribution.

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df_output['Sentiment_score'], bins=20, kde=True)
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Polarity (-1 to +1)")
plt.ylabel("Frequency")
plt.show()

## 🔮 Insights from Sentiment Analysis

Highly positive reviews dominate across most product categories.

Negative reviews often correlate with delivery or quality issues.

The average sentiment polarity is slightly positive, indicating overall customer satisfaction.

## 🧩 Libraries and Dependencies
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
beautifulsoup4
requests

## 🧠 Future Enhancements

Automate data collection using Amazon APIs (where permitted).

Deploy as a web app for live review analysis.

Integrate topic modeling (LDA) to detect common themes.

Use deep learning (LSTM/BERT) for more accurate sentiment detection.

## 📚 References

Scikit-learn Documentation

Pandas API Reference

NLTK Text Processing Guide

Amazon Web Scraping using BeautifulSoup

Developed By:
👩‍💻 Urooj Fatima & Kaynat Sajid
