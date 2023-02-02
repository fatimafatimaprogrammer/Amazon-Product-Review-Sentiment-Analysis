# Amazon-Product-Review-Sentiment-Analysis
Solution and Implementation
Steps for solution of our Research Assistant are:
Data Collection
EDA
Analysis through visualization
Data preprocessing
Data modelling
Data Collection:
Collected data from amazon website through web scabbing. We collected a lot of data related to many products then cut it down to small dataset because we labeled all our data manually on our own.
Visualize sentiment analysis output
This section will demonstrate how to analyze, visualize, and interpret the sentiment scores generated by the previous steps. Used the describe method in pandas to generate summary statics of values in the Sentiment_score field. These summary statistics reveal the numerical insights of this dataset using aggregatemetrics like count, min, max, median, etc.
The code snippet below generates summary statistics of sentiment_score field of df_output dataframe.

A quick review of these summary statistics reveals the following insights.
The min, which indicates the polarity or intensity of the most negative response is strongly negative (range of sentiment polarity score is -1 to +1)
The max, which indicates the polarity or intensity of the most positive response is highly positive (range of sentiment polarity score is -1 to +1)
The mean,which indicates the average polarity or intensity of sentiment across all responses is in the positive territory.

EDA (Exploratory Data Analysis):
The main purpose of EDA is to help look at data before making any assumptions. It can help identify obvious errors, as well as better understand patterns within the data, detect outliers or anomalous events, find interesting relations among the variables.
Data scientists can use exploratory analysis to ensure the results they produce are valid and applicable to any desired business outcomes and goals. EDA also helps stakeholders by confirming they are asking the right questions. EDA can help answer questions about standard deviations, categorical variables, and confidence intervals. Once EDA is complete and insights are drawn, its features can then be used for more sophisticated data analysis or modeling, including machine learning.

 
Data preprocessing
Lower Casing
Lower casing is a common text preprocessing technique. The idea is to convert the input text into same casing format so that 'text', 'Text' and 'TEXT' are treated the same way.
This is more helpful for text featurization techniques like frequency, tfidf as it helps to combine the same words together thereby reducing the duplication and get correct counts / tfidf values.

Removal of Punctuations
One another common text preprocessing technique is to remove the punctuations from the text data. This is again a text standardization process that will help to treat 'hurray' and 'hurray!' in the same way.

Removal of stop words 
Stopwords are commonly occuring words in a language like 'the', 'a' and so on. They can be removed from the text most of the times, as they don't provide valuable information for downstream analysis. In cases like Part of Speech tagging, we should not remove them as provide very valuable information about the POS.

Removal of Frequent and rare words
In the previos preprocessing step, we removed the stopwords based on language information. But say, if we have a domain specific corpus, we might also have some frequent words which are of not so much importance to us.
So this step is to remove the frequent words in the given corpus. If we use something like tfidf, this is automatically taken care of.
Data modelling
Base Algorithms
We’ve tried following four algorithms for classification 
1. Random Forest
2. Multi class SVM’s
3. Naïve Bayes
4. Decision Trees in Python

Best of all models what we applied which we used for our interactive application was Naïve bayes with 81% accuracy.
Testing our Model
Testing 



Code:
print("_____________________________________________________________________________")
review = input("Please enter the review you want to analyse: ")
review = preprocess_review(review)
r_type = model.predict(vec.transform([review]))
r_rating = model1.predict(vec.transform([review]))
print("This type is:", end =" ")
print(r_type)
print("This tweet is about:", end =" ")
print(r_rating)
print ("_____________________________________________________________________________")


