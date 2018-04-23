Algorithm 1: Supervised LDA with Logistic Regression Classification:

1. Performed LDA algorithm using Gensim topic modelling toolkit implemented in python. LDA algorithm is used to extract 25 topics from review datasets. 25 topics is chose based on trial and error method for achieving maximum accuracy and discriminative power of topics.
2. Positive and negative reviews are represented in topic space using LDA topic proportion.
3. Represented reviews in topic space and is fed as input to logistic regression model to train model for classifying sentiments of reviews.
4. Logistic regression model is used to predict sentiment of reviews as positive and negative.
5. Top terms of each identified topic of reviews are extracted as features of reviews.
