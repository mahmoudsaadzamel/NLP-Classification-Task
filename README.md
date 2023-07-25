# Task-3-Classification
build a machine learning-based classifier (LGBMClassifier)r and show its performance on the test set (20%).

Training Process and Model Evaluation

##Introduction

This README file provides a brief overview of the training process and model evaluation for the machine learning-based classifier on the provided news articles dataset. The goal of this task is to build a classifier that can accurately predict the topic of each news article.

## Data Preprocessing

1. **Text Cleaning and Tokenization:** The news articles were preprocessed by removing special characters, punctuation, and converting text to lowercase. The text was then tokenized to extract individual words.

2. **Stopword Removal:** Common Arabic stopwords were removed from the tokenized text to reduce noise in the data.

3. **Text Normalization:** The ISRIStemmer was used to normalize the words in the text by applying stemming.

4. **Vectorization:** The preprocessed text was transformed into a numerical representation using the TF-IDF Vectorizer.

## Model Building

The LightGBM classifier was chosen for this task due to its ability to handle large datasets and its efficiency in training. The classifier was configured with hyperparameters such as 'num_leaves' and 'learning_rate' to achieve optimal performance.

## Model Evaluation

The trained model was evaluated on the test set to assess its performance in predicting the topic of news articles. The following metrics were used to evaluate the model:

1. **Precision:** Precision measures the accuracy of positive predictions for each class. It represents the model's ability to avoid false positives.

2. **Recall:** Recall measures the classifier's ability to find all positive instances of each class. It represents the model's ability to avoid false negatives.

3. **F1-Score:** The F1-Score is the harmonic mean of precision and recall, providing a balanced measure of both metrics. It is useful when considering both false positives and false negatives.

4. **Accuracy:** Accuracy measures the overall correctness of predictions across all classes.

### Results

The performance of the model on the test set is as follows:

| Class              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| sport              | 85.92     | 88.50  | 87.19    |
| regions            | 78.92     | 88.00  | 83.21    |
| orbites            | 95.50     | 95.50  | 95.50    |
| medias             | 82.38     | 86.50  | 84.39    |
| marocains-du-monde | 93.58     | 87.50  | 90.44    |
| politique          | 73.30     | 70.00  | 71.61    |
| faits-divers       | 79.50     | 79.50  | 79.50    |
| economie           | 87.50     | 84.00  | 85.71    |
| art-et-culture     | 75.14     | 65.00  | 69.71    |
| tamazight          | 98.01     | 98.50  | 98.25    |
| societe            | 91.24     | 99.00  | 94.96    |

**Overall Metrics:**
- Precision: 85.55
- Recall: 85.64
- F1-Score: 85.50
- Accuracy: 85.64

## Enhancements

To achieve even better results, the following enhancements can be considered:

1. **Hyperparameter Tuning:** Perform a thorough search for the best combination of hyperparameters for the LightGBM classifier.

2. **Text Representation Techniques:** Explore advanced text representation techniques like word embeddings or pre-trained language models.

3. **Ensemble Methods:** Consider using ensemble methods like Voting or Stacking to combine predictions from multiple models.

4. **Cross-Validation:** Implement cross-validation to ensure robust performance estimation..

5. **Collect More Data:** If feasible, collect more data to increase the diversity and representativeness of the dataset.

6. **Data Augmentation:** If collecting more data is not feasible, consider data augmentation techniques, especially for text data. Techniques like back-translation, word replacement, or paraphrasing can increase the diversity of the training data.


By combining these enhancements with continuous experimentation and analysis, the model's performance can be further improved 
