import pandas as pd
import numpy as np
import re
import nltk
import time
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def load_data(file_path):

  df = pd.read_csv(file_path, nrows=10000, on_bad_lines='skip')
  df = df[['text']]
  df = df.drop_duplicates()
  df = df.dropna()

  return df


def clean_text(text):
  text = text.lower()
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
  text = re.sub(r'\@\w+|\#', '', text) # Remove mentions and hashtags
  text = re.sub(r'[^A-Za-z0-9 ]+', '', text) # Remove special characters, keep alphanumeric only

  # NOTE: VADER handles stopwords and doesn't require tokenization or stemming/lemmatization

  return text.strip()


def calc_vader_compound(text):
  """
  Calculate VADER compound sentiment for the given text.
  negative, neutral and positive sentiments range from [0, 1]
  compound sentiment ranges from [=1, 1] and denotes overall sentiment
  compound ranges: negative[-1, -0.05), neutral[-0.05, 0.05], positive(0.05, 1]
  ---
  return: (compound_sentiment, sentiment_category)
          compound_sentiment - float - compound sentiment score
          sentiment_category - string - category of 'Positive', 'Neutral', or 'Negative'
  """

  SIA = SentimentIntensityAnalyzer()
  sentiment = SIA.polarity_scores(text)
  compound_sentiment = sentiment['compound']

  if compound_sentiment > 0.05:
    sentiment_category = 'Positive'
  elif compound_sentiment < -0.05:
    sentiment_category = 'Negative'
  else:
    sentiment_category = 'Neutral'

  return (compound_sentiment, sentiment_category)


def time_diff(start_time, end_time):
  """
  Calculates the time difference in minutes.
  ---
  returns: float - the time difference in minutes.
  """

  return (end_time - start_time) / 60


def train_test_models(x_train, y_train, x_test, y_test):
  """
  Train the pre-set models, then evaluate their accuracy on a test set.
  Prints out the results (and the training time).
  ---
  returns: results - dict {string: float} - the models and their accuracies on the test set.
           best_model - scikit-learn model - the best model between KNN, SVM, and RF
  """

  models = {
      'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
      'Support Vector Machine': SVC(kernel='linear', random_state=42),
      'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
  }

  # Holds accuracy results
  results = {}
  # Checks for the best performing model
  best_accuracy = 0
  best_model = None

  # Train the models, then evaluate their accuracy
  for name, model in models.items():
    print(f"Training {model}...")

    start = time.time()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    end = time.time()

    process_time = time_diff(start, end)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Training time: {process_time:.2f} minutes")
    print(f"{name} Accuracy: {accuracy:.3f}\n")

    results[name] = accuracy
    # Check for best model
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model = model

  return results, best_model


def predict_sentiment_data(model, vectorizer, data_list):
  """
  Predicts sentiment values for a list of data files using a given model.
  Vectorizer will apply transform on the unseen data for predictions.
  ---
  returns: results - list (string, float) - the files and their predicted average sentiments.
  """

  results = []

  for data_file in data_list:
    # Load in the data, clean it, then predict its sentiment
    df = load_data(data_file)
    df['text'] = df['text'].apply(clean_text)
    # Since the data is an unseen 'test' set, transform only for seen words
    X = vectorizer.transform(df["text"])
    predictions = model.predict(X)
    results.append((data_file, np.mean(predictions)))

  return results


# Plot Model Accuracy Comparison
def plot_model_accuracy(model_accuracies, output_file_path):
    # models = ['KNN', 'Random Forest', 'SVM']
    # accuracies = [knn_acc, rf_acc, svm_acc]
    models = []
    accuracies = []
    for model, accuracy in model_accuracies.items():
      models.append(model)
      accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['red', 'green', 'blue'])
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.savefig(output_file_path)


# Graphing Sentiment Trends
def plot_sentiment_trends(dates, sentiments, output_file_path):
    # Get data, organized as (data_file, average sentiment)
    start_date = dates[0]
    end_date = dates[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, sentiments, marker='o', color='darkblue', label='Average Compound Sentiment')
    plt.axhline(0.05, color='green', linestyle='--', label='Positive Threshold')
    plt.axhline(-0.05, color='red', linestyle='--', label='Negative Threshold')
    plt.title(f'Sentiment Trends from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Sentiment')
    plt.legend()
    plt.savefig(output_file_path)


nltk.downloader.download('vader_lexicon')

"""========================INPUT DATA HERE========================"""
# Load in the training and testing data
data_file_path = "./data/"
training_file_path = data_file_path + "08-19-2022.csv"

# Set the files for sentiment analysis; first the starting set, then the ending set for comparison
sentiment_files_start = [
  "03-01-2022.csv",
  "03-02-2022.csv",
  "03-03-2022.csv",
  "03-04-2022.csv",
  "03-05-2022.csv",
  "03-06-2022.csv",
  "03-07-2022.csv"
]
sentiment_files_end = [
  "06-01-2023.csv",
  "06-02-2023.csv",
  "06-03-2023.csv",
  "06-04-2023.csv",
  "06-05-2023.csv",
  "06-06-2023.csv",
  "06-07-2023.csv"
]

# Add the data filepath to the specified sentiment data files
sentiment_data_start = [data_file_path + date_string for date_string in sentiment_files_start]
sentiment_data_end = [data_file_path + date_string for date_string in sentiment_files_end]
"""========================INPUT DATA HERE========================"""

"""========================OUTPUT DATA HERE========================"""
# training_output_path = "/res/08-19-2022.csv"
# Plot output files
plots_file_path = "./res/"
model_results_file = plots_file_path + 'ModelResults.png'
sentiment_trends_start = plots_file_path + 'SentimentTrendsStart.png'
sentiment_trends_end = plots_file_path + 'SentimentTrendsEnd.png'
"""========================OUTPUT DATA HERE========================"""


df = load_data(training_file_path)
df['text'] = df['text'].apply(clean_text)

start = time.time()
df[['compound_sentiment', 'sentiment_category']] = df['text'].apply(calc_vader_compound).apply(pd.Series)
end = time.time()

process_time_in_min = time_diff(start, end)
print(f"Data Pre-processing time: {process_time_in_min} minutes")

df = df.drop(df[df['compound_sentiment'] == 0.0].index)
# NOTE: non-English tweets are included in the dataset and scored as 0.0000 Neutral
#       this is a destructive method of circumventing that by removing all 0.0000 tweets
#       it is expected that the information loss is minimal considering the rarity of exact 0.0000 scores

# df.to_csv(training_output_path, index=False)

# Vectorize the text, then train and test the models
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
Y = df["sentiment_category"].map({"Positive": 1, "Neutral": 0, "Negative": -1})

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model_accuracies, best_model = train_test_models(x_train, y_train, x_test, y_test)


# Predict the values for the rest of the data using the best performing model
start_sentiments = predict_sentiment_data(best_model, vectorizer, sentiment_data_start)
end_sentiments = predict_sentiment_data(best_model, vectorizer, sentiment_data_end)


# Plot Model Accuracy Comparison
plot_model_accuracy(model_accuracies, model_results_file)

# Get the dates for plotting
start_dates = [os.path.splitext(data_file_path)[0] for data_file_path in sentiment_files_start]
end_dates = [os.path.splitext(data_file_path)[0] for data_file_path in sentiment_files_end]
# Get the sentiments for plotting
start_avg_sentiments = [tup[1] for tup in start_sentiments]
end_avg_sentiments = [tup[1] for tup in end_sentiments]

# Plot Sentiment Trends
plot_sentiment_trends(start_dates, start_avg_sentiments, output_file_path=sentiment_trends_start)
plot_sentiment_trends(end_dates, end_avg_sentiments, output_file_path=sentiment_trends_end)