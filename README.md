## CS410 Course Project

### Online Perceptions of the Russo-Ukrainian War


#### Overview
---
This project analyzes changes in user engagement and online sentiment regarding the Russo-Ukrainian War using Twitter data from March 2022, and June 2023 1. Sentiment values are generated during preprocessing with VADER[2], a sentiment analysis tool tailored for social media. Multiple scikit-learn models, including K-Nearest Neighbors, Support Vector Machines, and Random Forest, are trained and compared for accuracy against VADER's sentiment scores to select the best-performing model. The chosen model is then used to analyze sentiment trends over time, offering insights into shifts in public perception and online engagement during the war. The focus is on understanding online habits rather than the conflict itself.


#### How to use the software:
---
To use the software, you simply have to install the imported libraries and run main.py.

Data should be from the original Russo-Ukrainian War Twitter dataset, or refer to the data folder (data/) and use the provided, chunked data. You can conduct sentiment trend analysis by specifying start date data files and end date data files in main.py to obtain customized results from data files in the data folder.

To attain similar results to our experiment, you also should use the 08-19-2022.csv file to train and test the models.


#### Implementation
---
In our code, we utilize three text classification algorithms to categorize sentiments embedded in the tweets across time – K-Nearest Neighbors, Support Vector Machines, and Random Forest, making it a supervised learning experiment for fitting natural language to labeled sentiments. VADER is used to obtain the ground truth sentiment labels, and the best performing model on these ground truth labels was then used to predict sentiment scores for sentiment analysis on further data. 

Non-English tweets were destructively culled by removing all VADER scores == 0.0, but the information loss is expected to be minimal considering the rarity of exact 0.0 scores for English tweets.

Plotted results for model performance and predicted sentiment can be found in the resources folder (res/). For predicted sentiment, each data file is averaged out to get the overall sentiment of a particular day's tweets on the war.


#### Model Performance Results
---
For the 08-19-2022.csv data, the KNN algorithm attains 77.7% test accuracy, SVM classifier 88.2%, and Random Forest 83.4%. SVM and RF noticeably outperformed KNN, which was expected since the former two are less sensitive to outliers and more robust at high dimensionality compared to the latter. Mitigating the curse of dimensionality was significant since the training vocabulary was vectorized, thus the input data featured high dimensionality.

SVM's performance over RF was notable as RF is generally considered more suited to multiclass classification problems than SVM, which is more suited to binary classification. However, SVM models tend to perform better on sparse data, and for text vectors that have thousands of features with only a fraction of them having non-zero values, it appears that this was a much more significant factor than multiclass performance.

We concluded that SVM classifiers are most suitable for sentiment analysis in this particular dataset.


#### Sentiment Trends Analysis
---
For the first week of March 2022, sentiment trends were overall negative as expected following the outbreak of the war, but not extremely negative with the lowest compound sentiment almost reaching -0.4 where the lowest possible is -1.0.

For the first week of June 2023, sentiment trends were positive at the start of the week but quickly fell down to levels similar in March 2022. Considering the topic, it is expected that the overall sentiment would be fairly negative, so we considered the 'fall' a return to expectations. The positive bump is likely from news in late May of Ukraine's major counteroffensive that would start on June 8th, 2023.

Overall, sentiment was less negative than initially expected by us, but followed what we expected for major developments later in the war.
