# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Algorithmia
from sklearn.externals import joblib
import pickle
from sklearn.pipeline import Pipeline

# Importing the dataset
dataset = pd.read_csv('bully.tsv', delimiter='\t', quoting=3)

# Cleaning the Texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 13159):
    review = re.sub('[^a-zA-z]', ' ', dataset['Message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

joblib.dump(classifier, "classifier1.pkl")
joblib.dump(cv, "cv1.pkl")
joblib.dump(sc, "sc1.pkl")

# Test New Data
# Importing the dataset

corpus = ["I fucking hate food"]

X = cv.transform(corpus).toarray()
X = sc.transform(X)
pred = classifier.predict_proba(X)
input = {
        "document": corpus[0]
    }
client = Algorithmia.client('simrdlXrUKTiKeVsEYIaiuQVa7/1')
algo = client.algo('nlp/SentimentAnalysis/1.0.4')
sentiment = algo.pipe(input).__getattribute__("result")[0]["sentiment"]
print(pred)
print(sentiment)
