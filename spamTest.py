import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle

# Download NLTK data if not already available
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].replace(['ham', 'spam'], [0, 1])

# Text preprocessing
corpus = []
ps = PorterStemmer()

for i in range(len(df)):
    msg = df.loc[i, 'message']
    # Apply regular expressions
    msg = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', msg)
    msg = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
    msg = re.sub(r'£|\$', 'moneysymb', msg)
    msg = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', msg)
    msg = re.sub(r'\d+(\.\d+)?', 'numbr', msg)
    msg = re.sub(r'[^\w\d\s]', ' ', msg)
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word) for word in msg if word not in set(stopwords.words('english'))]
    msg = ' '.join(msg)
    corpus.append(msg)

# Convert text to numerical data
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df['label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train model
dt = DecisionTreeClassifier(random_state=50)
dt.fit(X_train, y_train)

# Evaluate model
y_pred = dt.predict(X_test)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save the model
with open('spam_detection_model.pkl', 'wb') as model_file:
    pickle.dump(dt, model_file)

# Save the vectorizer as well
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)
