import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from nltk.stem.snowball import SnowballStemmer
import re
from imblearn.over_sampling import SMOTE

# Define stopwords and stemmer (outside function for efficiency)
stop_words = 'english'
stemmer = SnowballStemmer('english')

# Pre-compile regex patterns for text cleaning
patterns = {
    r"what's": "what is",
    r"\'s": " ",
    r"\'ve": " have",
    r"can't": "can not",
    r"n't": " not",
    r"i'm": "i am",
    r"\'re": " are",
    r"\'d": " would",
    r"\'ll": " will",
    r'\W': ' ',  # Replace non-word characters
    r'\s+': ' '  # Replace multiple spaces with single space
}

# Function to preprocess text (combined clean and stem)
def preprocess_text(text):
    text = text.lower()
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    text = text.strip()
    return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

# Load the dataset and drop 'id'
df = pd.read_csv(r'balanced_train.csv').drop(columns=['id'])

# Preprocess comments in a single pass using the function
df['comment_text'] = df['comment_text'].apply(preprocess_text)

# Split input (X) and output (y)
X = df['comment_text']
y = df.drop(columns=['comment_text'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Logistic Regression Pipeline
LR_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),  # Reference defined stopword
    ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))
])

# Function to run the pipeline (avoid code duplication)
def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    pred_probs = pipeline.predict_proba(X_test)

    # Calculate ROC AUC for each label (same logic)
    roc_auc_scores = roc_auc_score(y_test, pred_probs, average=None)
    for idx, col in enumerate(y_test.columns):
        print(f'ROC AUC for {col}: {roc_auc_scores[idx]:.4f}')

    print('\nOverall accuracy:', accuracy_score(y_test, predictions))
    print('\nClassification report:')
    print(classification_report(y_test, predictions, target_names=y_train.columns))

# Run the pipeline and save model (unchanged)
run_pipeline(LR_pipeline, X_train_resampled, X_test, y_train_resampled, y_test)
joblib.dump(LR_pipeline, 'logistic_regression_model_balance.pkl')