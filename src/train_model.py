import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from src.preprocess import clean_text

DATA_PATH="data/"
MODEL_PATH="models/"
os.makedirs(MODEL_PATH,exist_ok=True)

def train():
    print("Loading datasets.......")
    try:
        true_df=pd.read_csv(os.path.join(DATA_PATH,'true.csv'))
        fake_df=pd.read_csv(os.path.join(DATA_PATH,'fake.csv'))
    
    except FileNotFoundError:
        print("Error:CSV file not found in 'data/' folder")
        return
    
    true_df['label']="REAL"
    fake_df['label']="FAKE"

    df=pd.concat([true_df,fake_df],ignore_index=True)
    df=df.sample(frac=1).reset_index(drop=True)

    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    print("Vectorizing..")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train=vectorizer.fit_transform(X_train)
    tfidf_test = vectorizer.transform(X_test)

    print("Training model...")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Evaluation
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully! Accuracy: {score*100:.2f}%")
    
    # Save Model and Vectorizer
    print("Saving artifacts to 'models/'...")
    joblib.dump(pac, os.path.join(MODEL_PATH, 'fake_news_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
    print("Done.")

if __name__ == "__main__":
    train()


