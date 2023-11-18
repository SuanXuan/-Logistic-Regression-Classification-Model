import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Data Preprocessing
def preprocess_data():
    # List of file names
    product_file = ['./dataset/products-data-0.tsv', './dataset/products-data-1.tsv', './dataset/products-data-2.tsv', './dataset/products-data-3.tsv']
    review_file = ['./dataset/reviews-0.tsv', './dataset/reviews-1.tsv', './dataset/reviews-2.tsv', './dataset/reviews-3.tsv']
    # Combine files and pay attentation on unicode
    products = [pd.read_csv(f, sep='\t', names=['id', 'category', 'title'], encoding='utf-8') for f in product_file]
    reviews = [pd.read_csv(f, sep='\t', names=['id', 'rating', 'review'], encoding='utf-8') for f in review_file]
    products_dataframe = pd.concat(products , ignore_index=True)
    # Correct some typo labels about 'Kitchen'
    products_dataframe['category'] = products_dataframe['category'].replace('Ktchen', 'Kitchen')
    reviews_dataframe = pd.concat(reviews, ignore_index=True)
    # Merge product and review dataframes on 'id'
    merged_data = pd.merge(products_dataframe, reviews_dataframe, on='id')
    return merged_data

# Feature Engineering
def feature_engineering(data):
    # Combine the title and review
    data['combined_text'] = data['title'] + ' ' + data['review']
    # Lowercase and remove punctuation
    data['combined_text'] = data['combined_text'].str.lower().str.replace('[^\w\s]', '')
    # Integer encode 'rating'
    data['rating'] = data['rating'].astype(int)
    return data

# Model Training
def train_model(data):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # adjust the number of features
    X_text = tfidf_vectorizer.fit_transform(data['combined_text'])
    X_rating = data[['rating']].values  
    # Combine text features and rating feature
    X = sparse.hstack([X_text, X_rating]) 
    y = data['category']
    # Start to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

# Main function
def main():
    data = preprocess_data()
    data = feature_engineering(data)
    model, X_test, y_test = train_model(data)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
