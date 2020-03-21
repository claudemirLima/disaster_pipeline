#imports

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
import pandas as pd
import re
import sys
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def load_data(database_filepath):
    
     """
    Load data from Database
    
    Args:
    database_filepath: file
    Returns:
    Pandas Series : X with colum message 
    DataFrame: y with  other columns target
    target_names : target labels 
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messege_categories', engine)
    
    X = df.message
    y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    
    target_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    
    
    return X,y,target_names


def tokenize(text):
    
    """
    Tokenizes text data
    Args:
    String : Messages as text data
    Returns:
    List of Words
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    find_urls = re.findall(url_regex, text)
    
    for url in find_urls:
        text = text.replace(url, 'urlplaceholder')
    # tokenize text and remove any characters
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    # remove stopswords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # extract root form of words
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
     
    return tokens


def build_model():
    
    """
    Build ne model ML
    Args:
    Returns:
    GridSearchCV model with pipeline
    """
    
    #Create Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #set Hyper Param
    parameters = [
            {"vect__stop_words":["english"],
             "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
             "tfidf__smooth_idf":[True],
             "tfidf__use_idf":[True],
             "clf": [RandomForestClassifier()],
             "clf__n_estimators": [10],
             "clf__min_samples_leaf":[1], 
             "clf__min_samples_split":[2],
             "clf__n_jobs":[1],
            }
    ]
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluete model ML
    Args:
    Model
    X_test: menssage separate for test 
    DataFrame Y_test: separate for test 
    List Categories: list labels
    Returns:
    """
    
    # predict
    y_pred = model.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save model ML
    Args:
    Model
    model_filepath: local to save model ML 
    Returns:
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()