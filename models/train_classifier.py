import sys

import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    '''
       load data from database
       return predictor dataframe, target variable dataframe and category name list
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)
    #df = df[~(df.related==2)]
    X = df.message
    y = df.iloc[:, 4:]
    category_name = y.columns.values
    return X, y, category_name


def tokenize(text):
    '''
    Clean text in message column
       - tokenization
       - lemmatization
       - lower
       - remove space
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    tokens = [word for word in tokens if word not in stopwords_]   
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
        
    return clean_tokens


def build_model():
    '''build a machine learning pipeline
       return a grid search object
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.75, 1.0),
        'clf__estimator__min_samples_split': [2, 3, 5]
        #'clf__estimator__max_depth': [3, 5, 7]
             
        

    }
    
    cv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model performance in each category'''
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    for i, var in enumerate(category_names):
        print(var)
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    "Save the model as a pickle file "
    pkl_file = model_filepath
    model_pickle = open(pkl_file,'wb')
    pickle.dump(model, model_pickle)
    model_pickle.close()



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(category_names)
        
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