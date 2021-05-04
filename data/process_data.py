import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''load and merge messages and categories data'''

    # load and merge data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):
    
    '''clean dataframe by category column splitting & duplicate row removal'''
    
    # split categories into seperate columns
    categories = df.categories.str.split(';', expand=True).apply(lambda x: x.str.split('-').str.get(1)).astype('int32')
    category_colnames = list(df.categories.str.split(';', expand=True).iloc[0, :].str.split('-').str.get(0))
    categories.columns = category_colnames

    # replace df.categories with splited category columns
    df = pd.concat([df.drop(columns=['categories']), categories], axis=1)
    
    # remove duplicates
    df = df.drop_duplicates()

    # drop rows with related=2
    df = df[~(df.related==2)]
    
    
    return df


def save_data(df, database_filepath):
    
    '''load dataframe to sqlite table'''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('messages', engine, if_exists='replace', index=False)

    return None 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        print(df.head(2))
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()