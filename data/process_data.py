#imports
import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    
    """
    Loads data from csv
    Args:
    messages_filepath:  file type csv
    categories_filepath: file type csv
    Returns:
    df : Data Frame with messeges and categories 
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages,categories)
    
    return df


def clean_data(df):
    
    """
    - Cleans data
    
    Args:
    DataFrame: df get in load_data function
    Returns:
    DataFrame df : Cleaned data to be used by model
    """
    
    #separate values
    categories = df["categories"].str.split(";",expand=True)
    #Alter name of new categories
    categories = categories.rename(columns=lambda x: (categories[x].str.split("-")[0])[0])
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.replace(x,"0") if '0' in x else x.replace(x,"1") )
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(str).astype(int)
    #drop column categories
    df.drop(['categories'],axis=1,inplace=True)
    #Concat two dataframe
    df= pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset=['id'],inplace=True)
    #drop colums 'original'
    df.drop(columns=['original'],axis=1,inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save data in persistence mode
    Args:
    DataFrame: df got in clean_data function
    database_filename: string of name database
    Returns:
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messege_categories', engine, index=False, if_exists='replace')
    


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
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()