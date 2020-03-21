This project is to build a Pipeline Project with objective  of   disaster response. We are using data from Figure Eight to build a model for an API that classifies disaster messages.


Project Components

There are three components in this project.
1. ETL Pipeline

In a Python script, process_data.py,  for data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline

In a Python script, train_classifier.py, There is machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App

There is a app in html and javascript with Flask to show  
the input message of the disasrter and show the result of analyzed cd ..