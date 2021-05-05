# Disaster Response Pipeline
## Objective
This project build ETL & Machine Learning pipelines to build a model for an API that classifies disaster messages

## File structure
![image](https://user-images.githubusercontent.com/52469788/117084208-71be5580-acfb-11eb-80c1-add230834fb0.png)

## Data Source
Disaster text messages from Figure Eight
* disaster _categories.csv
* disaster_messages.csv

## Steps
1. Built ETL pipeline with Python that extract, clean and load the emergency messages to a SQLite database
2. create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification)
3. Create a Flask App that contain data visulizations and allow users to use the trained model to classify new messages.

## How to run it
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
