# Disaster Response Pipeline
## Objective
This project build ETL & Machine Learning pipelines to build a model for an API that classifies disaster messages

## File structure
* app
** asdf
* data
* models
* WebApp screenshots/pdf
* requirements.txt
* README.md
* app     
│   ├── run.py                           # Flask file that runs app
│   └── templates   
│       ├── go.html                      # Classification result page of web app
│       └── master.html                  # Main page of web app    
├── data                   
│   ├── disaster_categories.csv          # Dataset including all the categories  
│   ├── disaster_messages.csv            # Dataset including all the messages
│   └── process_data.py                  # Data cleaning
├── models
│   ├── train_classifier.py              # Train ML model
│   └── classifier.pkl                   # pikkle file of model   
|   
|── requirements.txt                     # contains versions of all libraries used.
|
└── README.md

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
