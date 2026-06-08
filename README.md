# SMS Spam Detection System

## Overview

This project detects whether an SMS message is Spam or Ham using Machine Learning and Natural Language Processing (NLP).

## Features

* SMS Spam Classification
* TF-IDF Vectorization
* Multinomial Naive Bayes Model
* Streamlit Web Application
* Model Persistence using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

## Model Performance

* Accuracy: 96.68%
* Precision: 100%
* Recall: 75%
* F1 Score: 86%

## Project Structure

* spam_detection.py : Model training and evaluation
* app.py : Streamlit web application
* spam_model.pkl : Saved machine learning model
* vectorizer.pkl : Saved TF-IDF vectorizer
* SMSSpamCollection : Dataset

## Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```
