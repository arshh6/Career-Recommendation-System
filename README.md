# Career Recommendation System

A Machine Learning based web application that predicts suitable career domains based on user interests, skills, and technical preferences.

## Overview

This project is designed to help users discover potential career paths by answering a set of multiple-choice questions. The system processes user responses, converts them into feature vectors, and uses a trained Random Forest Classifier model to predict the most suitable career domains.

The application is built using:

* Python
* Flask
* Scikit-learn
* NumPy
* HTML/CSS

--_

## Features

* Interactive quiz-based career assessment
* Machine Learning based prediction system
* Top 3 career recommendations
* Flask backend integration
* Modern frontend UI
* Probability-based prediction logic

---

## Technologies Used

### Frontend

* HTML
* CSS

### Backend

* Flask

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* NumPy
* Joblib

---

## Project Structure

```bash
Career-Recommendation-System/
│
├── app.py
├── career_model.pkl
├── label_encoder.pkl
├── requirements.txt
│
├── templates/
│   ├── home.html
│   ├── quiz.html
│   └── result.html
```

---

## How It Works

1. User opens the web application
2. User starts the career quiz
3. User answers 10 MCQ-based questions
4. Responses are converted into weighted feature vectors
5. The trained ML model predicts career probabilities
6. Top 3 career recommendations are displayed

---

## Machine Learning Workflow

* Data preprocessing
* Feature engineering
* Label encoding
* Train-test split
* Random Forest model training
* Prediction using probability scores

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/arshh6/Career-Recommendation-System.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000
```

---

## Future Improvements

* Resume upload feature
* NLP-based career analysis
* LLM integration for career guidance
* Dashboard analytics
* User authentication
* Database integration

---

## Author

Arshdeep Kaur

---


This project is for educational and learning purposes.
