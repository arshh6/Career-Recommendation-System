from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and encoder
model = joblib.load("career_model.pkl")
encoders = joblib.load("label_encoder.pkl")


# MCQ Mapping
mcq_mapping = {
    0: {"A": ("Web Development", 4), "B": ("Data Science", 6), "C": ("Cybersecurity", 8), "D": ("Networking", 5)},
    1: {"A": ("Software Development", 4), "B": ("Programming Skills", 6), "C": ("Business Analysis", 8), "D": ("Technical Communication", 5)},
    2: {"A": ("AI ML", 6), "B": ("Cybersecurity", 8), "C": ("Software Engineering", 7), "D": ("Troubleshooting skills", 5)},
    3: {"A": ("Data Science", 7), "B": ("Cybersecurity", 9), "C": ("Distributed Computing Systems", 8), "D": ("Database Fundamentals", 6)},
    4: {"A": ("Cybersecurity", 8), "B": ("AI ML", 6), "C": ("Cloud Computing", 7), "D": ("Business Analysis", 5)},
    5: {"A": ("Programming Skills", 5), "B": ("Troubleshooting skills", 7), "C": ("Project Management", 8), "D": ("Technical Communication", 6)},
    6: {"A": ("Software Engineering", 7), "B": ("Cybersecurity", 9), "C": ("AI ML", 8), "D": ("Database Fundamentals", 6)},
    7: {"A": ("Computer Architecture", 7), "B": ("Distributed Computing Systems", 8), "C": ("Cybersecurity", 9), "D": ("Troubleshooting skills", 6)},
    8: {"A": ("Software Development", 7), "B": ("Networking", 6), "C": ("AI ML", 8), "D": ("Business Analysis", 5)},
    9: {"A": ("Computer Architecture", 6), "B": ("Software Engineering", 7), "C": ("Cybersecurity", 9), "D": ("AI ML", 8)}
}

# Dataset Features
dataset_features = [
    'Database Fundamentals',
    'Computer Architecture',
    'Distributed Computing Systems',
    'Cyber Security',
    'Networking',
    'Software Development',
    'Programming Skills',
    'Project Management',
    'Computer Forensics Fundamentals',
    'Technical Communication',
    'AI ML',
    'Software Engineering',
    'Business Analysis',
    'Communication skills',
    'Data Science',
    'Troubleshooting skills',
    'Graphics Designing'
]

# Convert user responses into ML feature vector
def convert_user_responses(user_answers):

    user_features = np.zeros(len(dataset_features))

    for i, answer in enumerate(user_answers):

        if answer in mcq_mapping[i]:

            skill_name, points = mcq_mapping[i][answer]

            if skill_name in dataset_features:

                skill_index = dataset_features.index(skill_name)

                user_features[skill_index] += points

    return user_features.reshape(1, -1)


# ---------------- HOME PAGE ----------------

@app.route('/')
def home():
    return render_template('home.html')


# ---------------- QUIZ PAGE ----------------

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')


# ---------------- PREDICTION ROUTE ----------------

@app.route('/predict', methods=['POST'])
def predict():

    user_answers = [
        request.form['q1'],
        request.form['q2'],
        request.form['q3'],
        request.form['q4'],
        request.form['q5'],
        request.form['q6'],
        request.form['q7'],
        request.form['q8'],
        request.form['q9'],
        request.form['q10']
    ]

    # Convert responses to feature vector
    user_features = convert_user_responses(user_answers)

    # Predict probabilities
    career_probabilities = model.predict_proba(user_features)

    # Get Top 3 careers
    top_3_indices = np.argsort(career_probabilities[0])[-3:][::-1]

    top_3_careers = encoders.inverse_transform(top_3_indices)

    # Send results to result page
    return render_template(
        'result.html',
        careers=top_3_careers
    )


# ---------------- RUN APP ----------------

if __name__ == '__main__':
    app.run(debug=True)