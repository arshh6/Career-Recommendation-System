#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("carrer_datset.csv")


# In[3]:


df.head()


# In[4]:


df.describe


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
print(numeric_columns)


# In[8]:


categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[10]:


def label_encode_dataframe(df):
    df = df.copy()  # Preserve original dataset
    label_encoders = {}

    # Fill missing values with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # Apply Label Encoding
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for inverse transformation

    return df, label_encoders

# Usage
encoded_df, encoders = label_encode_dataframe(df)


# In[11]:


encoded_df.head()


# In[12]:


plt.figure(figsize=(10, 6))
sns.heatmap(encoded_df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV


# In[15]:


X = encoded_df.drop(columns=['Role'])
y = encoded_df['Role']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[16]:


print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# In[17]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[19]:


mcq_mapping = {
    0: {  # MCQ 1: What kind of tasks do you enjoy the most?
        "A": ("Web Development", 4),  
        "B": ("Data Science", 6),  
        "C": ("Cybersecurity", 8),  
        "D": ("Networking", 5)  
    },
    1: {  # MCQ 2: If you had to choose a tech-related hobby, what would it be?
        "A": ("Software Development", 4),  
        "B": ("Programming Skills", 6),  
        "C": ("Business Analysis", 8),  
        "D": ("Technical Communication", 5)  
    },
    2: {  # MCQ 3: What interests you the most in a tech job?
        "A": ("AI ML", 6),  
        "B": ("Cybersecurity", 8),  
        "C": ("Software Engineering", 7),  
        "D": ("Troubleshooting skills", 5)  
    },
    3: {  # MCQ 4: What type of projects would you like to work on?
        "A": ("Data Science", 7),  
        "B": ("Cybersecurity", 9),  
        "C": ("Distributed Computing Systems", 8),  
        "D": ("Database Fundamentals", 6)  
    },
    4: {  # MCQ 5: If you had to solve a real-world problem, what would you focus on?
        "A": ("Cybersecurity", 8),  
        "B": ("AI ML", 6),  
        "C": ("Cloud Computing", 7),  
        "D": ("Business Analysis", 5)  
    },
    5: {  # MCQ 6: How do you like to approach technical tasks?
        "A": ("Programming Skills", 5),  
        "B": ("Troubleshooting skills", 7),  
        "C": ("Project Management", 8),  
        "D": ("Technical Communication", 6)  
    },
    6: {  # MCQ 7: If you were leading a project, what role would you take?
        "A": ("Software Engineering", 7),  
        "B": ("Cybersecurity", 9),  
        "C": ("AI ML", 8),  
        "D": ("Database Fundamentals", 6)  
    },
    7: {  # MCQ 8: A company is facing frequent system crashes. What would you focus on?
        "A": ("Computer Architecture", 7),  
        "B": ("Distributed Computing Systems", 8),  
        "C": ("Cybersecurity", 9),  
        "D": ("Troubleshooting skills", 6)  
    },
    8: {  # MCQ 9: What motivates you to explore technology?
        "A": ("Software Development", 7),  
        "B": ("Networking", 6),  
        "C": ("AI ML", 8),  
        "D": ("Business Analysis", 5)  
    },
    9: {  # MCQ 10: What kind of technical skill do you want to master?
        "A": ("Computer Architecture", 6),  
        "B": ("Software Engineering", 7),  
        "C": ("Cybersecurity", 9),  
        "D": ("AI ML", 8)  
    }
}


# In[20]:


# user_answers = {
#     0: "D",  # User selects "AI ML"
#     1: "B",  # User selects "Networking"
#     2: "C",  # User selects "Troubleshooting skills"
# }

# user_answers = {}  # Empty dictionary to store user responses

# # Loop through all 10 questions and take input from the user
# for i in range(10):  # Since we have 10 MCQs
#     print(f"Question {i+1}: Select an option (A, B, C, D)")
#     user_choice = input("Your answer: ").strip().upper()  # Get user input and format it
    
#     while user_choice not in ["A", "B", "C", "D"]:  # Validate input
#         print("Invalid choice! Please enter A, B, C, or D.")
#         user_choice = input("Your answer: ").strip().upper()
    
#     user_answers[i] = user_choice  # Store the response

def get_user_responses():
    """
    Takes user input for 10 MCQs in a single line.
    Returns a dictionary mapping question numbers to selected options.
    """
    user_answers = {}  # Store user responses
    
    # Ask the user to enter all 10 answers in one line (e.g., A B C D A B C D A B)
    user_input = input("Enter your answers for all 10 questions (e.g., A B C D A B C D A B): ").strip().upper()
    
    # Split the input into a list of answers
    answers = user_input.split()

    # Validate the input length
    if len(answers) != 10:
        print("Error: Please enter exactly 10 responses separated by spaces.")
        return get_user_responses()  # Ask again if the input is invalid
    
    # Validate each answer
    for i, choice in enumerate(answers):
        if choice not in ["A", "B", "C", "D"]:
            print(f"Error: Invalid choice '{choice}' at position {i+1}. Please enter A, B, C, or D.")
            return get_user_responses()  # Ask again if there is an invalid input
        
        user_answers[i] = choice  # Store valid responses
    
    return user_answers

# Example Usage
user_answers = get_user_responses()
print("User Responses:", user_answers)




# In[21]:


import numpy as np

def convert_user_responses(user_answers, mcq_mapping, dataset_features):
    """
    Converts user MCQ responses into a numerical feature vector.
    
    Parameters:
    - user_answers (dict): User's answers to MCQs.
    - mcq_mapping (dict): Mapping of MCQ answers to dataset features.
    - dataset_features (list): List of dataset feature names.

    Returns:
    - NumPy array representing the user's feature vector.
    """

    # Initialize feature vector with zeros
    user_feature_vector = np.zeros(len(dataset_features))  

    # Map user answers to dataset features
    for question, answer in user_answers.items():
        if answer in mcq_mapping[question]:  # Check if answer exists in mapping
            skill_name, points = mcq_mapping[question][answer]  # Get skill & points
            if skill_name in dataset_features:
                skill_index = dataset_features.index(skill_name)  # Get skill index
                user_feature_vector[skill_index] += points  # Assign points

    return user_feature_vector.reshape(1, -1)  # Reshape for ML model input


# In[26]:


# print("Model was trained with:", X_train.shape[1], "features")  # Should be 17
# print("User input vector has:", user_features.shape[1], "features")  # Should also be 17
# print("Dataset columns:", dataset_features)


# In[22]:


dataset_features = list(encoded_df.columns)  # Get all feature names
if 'Role' in dataset_features:
    dataset_features.remove('Role')  # Remove 'Role' to match the trained model


# In[23]:


user_features = convert_user_responses(user_answers, mcq_mapping, dataset_features)
print("Final user input shape:", user_features.shape)  # Should match model's 17 features


# In[24]:


career_probabilities = model.predict_proba(user_features)  # Should work correctly
top_3_indices = np.argsort(career_probabilities[0])[-3:][::-1]
top_3_careers = encoders['Role'].inverse_transform(top_3_indices)

print("Top 3 Recommended Careers:", top_3_careers)


# In[ ]:





# In[ ]:





# In[94]:


# skill_map = {
#     0: {"A": 2, "B": 4, "C": 6, "D": 8},  
#     1: {"A": 1, "B": 3, "C": 5, "D": 7},  
#     2: {"A": 0, "B": 2, "C": 4, "D": 6},  
#     3: {"A": 1, "B": 3, "C": 5, "D": 7},  
#     4: {"A": 0, "B": 2, "C": 4, "D": 6},  
#     5: {"A": 1, "B": 3, "C": 5, "D": 7},  
#     6: {"A": 0, "B": 2, "C": 4, "D": 6},  
#     7: {"A": 1, "B": 3, "C": 5, "D": 7},  
#     8: {"A": 0, "B": 2, "C": 4, "D": 6},  
#     9: {"A": 1, "B": 3, "C": 5, "D": 7},  
# }


# In[95]:


# user_answers = {
#     0: "A",  # Maps to skill 2
#     1: "C",  # Maps to skill 5
#     2: "B",  # Maps to skill 2
#     3: "D",  # Maps to skill 7
#     4: "A",  # Maps to skill 0
#     5: "C",  # Maps to skill 5
#     6: "B",  # Maps to skill 2
#     7: "D",  # Maps to skill 7
#     8: "A",  # Maps to skill 0
#     9: "C",  # Maps to skill 5
# }


# In[97]:


# import numpy as np

# def predict_career(mcq_responses, trained_model, skill_mapping, total_features=17):
#     """
#     Predicts top 3 career domains based on user MCQ responses.

#     Parameters:
#     - mcq_responses (dict): User's answers mapped to relevant skills.
#     - trained_model (ML Model): Pre-trained classifier model.
#     - skill_mapping (dict): Mapping of MCQ answers to skills.
#     - total_features (int): Total number of skills (should match the model input features).

#     Returns:
#     - top_3_careers (list): Top 3 recommended career domains.
#     """

#     # Step 1: Initialize user skill vector with zeros
#     user_skills = np.zeros(total_features)  

#     # Step 2: Fill the skills based on user MCQ responses
#     for i, answer in mcq_responses.items():
#         if answer in skill_mapping[i]:
#             skill_index = skill_mapping[i][answer]
#             if skill_index < total_features:  # Avoid index errors
#                 user_skills[skill_index] += 1  

#     # Step 3: Reshape the input to match the model's expected shape
#     user_skills = user_skills.reshape(1, -1)  

#     # Step 4: Predict probabilities for each career domain
#     career_probabilities = trained_model.predict_proba(user_skills)

#     # Step 5: Extract top 3 career recommendations
#     top_3_indices = np.argsort(career_probabilities[0])[-3:][::-1]  # Get indices of top 3
#     top_3_careers = trained_model.classes_[top_3_indices]  # Get career labels

#     return top_3_careers


# In[98]:


# top_careers = predict_career(user_answers, model, skill_map, total_features=17)
# print("Top 3 Recommended Careers:", top_careers)


# In[99]:


# # Convert numeric output to career names using the stored encoder
# top_careers_names = encoders['Role'].inverse_transform(top_careers)

# print("Top 3 Recommended Careers:", top_careers_names)


# In[ ]:





# In[ ]:




