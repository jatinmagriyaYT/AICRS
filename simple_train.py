#!/usr/bin/env python3
"""
Simple Model Training Script - No Unicode Issues
"""

import pandas as pd
import numpy as np
import joblib
import os
import random
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("AI Career Recommender - Model Training Started")
print("=" * 60)

def load_datasets():
    """Load all required datasets"""
    print("Loading datasets...")

    try:
        career_df = pd.read_csv('datasets/career_dataset.csv')
        skills_df = pd.read_csv('datasets/skills_dataset.csv')
        personality_df = pd.read_csv('datasets/personality.csv')

        print(f"Career dataset: {career_df.shape}")
        print(f"Skills dataset: {skills_df.shape}")
        print(f"Personality dataset: {personality_df.shape}")

        return career_df, skills_df, personality_df

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None

def train_career_model(career_df):
    """Train Career Recommender Model"""
    print("Training Career Recommender Model...")

    # Create training data
    training_data = []

    for _, career in career_df.iterrows():
        for i in range(50):
            education_match = 1 if random.random() > 0.3 else 0
            skills_match = random.randint(0, 1)
            experience_level = 1 if random.random() > 0.4 else 0
            interest_match = random.randint(0, 1)

            features = [education_match, skills_match, experience_level, interest_match]
            training_data.append(features + [career['id']])

    training_df = pd.DataFrame(training_data, columns=['education', 'skills', 'experience', 'interests', 'career_id'])

    X = training_df.drop('career_id', axis=1)
    y = training_df['career_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.3f}")

    return model

def train_personality_model(personality_df):
    """Train Personality Assessment Model"""
    print("Training Personality Assessment Model...")

    training_data = []

    for _, personality in personality_df.iterrows():
        for i in range(40):
            responses = []

            for question in range(10):
                if question < 3:
                    base_score = 4 if personality['personality_type'][0] == 'E' else 2
                elif question < 5:
                    base_score = 4 if personality['personality_type'][1] == 'S' else 2
                elif question < 7:
                    base_score = 4 if personality['personality_type'][2] == 'T' else 2
                else:
                    base_score = 4 if personality['personality_type'][3] == 'J' else 2

                response = max(1, min(5, base_score + random.randint(-1, 1)))
                responses.append(response)

            training_data.append(responses + [personality['personality_type']])

    columns = [f'q{i+1}' for i in range(10)] + ['personality_type']
    training_df = pd.DataFrame(training_data, columns=columns)

    le = LabelEncoder()
    y_encoded = le.fit_transform(training_df['personality_type'])

    X = training_df.drop('personality_type', axis=1)
    y = y_encoded

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.3f}")

    return model, scaler, le

def train_text_vectorizer(career_df, skills_df, personality_df):
    """Train Text Vectorizer"""
    print("Training Text Vectorizer...")

    text_data = []
    text_data.extend(career_df['description'].tolist())
    text_data.extend(skills_df['skill_name'].tolist())
    text_data.extend(personality_df['posts'].tolist())

    sample_texts = [
        "Experienced software developer with Python and JavaScript skills",
        "Marketing professional with digital campaign experience",
        "Data analyst with SQL and business intelligence skills"
    ]
    text_data.extend(sample_texts * 10)

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return vectorizer

def create_skill_extractor(skills_df):
    """Create skill extractor"""
    print("Creating Skill Extractor...")

    skills_mapping = {}
    for _, row in skills_df.iterrows():
        skill_name = row['skill_name'].lower()
        category = row['category']
        skills_mapping[skill_name] = category

    return skills_mapping

def main():
    """Main training function"""
    career_df, skills_df, personality_df = load_datasets()

    if career_df is None or skills_df is None or personality_df is None:
        print("Failed to load datasets")
        return

    # Train models
    print("Starting model training...")

    career_model = train_career_model(career_df)
    personality_model, scaler, label_encoder = train_personality_model(personality_df)
    vectorizer = train_text_vectorizer(career_df, skills_df, personality_df)
    skill_extractor = create_skill_extractor(skills_df)

    # Save models
    os.makedirs('ml_models', exist_ok=True)

    joblib.dump(career_model, 'ml_models/career_model.pkl')
    joblib.dump({
        'model': personality_model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }, 'ml_models/personality_model.pkl')
    joblib.dump(vectorizer, 'ml_models/vectorizer.pkl')
    joblib.dump(skill_extractor, 'ml_models/skill_extractor.pkl')

    print("All models trained and saved successfully!")

if __name__ == "__main__":
    main()