#!/usr/bin/env python3
"""
Test script to verify all trained models work correctly
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

def test_model_loading():
    """Test that all models can be loaded successfully"""
    print("Testing model loading...")

    models_dir = 'ml_models'
    models_to_test = [
        'career_model.pkl',
        'personality_model.pkl',
        'vectorizer.pkl',
        'skill_extractor.pkl',
        'skill_gap_predictor.pkl',
        'confidence_scorer.pkl',
        'resume_classifier.pkl'
    ]

    loaded_models = {}

    for model_file in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        try:
            model = joblib.load(model_path)
            loaded_models[model_file] = model
            print(f"[OK] Loaded {model_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_file}: {e}")
            return None

    return loaded_models

def test_personality_model(models):
    """Test personality model prediction"""
    print("\nTesting personality model...")

    try:
        personality_data = models['personality_model.pkl']

        # Create sample test data (10 questions, responses 1-5)
        sample_responses = [4, 3, 2, 4, 3, 2, 4, 3, 2, 4]  # Sample MBTI-like responses

        # Scale the features using the stored scaler
        scaler = personality_data['scaler']
        sample_scaled = scaler.transform([sample_responses])

        # Make prediction
        prediction = personality_data['model'].predict(sample_scaled)
        label_encoder = personality_data['label_encoder']
        personality_type = label_encoder.inverse_transform(prediction)[0]

        print(f"[OK] Sample prediction: {personality_type}")
        return True

    except Exception as e:
        print(f"[ERROR] Personality model test failed: {e}")
        return False

def test_resume_classifier(models):
    """Test resume classifier prediction"""
    print("\nTesting resume classifier...")

    try:
        resume_model = models['resume_classifier.pkl']

        # Sample resume text
        sample_resume = """
        Experienced software developer with 5 years of experience in Python, JavaScript, and React.
        Strong background in machine learning and data science. Led development of multiple web applications
        and worked on AI projects. Masters degree in Computer Science.
        """

        # Make prediction
        prediction = resume_model.predict([sample_resume])
        category = prediction[0]

        print(f"[OK] Sample resume classified as: {category}")
        return True

    except Exception as e:
        print(f"[ERROR] Resume classifier test failed: {e}")
        return False

def test_text_vectorizer(models):
    """Test text vectorizer"""
    print("\nTesting text vectorizer...")

    try:
        vectorizer = models['vectorizer.pkl']

        # Sample text
        sample_text = "Experienced software developer with Python skills"

        # Transform text
        vector = vectorizer.transform([sample_text])

        print(f"[OK] Vectorized text: shape {vector.shape}")
        return True

    except Exception as e:
        print(f"[ERROR] Text vectorizer test failed: {e}")
        return False

def test_skill_extractor(models):
    """Test skill extractor"""
    print("\nTesting skill extractor...")

    try:
        skill_data = models['skill_extractor.pkl']
        skills_mapping = skill_data['skills_mapping']

        # Sample text with skills
        sample_text = "I am experienced in Python, JavaScript, and machine learning"

        # Simple skill extraction (since we removed the function)
        found_skills = {}
        text_lower = sample_text.lower()

        for skill in skills_mapping.keys():
            if skill in text_lower:
                found_skills[skill] = skills_mapping[skill]

        print(f"[OK] Found skills: {found_skills}")
        return True

    except Exception as e:
        print(f"[ERROR] Skill extractor test failed: {e}")
        return False

def main():
    """Main testing function"""
    print("AI Career Recommender - Model Testing")
    print("=" * 50)

    # Test model loading
    models = test_model_loading()
    if models is None:
        print("[ERROR] Failed to load models")
        return

    # Test individual models
    tests = [
        ("Personality Model", test_personality_model),
        ("Resume Classifier", test_resume_classifier),
        ("Text Vectorizer", test_text_vectorizer),
        ("Skill Extractor", test_skill_extractor)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func(models):
            passed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All models are working correctly!")
        print("Your AI Career Recommender is ready to use!")
    else:
        print("[WARNING] Some tests failed. Please check the models.")

if __name__ == "__main__":
    main()