#!/usr/bin/env python3
"""
AI Career Recommender - ENHANCED MODEL TRAINING SCRIPT
================================================================
LIBRARY USAGES:
pandas, numpy - Data manipulation and numerical operations
joblib - Model saving/loading
warnings - Handling warnings
os - File system operations
re - Regular expressions for text cleaning
json - JSON serialization
datetime - Timestamp handling
sklearn - Machine learning models and utilities
nltk - Natural Language Processing
imblearn - Handling imbalanced datasets
xgboost - Gradient boosting algorithm
"""

import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical operations
import joblib        # Model serialization (saving/loading models)
import warnings      # Warning control
import os            # Operating system interfaces (file paths)
import re            # Regular expressions (text pattern matching)
import json          # JSON serialization/deserialization
from datetime import datetime  # Date and time operations

# ================================================================================
# MACHINE LEARNING LIBRARIES FROM scikit-learn
# ================================================================================
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
# train_test_split - Split data into training and test sets
# cross_val_score - Cross-validation performance scoring
# GridSearchCV - Hyperparameter tuning
# StratifiedKFold - Cross-validation with class distribution preservation

from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.feature_extraction.text import TfidfVectorizer  # Text to numerical features
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
# accuracy_score - Calculate accuracy metric
# classification_report - Detailed classification metrics
# precision_recall_fscore_support - Precision, recall, F1 scores
# confusion_matrix - Confusion matrix for evaluation

from sklearn.preprocessing import LabelEncoder  # Encode categorical labels as numbers
from sklearn.naive_bayes import MultinomialNB   # Naive Bayes classifier for text
from sklearn.pipeline import Pipeline  # ML pipeline creation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Ensemble methods
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity calculation
from sklearn.decomposition import TruncatedSVD  # Dimensionality reduction

# ================================================================================
# IMBALANCED LEARNING LIBRARIES
# ================================================================================
from imblearn.over_sampling import SMOTE  # Synthetic Minority Over-sampling Technique
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline that supports SMOTE

# ================================================================================
# GRADIENT BOOSTING LIBRARY
# ================================================================================
from xgboost import XGBClassifier  # Extreme Gradient Boosting classifier

# ================================================================================
# NATURAL LANGUAGE PROCESSING LIBRARIES
# ================================================================================
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # Common stopwords collection
from nltk.stem import WordNetLemmatizer  # Word lemmatization
from nltk.tokenize import word_tokenize  # Text tokenization

# Download NLTK resources (run first time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  # Download punkt tokenizer models

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')  # Download stopwords corpus

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')  # Download WordNet lexical database

warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

# --- ADD: dependency diagnostics to help troubleshoot imbalanced-learn warnings ---
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    # fallback for older Python / environments
    try:
        from pkg_resources import get_distribution as _get_dist
        def version(pkg):
            try:
                return _get_dist(pkg).version
            except Exception:
                raise PackageNotFoundError
    except Exception:
        def version(pkg):
            raise Exception("Cannot determine package versions on this Python environment.")
    class PackageNotFoundError(Exception):
        pass

def dependency_diagnostics():
    """Print versions of key libraries to help debug imbalanced-learn warnings."""
    print("\n[DIAG] Checking installed package versions:")
    pkgs = ['imbalanced-learn', 'scikit-learn', 'numpy', 'xgboost', 'pandas']
    for p in pkgs:
        try:
            v = version(p)
        except PackageNotFoundError:
            v = 'NOT INSTALLED'
        except Exception:
            v = 'UNKNOWN'
        print(f"[DIAG] {p}: {v}")
    print("[DIAG] If imbalanced-learn is installed but you still see warnings, you may have multiple Python environments or an incompatible scikit-learn version.")
    print("[DIAG] Quick fixes:")
    print("  • Ensure you run the same Python interpreter where you installed the package.")
    print("  • Upgrade/install compatible versions: pip install -U scikit-learn imbalanced-learn")
    print("  • If using conda: conda install -c conda-forge imbalanced-learn scikit-learn")
    print("")

# ================================================================================
# CONFIGURATION
# ================================================================================
DATASET_DIR = 'datasets'  # Directory for dataset files
MODEL_DIR = 'ml_models'   # Directory for basic models
ENHANCED_MODEL_DIR = 'enhanced_models'  # Directory for enhanced models

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)  # Create datasets directory
os.makedirs(MODEL_DIR, exist_ok=True)    # Create models directory
os.makedirs(ENHANCED_MODEL_DIR, exist_ok=True)  # Create enhanced models directory

# ================================================================================
# ENHANCED TEXT PREPROCESSING (ALL MODELS)
# ================================================================================
class AdvancedTextPreprocessor:
    """Advanced text preprocessing for all text-based models"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer for word normalization
        self.stop_words = set(stopwords.words('english'))  # Load English stopwords
        
    def clean_text(self, text):
        """Comprehensive text cleaning"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, numbers, and special characters using regex
        text = re.sub(r'http\S+|www\S+|[0-9]+', ' ', text)  # Remove URLs and numbers
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation and special chars
        
        # Tokenization - split text into individual words
        tokens = word_tokenize(text)
        
        # Lemmatization and stopword removal
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]  # Keep words longer than 2 chars
        
        return ' '.join(tokens)  # Join tokens back into string

# ================================================================================
# DATA LOADING FUNCTION
# ================================================================================
def load_datasets():
    """Load all required datasets (6 files)"""
    print("\n" + "=" * 80)
    print("AI Career Recommender - ENHANCED MODEL TRAINING STARTED")
    print("=" * 80)
    print("[DATA] Loading datasets...")

    try:
        # Core datasets for Recommender Logic
        career_df = pd.read_csv(f'{DATASET_DIR}/career_dataset.csv')  # Career information
        skills_df = pd.read_csv(f'{DATASET_DIR}/skills_dataset.csv')  # Skills data
        market_df = pd.read_csv(f'{DATASET_DIR}/job_market.csv')      # Job market trends

        # Personality datasets
        personality_ocean_df = pd.read_csv(f'{DATASET_DIR}/personality.csv')  # OCEAN personality traits
        personality_mbti_df = pd.read_csv(f'{DATASET_DIR}/mbti_training_data.csv')  # MBTI personality data
        
        # Resume dataset for Classifier training
        resume_df = pd.read_csv(f'{DATASET_DIR}/Resume.csv')  # Resume text data

        print(f"[OK] Career dataset: {career_df.shape}")
        print(f"[OK] Skills dataset: {skills_df.shape}")
        print(f"[OK] Personality (OCEAN) dataset: {personality_ocean_df.shape}")
        print(f"[OK] MBTI Training dataset: {personality_mbti_df.shape}")
        print(f"[OK] Resume dataset: {resume_df.shape}")

        return career_df, skills_df, personality_ocean_df, market_df, personality_mbti_df, resume_df

    except Exception as e:
        print(f"[ERROR] Error loading datasets: {e}")
        print("Please ensure all 6 files are in the 'datasets/' folder.")
        return None, None, None, None, None, None

# ================================================================================
# MODEL 1: ENHANCED RESUME CLASSIFIER
# ================================================================================
def train_enhanced_resume_classifier(resume_df):
    """ENHANCED Resume Classification Model with Ensemble and SMOTE"""
    print("\n" + "=" * 60)
    print("MODEL 1: ENHANCED RESUME CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Advanced text preprocessing
    print("[RESUME] Applying advanced text preprocessing...")
    resume_df['cleaned_text'] = resume_df['resume_text'].apply(preprocessor.clean_text)
    
    # Filter for valid data
    X = resume_df['cleaned_text']  # Features (text data)
    y = resume_df['category']      # Target (categories)

    # Filter out categories with too few samples
    min_samples = 5
    valid_categories = y.value_counts()[y.value_counts() >= min_samples].index
    resume_df_filtered = resume_df[y.isin(valid_categories)]
    
    X_filtered = resume_df_filtered['cleaned_text']
    y_filtered = resume_df_filtered['category']
    
    print(f"[RESUME] After filtering: {X_filtered.shape[0]} samples, {y_filtered.nunique()} categories")
    
    if len(X_filtered) < 100:
        print("[WARNING] Not enough filtered resume data to train. Skipping.")
        return None

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    
    # ENHANCED PIPELINE WITH ENSEMBLE AND SMOTE
    print("[RESUME] Building enhanced pipeline with ensemble and SMOTE...")
    
    resume_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(  # Convert text to TF-IDF features
            max_features=8000,      # Maximum number of features
            ngram_range=(1, 3),     # Use 1-3 word combinations
            stop_words='english',   # Remove English stopwords
            min_df=2,               # Ignore terms that appear in less than 2 documents
            max_df=0.9,             # Ignore terms that appear in more than 90% of documents
            sublinear_tf=True       # Use sublinear TF scaling
        )),
        ('smote', SMOTE(random_state=42)),  # Handle class imbalance
        ('classifier', XGBClassifier(  # XGBoost classifier
            random_state=42,
            eval_metric='logloss',   # Evaluation metric
            max_depth=6,             # Maximum tree depth
            learning_rate=0.1,       # Learning rate
            n_estimators=200         # Number of trees
        ))
    ])

    # Cross-validation for performance estimation
    print("[RESUME] Performing cross-validation...")
    cv_scores = cross_val_score(resume_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[RESUME] Cross-validation scores: {cv_scores}")
    print(f"[RESUME] Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    print("[RESUME] Training final ensemble model...")
    resume_pipeline.fit(X_train, y_train)

    # Comprehensive evaluation
    print("[RESUME] Comprehensive model evaluation...")
    y_pred = resume_pipeline.predict(X_test)  # Predict on test set
    
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"[RESUME] Final Test Accuracy: {accuracy:.3f}")
    print(f"[RESUME] Precision: {precision:.3f}")
    print(f"[RESUME] Recall: {recall:.3f}")
    print(f"[RESUME] F1-Score: {f1:.3f}")
    
    # Save enhanced model with metadata
    model_metadata = {
        'model': resume_pipeline,
        'preprocessor': preprocessor,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_scores': cv_scores.tolist(),
        'training_date': datetime.now().isoformat(),
        'feature_count': 8000,
        'classes': list(y_filtered.unique())
    }
    
    model_path = f'{ENHANCED_MODEL_DIR}/enhanced_resume_classifier.pkl'
    joblib.dump(model_metadata, model_path)  # Save model to disk
    print(f"[SAVE] Enhanced Resume Classifier saved to: {model_path}")
    
    return model_metadata

# ================================================================================
# MODEL 2: ENHANCED PERSONALITY CLASSIFIER (MBTI)
# ================================================================================
def train_enhanced_personality_classifier(personality_mbti_df):
    """ENHANCED MBTI Personality Classifier with Advanced Features"""
    print("\n" + "=" * 60)
    print("MODEL 2: ENHANCED PERSONALITY CLASSIFIER (MBTI)")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Advanced text preprocessing
    print("[PERSONALITY] Applying advanced text preprocessing...")
    personality_mbti_df['cleaned_posts'] = personality_mbti_df['posts'].apply(preprocessor.clean_text)
    
    X = personality_mbti_df['cleaned_posts']  # Features (social media posts)
    y = personality_mbti_df['type']           # Target (MBTI types)

    # Encode personality types as numerical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Convert text labels to numbers
    
    print(f"[PERSONALITY] Unique personality types: {len(le.classes_)}")
    print(f"[PERSONALITY] Classes: {le.classes_}")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ENHANCED ENSEMBLE PIPELINE
    print("[PERSONALITY] Building enhanced ensemble pipeline...")
    
    # Create ensemble of multiple classifiers
    ensemble_classifier = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),  # Logistic Regression
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),  # Random Forest
            ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))     # XGBoost
        ],
        voting='soft'  # Use soft voting for probability estimates
    )
    
    personality_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,     # Large feature set for text
            ngram_range=(1, 3),     # Use 1-3 word combinations
            stop_words='english',
            min_df=3,               # Ignore very rare terms
            max_df=0.85,
            sublinear_tf=True
        )),
        ('smote', SMOTE(random_state=42)),  # Handle class imbalance
        ('classifier', ensemble_classifier)  # Use ensemble classifier
    ])

    # Cross-validation
    print("[PERSONALITY] Performing cross-validation...")
    cv_scores = cross_val_score(personality_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[PERSONALITY] Cross-validation scores: {cv_scores}")
    print(f"[PERSONALITY] Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    print("[PERSONALITY] Training final ensemble model...")
    personality_pipeline.fit(X_train, y_train)

    # Comprehensive evaluation
    print("[PERSONALITY] Comprehensive model evaluation...")
    y_pred = personality_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"[PERSONALITY] Final Test Accuracy: {accuracy:.3f}")
    print(f"[PERSONALITY] Precision: {precision:.3f}")
    print(f"[PERSONALITY] Recall: {recall:.3f}")
    print(f"[PERSONALITY] F1-Score: {f1:.3f}")
    
    # Save enhanced model with metadata
    model_metadata = {
        'model': personality_pipeline,
        'preprocessor': preprocessor,
        'label_encoder': le,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_scores': cv_scores.tolist(),
        'training_date': datetime.now().isoformat(),
        'feature_count': 10000,
        'classes': le.classes_.tolist()
    }
    
    model_path = f'{ENHANCED_MODEL_DIR}/enhanced_personality_classifier.pkl'
    joblib.dump(model_metadata, model_path)
    print(f"[SAVE] Enhanced Personality Classifier saved to: {model_path}")
    
    return model_metadata

# ================================================================================
# MODEL 3: ENHANCED TEXT VECTORIZER
# ================================================================================
def train_enhanced_text_vectorizer(career_df, skills_df, personality_mbti_df, resume_df):
    """ENHANCED Global Text Vectorizer with Advanced Features"""
    print("\n" + "=" * 60)
    print("MODEL 3: ENHANCED TEXT VECTORIZER")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Collect text data from all sources
    text_data = []
    
    # Career descriptions
    if 'description' in career_df.columns:
        career_texts = career_df['description'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(career_texts.tolist())
    
    # Skill names
    if 'skill_name' in skills_df.columns:
        skill_texts = skills_df['skill_name'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(skill_texts.tolist())
    
    # Personality posts
    if 'posts' in personality_mbti_df.columns:
        personality_texts = personality_mbti_df['posts'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(personality_texts.tolist())
    
    # Resume texts
    if 'resume_text' in resume_df.columns:
        resume_texts = resume_df['resume_text'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(resume_texts.tolist())
    
    # Filter out very short texts
    text_data = [t for t in text_data if len(t.strip()) > 10]
    
    print(f"[VECTORIZER] Total text samples: {len(text_data)}")
    print(f"[VECTORIZER] Sample texts: {text_data[:2]}")

    # ENHANCED TF-IDF VECTORIZER
    print("[VECTORIZER] Training enhanced TF-IDF vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=15000,     # Large vocabulary
        ngram_range=(1, 3),     # Capture phrases
        stop_words='english',
        min_df=2,               # Ignore very rare terms
        max_df=0.9,             # Ignore very common terms
        sublinear_tf=True,      # Sublinear TF scaling
        use_idf=True,           # Use IDF weighting
        smooth_idf=True         # Smooth IDF weights
    )
    
    # Fit vectorizer on all text data
    vectorizer.fit(text_data)
    
    print(f"[VECTORIZER] Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # DIMENSIONALITY REDUCTION WITH TruncatedSVD
    print("[VECTORIZER] Applying dimensionality reduction...")
    
    # Transform text data to TF-IDF vectors
    text_vectors = vectorizer.transform(text_data)
    
    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=500, random_state=42)  # Reduce to 500 components
    reduced_vectors = svd.fit_transform(text_vectors)
    
    print(f"[VECTORIZER] Original dimensions: {text_vectors.shape[1]}")
    print(f"[VECTORIZER] Reduced dimensions: {reduced_vectors.shape[1]}")
    print(f"[VECTORIZER] Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Save enhanced vectorizer with metadata
    vectorizer_metadata = {
        'vectorizer': vectorizer,
        'svd': svd,
        'preprocessor': preprocessor,
        'vocabulary_size': len(vectorizer.vocabulary_),
        'feature_names': vectorizer.get_feature_names_out().tolist(),
        'explained_variance': svd.explained_variance_ratio_.sum(),
        'training_date': datetime.now().isoformat()
    }
    
    vectorizer_path = f'{ENHANCED_MODEL_DIR}/enhanced_vectorizer.pkl'
    joblib.dump(vectorizer_metadata, vectorizer_path)
    print(f"[SAVE] Enhanced Text Vectorizer saved to: {vectorizer_path}")
    
    return vectorizer_metadata

# ================================================================================
# MODEL 4: ENHANCED SKILL EXTRACTOR
# ================================================================================
def train_enhanced_skill_extractor(skills_df):
    """ENHANCED Rule-Based Skill Extractor with Advanced Matching"""
    print("\n" + "=" * 60)
    print("MODEL 4: ENHANCED SKILL EXTRACTOR")
    print("=" * 60)
    
    if 'skill_name' not in skills_df.columns or 'category' not in skills_df.columns:
        print("[ERROR] Skills DF missing essential columns. Skipping Skill Extractor.")
        return None

    # Create comprehensive skills mapping
    skills_mapping = {}
    category_mapping = {}
    proficiency_mapping = {}
    
    print("[SKILLS] Building enhanced skills database...")
    
    for _, row in skills_df.iterrows():
        skill_name = str(row['skill_name']).strip().lower()
        category = str(row['category'])
        
        # Add main skill
        skills_mapping[skill_name] = category
        
        # Create variations for better matching
        words = skill_name.split()
        if len(words) > 1:
            # Add acronym
            acronym = ''.join([word[0] for word in words if len(word) > 0])
            if acronym not in skills_mapping:
                skills_mapping[acronym] = category
            
            # Add without common suffixes
            if skill_name.endswith('s') and skill_name[:-1] not in skills_mapping:
                skills_mapping[skill_name[:-1]] = category
        
        # Store category and proficiency information
        category_mapping[skill_name] = category
        if 'proficiency_levels' in skills_df.columns:
            proficiency_mapping[skill_name] = str(row['proficiency_levels'])

    print(f"[SKILLS] Total skills in database: {len(skills_mapping)}")
    print(f"[SKILLS] Sample skills: {list(skills_mapping.keys())[:5]}")
    
    # Save enhanced skill extractor
    extractor_data = {
        'skills_mapping': skills_mapping,
        'category_mapping': category_mapping,
        'proficiency_mapping': proficiency_mapping,
        'total_skills': len(skills_mapping),
        'created_date': datetime.now().isoformat()
    }

    extractor_path = f'{ENHANCED_MODEL_DIR}/enhanced_skill_extractor.pkl'
    joblib.dump(extractor_data, extractor_path)
    print(f"[SAVE] Enhanced Skill Extractor saved to: {extractor_path}")

    return extractor_data

# ================================================================================
# MODEL 5: ENHANCED CONFIDENCE SCORER (CAREER RECOMMENDER CORE)
# ================================================================================
def train_enhanced_confidence_scorer(career_df, vectorizer_metadata):
    """ENHANCED Confidence Scoring System for Career Recommendations"""
    print("\n" + "=" * 60)
    print("MODEL 5: ENHANCED CONFIDENCE SCORER")
    print("=" * 60)
    
    # Extract components from vectorizer metadata
    vectorizer = vectorizer_metadata['vectorizer']
    svd = vectorizer_metadata['svd']
    preprocessor = vectorizer_metadata['preprocessor']
    
    # Create enhanced career profiles
    print("[SCORER] Creating enhanced career profiles...")
    
    career_profiles = []
    career_details = []
    
    for _, career in career_df.iterrows():
        # Combine multiple text fields for comprehensive profile
        profile_text = ""
        
        if 'career_name' in career_df.columns:
            profile_text += f" {career['career_name']}"
        if 'description' in career_df.columns:
            profile_text += f" {career['description']}"
        if 'required_skills' in career_df.columns:
            profile_text += f" {career['required_skills']}"
        if 'domain' in career_df.columns:
            profile_text += f" {career['domain']}"
        
        # Clean and preprocess
        cleaned_text = preprocessor.clean_text(profile_text)
        career_profiles.append(cleaned_text)
        
        # Store career details
        career_details.append({
            'career_id': career['career_id'] if 'career_id' in career_df.columns else _,
            'career_name': career['career_name'] if 'career_name' in career_df.columns else "Unknown",
            'domain': career['domain'] if 'domain' in career_df.columns else "Unknown",
            'average_salary': career['average_salary'] if 'average_salary' in career_df.columns else 0,
            'job_growth_rate': career['job_growth_rate'] if 'job_growth_rate' in career_df.columns else 0
        })
    
    print(f"[SCORER] Processed {len(career_profiles)} career profiles")
    
    # Transform career profiles to vector space
    print("[SCORER] Transforming career profiles to vector space...")
    
    career_vectors_tfidf = vectorizer.transform(career_profiles)
    career_vectors_reduced = svd.transform(career_vectors_tfidf)
    
    print(f"[SCORER] Career vectors shape: {career_vectors_reduced.shape}")
    
    # Calculate career similarity matrix
    print("[SCORER] Calculating career similarity matrix...")
    career_similarity = cosine_similarity(career_vectors_reduced)
    
    # Save enhanced confidence scorer
    scorer_data = {
        'vectorizer': vectorizer,
        'svd': svd,
        'preprocessor': preprocessor,
        'career_vectors': career_vectors_reduced,
        'career_similarity': career_similarity,
        'career_details': career_details,
        'career_profiles': career_profiles,
        'training_date': datetime.now().isoformat(),
        'total_careers': len(career_profiles)
    }
    
    scorer_path = f'{ENHANCED_MODEL_DIR}/enhanced_confidence_scorer.pkl'
    joblib.dump(scorer_data, scorer_path)
    print(f"[SAVE] Enhanced Confidence Scorer saved to: {scorer_path}")
    
    return scorer_data

# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================
def main():
    """Main training function to orchestrate all enhanced model training"""
    print("\n" + "=" * 80)
    print("AI CAREER RECOMMENDER - ENHANCED MODEL TRAINING")
    print("=" * 80)
    
    # DIAGNOSTIC: print versions to help debug warnings about imbalanced-learn
    try:
        dependency_diagnostics()
    except Exception as e:
        print(f"[DIAG] Could not run dependency diagnostics: {e}")

    # Load datasets
    datasets = load_datasets()
    if any(df is None for df in datasets):
        print("\n[FATAL ERROR] Data loading failed. Training stopped.")
        return

    career_df, skills_df, personality_ocean_df, market_df, personality_mbti_df, resume_df = datasets
    
    trained_models = {}
    
    try:
        # MODEL 1: Enhanced Resume Classifier
        resume_model = train_enhanced_resume_classifier(resume_df)
        trained_models['resume_classifier'] = resume_model
        
        # MODEL 2: Enhanced Personality Classifier
        personality_model = train_enhanced_personality_classifier(personality_mbti_df)
        trained_models['personality_classifier'] = personality_model
        
        # MODEL 3: Enhanced Text Vectorizer
        vectorizer_model = train_enhanced_text_vectorizer(career_df, skills_df, personality_mbti_df, resume_df)
        trained_models['text_vectorizer'] = vectorizer_model
        
        # MODEL 4: Enhanced Skill Extractor
        skill_extractor = train_enhanced_skill_extractor(skills_df)
        trained_models['skill_extractor'] = skill_extractor
        
        # MODEL 5: Enhanced Confidence Scorer (Requires vectorizer)
        if vectorizer_model is not None:
            confidence_scorer = train_enhanced_confidence_scorer(career_df, vectorizer_model)
            trained_models['confidence_scorer'] = confidence_scorer
        
        # Final summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY - ENHANCED MODELS")
        print("=" * 80)
        
        for model_name, model_data in trained_models.items():
            if model_data and 'accuracy' in model_data:
                print(f"✓ {model_name.upper()}: Accuracy = {model_data['accuracy']:.3f}")
            elif model_data:
                print(f"✓ {model_name.upper()}: Trained successfully")
        
        print(f"\n[SUCCESS] All {len(trained_models)} enhanced models trained and saved!")
        print(f"Models saved in: {ENHANCED_MODEL_DIR}/")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()