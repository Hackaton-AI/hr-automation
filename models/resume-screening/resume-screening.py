# model_service.py - Complete model service with preprocessing
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import warnings
warnings.filterwarnings('ignore')


class ResumeScreeningModel:
    """
    Complete model service that handles preprocessing and prediction
    """

    def __init__(self, model_path="C:\\Users\\Hp Victus\\Downloads\\hr_automation\models\\resume-screening\\resume_screening_model.h5",
                 preprocessing_path="C:\\Users\\Hp Victus\\Downloads\\hr_automation\\models\\resume-screening\\model_preprocessing.pkl"):
        """Initialize the model service"""
        self.model = None
        self.preprocessing_components = None
        self.lemmatizer = None
        self.stop_words = None
        self.is_initialized = False

        self.model_path = model_path
        self.preprocessing_path = preprocessing_path

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """Load model and initialize preprocessing components"""
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info("✓ Keras model loaded successfully")

            # Load preprocessing components
            with open(self.preprocessing_path, 'rb') as f:
                self.preprocessing_components = pickle.load(f)
            self.logger.info("✓ Preprocessing components loaded successfully")

            # Initialize NLTK components
            self._initialize_nltk()

            self.is_initialized = True
            self.logger.info("✓ Model service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            return False

    def _initialize_nltk(self):
        """Initialize NLTK components with fallback"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.logger.info("✓ NLTK components initialized")
        except Exception as e:
            self.logger.warning(f"NLTK initialization warning: {e}")
            # Fallback - create basic stop words
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was',
                'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might'
            }
            self.lemmatizer = None

    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text) or not text:
            return ""

        try:
            # Convert to string and lowercase
            text = str(text).lower()

            # Remove non-alphabetical characters
            text = re.sub(r'\W', ' ', text)

            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)

            # Remove very short words (1-2 characters)
            text = re.sub(r'\b\w{1,2}\b', '', text)

            # Tokenize
            words = text.split()

            # Remove stop words and short words
            if self.stop_words:
                words = [word for word in words
                         if word not in self.stop_words and len(word) > 2]

            # Lemmatize if available
            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(word) for word in words]

            return ' '.join(words)

        except Exception as e:
            self.logger.warning(f"Text preprocessing error: {e}")
            return str(text).lower()  # Fallback to simple lowercase

    def create_tfidf_features(self, resume_text, jd_text):
        """
        Create TF-IDF features from text inputs

        Args:
            resume_text (str): Resume text
            jd_text (str): Job description text

        Returns:
            tuple: (resume_tfidf_array, jd_tfidf_array)
        """
        # Preprocess texts
        resume_clean = self.preprocess_text(resume_text)
        jd_clean = self.preprocess_text(jd_text)

        # Transform using fitted vectorizers
        resume_tfidf = self.preprocessing_components['tfidf_resume'].transform(
            [resume_clean]).toarray()
        jd_tfidf = self.preprocessing_components['tfidf_jd'].transform(
            [jd_clean]).toarray()

        return resume_tfidf, jd_tfidf

    def create_categorical_features(self, job_family, seniority):
        """
        Create one-hot encoded categorical features

        Args:
            job_family (str): Job family category
            seniority (str): Seniority level

        Returns:
            np.array: Encoded categorical features
        """
        # Create DataFrame with input values
        cat_data = pd.DataFrame({
            'job_family': [job_family],
            'seniority': [seniority]
        })

        # One-hot encode with drop_first=True (same as training)
        cat_encoded = pd.get_dummies(cat_data, drop_first=True)

        # Ensure all expected columns are present
        for col in self.preprocessing_components['categorical_columns']:
            if col not in cat_encoded.columns:
                cat_encoded[col] = 0

        # Reorder columns to match training data
        cat_encoded = cat_encoded.reindex(
            columns=self.preprocessing_components['categorical_columns'],
            fill_value=0
        )

        return cat_encoded.values.flatten()

    def create_feature_vector(self, resume_text, jd_text, job_family, seniority):
        """
        Create complete feature vector for model input

        Args:
            resume_text (str): Resume text
            jd_text (str): Job description text
            job_family (str): Job family category
            seniority (str): Seniority level

        Returns:
            np.array: Complete feature vector
        """
        # Create TF-IDF features
        resume_tfidf, jd_tfidf = self.create_tfidf_features(
            resume_text, jd_text)

        # Create categorical features
        categorical_features = self.create_categorical_features(
            job_family, seniority)

        # Combine all features
        feature_vector = np.concatenate([
            resume_tfidf.flatten(),
            jd_tfidf.flatten(),
            categorical_features
        ]).reshape(1, -1)

        return feature_vector

    def predict(self, resume_text, jd_text, job_family, seniority):
        """
        Make prediction for a single resume

        Args:
            resume_text (str): Resume text
            jd_text (str): Job description text  
            job_family (str): Job family category
            seniority (str): Seniority level

        Returns:
            dict: Prediction results
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Model not initialized. Call initialize() first.")

        try:
            # Create feature vector
            features = self.create_feature_vector(
                resume_text, jd_text, job_family, seniority)

            # Scale features
            features_scaled = self.preprocessing_components['scaler'].transform(
                features)

            # Make prediction
            probability = float(self.model.predict(
                features_scaled, verbose=0)[0][0])
            binary_decision = probability > 0.5
            confidence = max(probability, 1 - probability)

            # Prepare result
            result = {
                'probability': round(probability, 4),
                'decision': 'advance' if binary_decision else 'no_advance',
                'binary_decision': binary_decision,
                'confidence': round(confidence, 4),
                'recommendation': 'Recommend for interview' if binary_decision else 'Do not recommend',
                'inputs': {
                    'job_family': job_family,
                    'seniority': seniority,
                    'resume_length': len(resume_text),
                    'jd_length': len(jd_text),
                    'resume_word_count': len(resume_text.split()),
                    'jd_word_count': len(jd_text.split())
                }
            }

            self.logger.info(
                f"Prediction: {result['decision']} (prob: {probability:.3f})")
            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(self, candidates):
        """
        Make predictions for multiple candidates

        Args:
            candidates (list): List of candidate dictionaries

        Returns:
            list: List of prediction results
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Model not initialized. Call initialize() first.")

        results = []

        for i, candidate in enumerate(candidates):
            try:
                # Validate candidate data
                required_fields = ['resume_text',
                                   'jd_text', 'job_family', 'seniority']
                for field in required_fields:
                    if field not in candidate or candidate[field] is None:
                        raise ValueError(f'Missing or null field: {field}')

                # Make prediction
                result = self.predict(
                    candidate['resume_text'],
                    candidate['jd_text'],
                    candidate['job_family'],
                    candidate['seniority']
                )

                # Add candidate ID
                result['candidate_id'] = candidate.get('id', i)
                results.append(result)

            except Exception as e:
                results.append({
                    'candidate_id': candidate.get('id', i),
                    'error': str(e)
                })
                self.logger.error(f"Error processing candidate {i}: {e}")

        return results

    def get_model_info(self):
        """
        Get model information and metrics

        Returns:
            dict: Model information
        """
        if not self.is_initialized:
            return {'error': 'Model not initialized'}

        return {
            'model_metrics': self.preprocessing_components.get('model_metrics', {}),
            'feature_count': len(self.preprocessing_components['feature_columns']),
            'categorical_features': self.preprocessing_components['categorical_columns'],
            'tfidf_features': {
                'resume_features': len([col for col in self.preprocessing_components['feature_columns'] if col.startswith('resume_')]),
                'jd_features': len([col for col in self.preprocessing_components['feature_columns'] if col.startswith('jd_')])
            },
            'model_architecture': {
                'layers': len(self.model.layers),
                'total_params': self.model.count_params()
            },
            'is_initialized': self.is_initialized
        }


# Usage example and testing
if __name__ == "__main__":
    # Initialize model service
    model_service = ResumeScreeningModel()

    if model_service.initialize():
        print("✓ Model service initialized successfully!")

        # Test single prediction
        test_result = model_service.predict(
            resume_text="Python developer with 5 years experience in machine learning and web development",
            jd_text="Looking for senior Python developer with ML experience",
            job_family="Engineering",
            seniority="Senior"
        )

        print("\nTest prediction result:")
        print(f"Decision: {test_result['decision']}")
        print(f"Probability: {test_result['probability']}")
        print(f"Confidence: {test_result['confidence']}")

        # Test model info
        model_info = model_service.get_model_info()
        print(f"\nModel info:")
        print(f"Total features: {model_info['feature_count']}")
        print(
            f"Model accuracy: {model_info['model_metrics'].get('accuracy', 'N/A')}")

    else:
        print("❌ Failed to initialize model service")
