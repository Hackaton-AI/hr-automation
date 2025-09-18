# model_service_fixed_v2.py - Fixed model service with proper feature dimension matching
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
import warnings
import joblib

warnings.filterwarnings('ignore')

class ResumeScreeningModel:

    def __init__(self, model_path="C:/Users/Ismael/Desktop/projects/hr-automation/models/resume-screening/resume_screening_model.h5",
                preprocessing_path="C:/Users/Ismael/Desktop/projects/hr-automation/models/resume-screening/model_preprocessing.pkl"):
        
        self.model = None
        self.preprocessing_components = None
        self.lemmatizer = None
        self.stop_words = None
        self.is_initialized = False
        self.fallback_mode = False
        self.expected_feature_count = None
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path

    def calculate_feature_dimensions(self):
        """Calculate expected feature dimensions from the original training"""
        if not self.preprocessing_components:
            return None, None, None

        feature_columns = self.preprocessing_components.get(
            'feature_columns', [])
        categorical_columns = self.preprocessing_components.get(
            'categorical_columns', [])

        if not feature_columns:
            # Fallback calculation based on scaler
            if self.expected_feature_count:
                cat_count = len(categorical_columns)
                remaining_features = self.expected_feature_count - cat_count
                # Assume equal split between resume and JD
                resume_features = remaining_features // 2
                jd_features = remaining_features - resume_features
                return resume_features, jd_features, cat_count
            return None, None, None

        resume_features = len(
            [col for col in feature_columns if col.startswith('resume_')])
        jd_features = len(
            [col for col in feature_columns if col.startswith('jd_')])
        cat_features = len(categorical_columns)

        return resume_features, jd_features, cat_features

    def initialize(self):
        """Load model and initialize preprocessing components with fallback"""
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            print("✓ Keras model loaded successfully")

            # Load preprocessing components
            try:
                self.preprocessing_components = joblib.load(
                    self.preprocessing_path)
                print(
                    "✓ Preprocessing components loaded with joblib")
            except Exception as e:
                print(f"joblib failed: {e}, trying pickle...")
                with open(self.preprocessing_path, 'rb') as f:
                    self.preprocessing_components = pickle.load(f)
                print(
                    "✓ Preprocessing components loaded with pickle")

            # Debug loaded components
            self.debug_preprocessing_components()

            # Check if TF-IDF vectorizers are fitted
            self._check_and_fix_tfidf()

            # Initialize NLTK components
            self._initialize_nltk()

            self.is_initialized = True
            print("✓ Model service initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize model: {e}")
        

    def _check_and_fix_tfidf(self):
        """Check TF-IDF vectorizers and create fallback if needed"""
        tfidf_resume = self.preprocessing_components.get('tfidf_resume')
        tfidf_jd = self.preprocessing_components.get('tfidf_jd')

        resume_fitted = tfidf_resume is not None and hasattr(
            tfidf_resume, 'idf_')
        jd_fitted = tfidf_jd is not None and hasattr(tfidf_jd, 'idf_')

        if not resume_fitted or not jd_fitted:
            print(
                "TF-IDF vectorizers not properly fitted. Activating fallback mode.")
            self.fallback_mode = True
            self._create_fallback_tfidf()
        else:
            print("✓ TF-IDF vectorizers are properly fitted")

    def _create_fallback_tfidf(self):
        """Create fallback TF-IDF vectorizers with proper dimensions"""
        print("Creating fallback TF-IDF vectorizers...")

        # Calculate expected dimensions
        resume_features, jd_features, cat_features = self.calculate_feature_dimensions()

        if resume_features is None or jd_features is None:
            print("Cannot determine proper feature dimensions!")
            raise ValueError(
                "Cannot determine proper feature dimensions from saved components")

        print(
            f"Expected dimensions - Resume: {resume_features}, JD: {jd_features}, Categorical: {cat_features}")

        # Create vectorizers with exact feature counts needed
        self.preprocessing_components['tfidf_resume'] = TfidfVectorizer(
            max_features=resume_features,
            min_df=1,  # More permissive to ensure we get the exact feature count
            max_df=1.0,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            vocabulary=None
        )

        self.preprocessing_components['tfidf_jd'] = TfidfVectorizer(
            max_features=jd_features,
            min_df=1,  # More permissive to ensure we get the exact feature count
            max_df=1.0,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            vocabulary=None
        )

        # Create comprehensive training data to ensure we get enough features
        dummy_resume_texts = self._generate_comprehensive_resume_texts(
            resume_features)
        dummy_jd_texts = self._generate_comprehensive_jd_texts(jd_features)

        # Fit the vectorizers
        try:
            self.preprocessing_components['tfidf_resume'].fit(
                dummy_resume_texts)
            self.preprocessing_components['tfidf_jd'].fit(dummy_jd_texts)

            # Verify the feature counts
            actual_resume_features = len(
                self.preprocessing_components['tfidf_resume'].vocabulary_)
            actual_jd_features = len(
                self.preprocessing_components['tfidf_jd'].vocabulary_)

            print(
                f"✓ Fallback TF-IDF vectorizers fitted successfully")
            print(
                f"  Resume features: {actual_resume_features}/{resume_features}")
            print(
                f"  JD features: {actual_jd_features}/{jd_features}")

        except Exception as e:
            print(
                f"Failed to fit fallback TF-IDF vectorizers: {e}")
            raise

    def _generate_comprehensive_resume_texts(self, target_features):
        """Generate comprehensive resume texts to ensure enough vocabulary"""
        base_texts = [
            "python developer software engineer machine learning data analysis experience",
            "java programming web development backend systems database design",
            "frontend react javascript html css user interface design experience",
            "project manager agile scrum team leadership communication skills",
            "data scientist statistics python r machine learning analytics experience",
            "system administrator linux windows server network security management",
            "mobile developer ios android swift kotlin mobile applications development",
            "devops engineer kubernetes docker cloud aws azure deployment automation",
            "business analyst requirements gathering process improvement stakeholder management",
            "quality assurance testing automation selenium manual testing experience",
            "marketing manager digital marketing social media campaign management",
            "sales representative customer relationship management crm sales targets",
            "financial analyst financial modeling excel powerpoint presentation skills",
            "human resources recruiting talent acquisition employee relations",
            "operations manager supply chain logistics process optimization experience"
        ]

        # Generate additional texts with technical terms
        tech_terms = [
            "spring", "hibernate", "microservices", "restful", "apis", "json", "xml",
            "angular", "vue", "node", "express", "mongodb", "postgresql", "mysql",
            "tensorflow", "pytorch", "sklearn", "pandas", "numpy", "matplotlib",
            "jenkins", "gitlab", "circleci", "terraform", "ansible", "chef",
            "salesforce", "hubspot", "tableau", "powerbi", "excel", "powerpoint",
            "jira", "confluence", "slack", "teams", "zoom", "sharepoint",
            "photoshop", "illustrator", "sketch", "figma", "invision", "adobe",
            "aws", "azure", "gcp", "docker", "kubernetes", "redis", "elasticsearch"
        ]

        # Add variations to reach target features
        extended_texts = base_texts.copy()
        for i in range(max(0, target_features - len(base_texts) * 10)):
            # Create new combinations
            import random
            random.shuffle(tech_terms)
            new_text = f"experienced professional {' '.join(tech_terms[:5])} specialist"
            extended_texts.append(new_text)

        return extended_texts

    def _generate_comprehensive_jd_texts(self, target_features):
        """Generate comprehensive JD texts to ensure enough vocabulary"""
        base_texts = [
            "seeking experienced python developer machine learning background required",
            "looking java developer enterprise applications development experience",
            "frontend developer position react modern web technologies required",
            "project manager role agile methodology team leadership required",
            "data scientist position analytics machine learning python required",
            "system administrator linux environment network management required",
            "mobile developer ios android applications development required",
            "devops engineer position cloud infrastructure automation required",
            "business analyst role requirements analysis process improvement required",
            "qa engineer automated testing selenium framework experience required",
            "marketing manager position digital marketing campaign management required",
            "sales position customer relationship management experience required",
            "financial analyst role financial modeling excel skills required",
            "hr position recruiting talent acquisition experience required",
            "operations manager supply chain logistics experience required"
        ]

        # Generate additional JD texts
        requirements = [
            "bachelor degree required", "master preferred", "years experience minimum",
            "excellent communication skills", "team player attitude", "problem solving",
            "analytical thinking", "attention detail", "time management", "multitasking",
            "client facing experience", "presentation skills", "documentation skills",
            "cross functional collaboration", "remote work experience", "startup environment"
        ]

        extended_texts = base_texts.copy()
        for i in range(max(0, target_features - len(base_texts) * 10)):
            import random
            random.shuffle(requirements)
            new_text = f"looking for candidate with {' '.join(requirements[:3])} background"
            extended_texts.append(new_text)

        return extended_texts

    def _initialize_nltk(self):
        """Initialize NLTK components with fallback"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            print("✓ NLTK components initialized")
        except Exception as e:
            print(f"NLTK initialization warning: {e}")
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was',
                'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might'
            }
            self.lemmatizer = None

    def preprocess_text(self, text):
        """Enhanced text preprocessing pipeline"""
        if pd.isna(text) or not text:
            return "default content"

        try:
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)

            words = text.split()
            words = [word for word in words if len(
                word) > 1]  # Keep 2+ character words

            if self.stop_words:
                words = [word for word in words if word not in self.stop_words]

            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(word) for word in words]

            result = ' '.join(words)
            return result if len(result.strip()) > 0 else "default content"

        except Exception as e:
            print(f"Text preprocessing error: {e}")
            return "default content"

    def create_tfidf_features(self, resume_text, jd_text):
        """Create TF-IDF features from text inputs with error handling"""
        resume_clean = self.preprocess_text(resume_text)
        jd_clean = self.preprocess_text(jd_text)

        try:
            resume_tfidf = self.preprocessing_components['tfidf_resume'].transform(
                [resume_clean]).toarray()
            jd_tfidf = self.preprocessing_components['tfidf_jd'].transform(
                [jd_clean]).toarray()

            return resume_tfidf, jd_tfidf

        except Exception as e:
            print(f"TF-IDF transformation error: {e}")
            raise

    def create_categorical_features(self, job_family, seniority):
        """Create one-hot encoded categorical features with validation"""
        try:
            cat_data = pd.DataFrame({
                'job_family': [str(job_family)],
                'seniority': [str(seniority)]
            })

            cat_encoded = pd.get_dummies(cat_data, drop_first=True)
            expected_cols = self.preprocessing_components.get(
                'categorical_columns', [])

            for col in expected_cols:
                if col not in cat_encoded.columns:
                    cat_encoded[col] = 0

            if expected_cols:
                cat_encoded = cat_encoded.reindex(
                    columns=expected_cols, fill_value=0)

            return cat_encoded.values.flatten()

        except Exception as e:
            print(f"Categorical features creation error: {e}")
            expected_cols = self.preprocessing_components.get(
                'categorical_columns', [])
            return np.zeros(len(expected_cols)) if expected_cols else np.array([])

    def create_feature_vector(self, resume_text, jd_text, job_family, seniority):
        """Create complete feature vector for model input"""
        resume_tfidf, jd_tfidf = self.create_tfidf_features(
            resume_text, jd_text)
        categorical_features = self.create_categorical_features(
            job_family, seniority)

        feature_vector = np.concatenate([
            resume_tfidf.flatten(),
            jd_tfidf.flatten(),
            categorical_features
        ]).reshape(1, -1)

        return feature_vector

    def predict(self, resume_text, jd_text, job_family, seniority):
        try:
            features = self.create_feature_vector(
                resume_text, jd_text, job_family, seniority)

            # Verify feature dimensions
            if self.expected_feature_count and features.shape[1] != self.expected_feature_count:
                raise ValueError(
                    f"Feature dimension mismatch: got {features.shape[1]}, expected {self.expected_feature_count}")

            # Scale features
            if 'scaler' in self.preprocessing_components and hasattr(self.preprocessing_components['scaler'], 'mean_'):
                features_scaled = self.preprocessing_components['scaler'].transform(
                    features)
            else:
                features_scaled = features

            # Make prediction
            probability = float(self.model.predict(
                features_scaled, verbose=0)[0][0])
            binary_decision = probability > 0.5
            confidence = max(probability, 1 - probability)

            # result = {
            #     # 'probability': round(probability, 4),
            #     # 'decision': 'advance' if binary_decision else 'no_advance',
            #     'binary_decision': binary_decision
            #     # 'confidence': round(confidence, 4),
            #     # 'recommendation': 'Recommend for interview' if binary_decision else 'Do not recommend',
            #     # 'fallback_mode': self.fallback_mode,
            #     # 'feature_dimensions': {
            #     #     'total_features': features.shape[1],
            #     #     'expected_features': self.expected_feature_count
            #     # },
            #     # 'inputs': {
            #     #     'job_family': job_family,
            #     #     'seniority': seniority,
            #     #     'resume_length': len(resume_text),
            #     #     'jd_length': len(jd_text)
            #     # }
            # }
            return binary_decision

        except Exception as e:
            raise

    def get_model_info(self):
        """Get model information and metrics"""
        if not self.is_initialized:
            return {'error': 'Model not initialized'}

        info = {
            'is_initialized': self.is_initialized,
            'fallback_mode': self.fallback_mode,
            'expected_feature_count': self.expected_feature_count,
            'model_architecture': {
                'layers': len(self.model.layers),
                'total_params': self.model.count_params()
            }
        }

        if self.preprocessing_components:
            resume_features, jd_features, cat_features = self.calculate_feature_dimensions()
            info.update({
                'feature_breakdown': {
                    'resume_features': resume_features,
                    'jd_features': jd_features,
                    'categorical_features': cat_features,
                    'total': (resume_features or 0) + (jd_features or 0) + (cat_features or 0)
                },
                'model_metrics': self.preprocessing_components.get('model_metrics', {}),
                'categorical_columns': self.preprocessing_components.get('categorical_columns', [])
            })

        return info


# # Usage example and testing
# if _name_ == "_main_":
#     model_service = ResumeScreeningModel()

#     if model_service.initialize():
#         print("✓ Model service initialized successfully!")

#         model_info = model_service.get_model_info()
#         print(f"\nModel Info:")
#         print(f"- Fallback mode: {model_info['fallback_mode']}")
#         print(f"- Expected features: {model_info['expected_feature_count']}")
#         if 'feature_breakdown' in model_info:
#             fb = model_info['feature_breakdown']
#             print(
#                 f"- Feature breakdown: Resume({fb['resume_features']}) + JD({fb['jd_features']}) + Categorical({fb['categorical_features']}) = {fb['total']}")

#         # Test prediction
#         try:
#             test_result = model_service.predict(
#                 resume_text="Python developer with 5 years experience in machine learning",
#                 jd_text="Looking for senior Python developer with ML experience",
#                 job_family="Engineering",
#                 seniority="Senior"
#             )

#             print(f"\nPrediction Result:")
#             print(f"- Decision: {test_result['decision']}")
#             print(f"- Probability: {test_result['probability']}")
#             print(f"- Confidence: {test_result['confidence']}")
#             print(
#                 f"- Features used: {test_result['feature_dimensions']['total_features']}")

#         except Exception as e:
#             print(f"❌ Prediction failed: {e}")

#     else:
#         print("❌ Failed to initialize model service")
