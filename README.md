# Disaster-Prediction-from-Tweets-Using-Deep-Learning-with-BERT

## Project Description
This project focuses on classifying tweets as disaster-related or not using **BERT (Bidirectional Encoder Representations from Transformers)**, a state-of-the-art deep learning model for Natural Language Processing (NLP). The goal is to build a scalable system that can analyze real-time tweets to identify potential disasters, enabling faster response and decision-making.

- **Problem**: Social media platforms like Twitter are flooded with information during disasters, but not all tweets are relevant. Manually identifying disaster-related tweets is time-consuming and inefficient.
- **Solution**: Built a BERT-based deep learning model to automatically classify tweets as disaster-related or not, achieving **83.5% accuracy** on the test set.
- **Impact**: This system can be deployed to monitor social media in real-time, helping organizations respond quickly to disasters and save lives.

-------

## Project Specifications

- **Dataset:** 10,000+ social media posts containing emergency-related terms (e.g., "emergency", "evacuation", "wildfire").
- **Variables:** id, keyword, location, text, target (1 = Emergency, 0 = Non-emergency).
- **Limitations:** 33% incomplete location data, 0.8% missing keyword entries.

## Data Analysis & Variable Development:
- **Text Preprocessing:** Link elimination, punctuation standardization, noise word filtering.
- **Language Processing:** N-grams, text metrics, character statistics, hashtag analysis, user references.


## Key Features
- **BERT Fine-Tuning**: Fine-tuned the pre-trained BERT model on a dataset of 7,613 labeled tweets using PyTorch.
- **Data Preprocessing**: Cleaned and preprocessed raw tweet data by removing punctuation, numbers, and special characters.
- **Model Optimization**: Implemented early stopping and used cross-entropy loss with AdamW optimizer for efficient training.
- **Scalable Deployment**: Optimized the model for real-time inference using Google Cloud AI services.
- **Interactive Dashboard**: Created an interactive dashboard using Streamlit for real-time monitoring and decision-making.


## Model Selection & Evaluation:

| Model        | Precision | Recall | Accuracy | F1-Score |
|-------------|-----------|--------|-----------|---------|
| **BERT**   | **86%**  | **84%** | **85%**  | **86%** |
| **Naïve Bayes** | 82% | 70% | 56% | 75% |


BERT outperforms Naïve Bayes by 11% in F1-score, making it the final choice.


-----

## Programming Languages & Libraries
- **Python –** Primary language for data processing & modeling
- **Pandas & NumPy –** Data manipulation & numerical computation
- **Matplotlib & Seaborn –** Data visualization & EDA
- **Scikit-learn –** Machine learning (Naïve Bayes, evaluation metrics)
- **NLTK & SpaCy –** Text preprocessing & tokenization
- **Transformers (Hugging Face) –** BERT-based deep learning model
- **PyTorch / TensorFlow –** Model training & fine-tuning

## Development & Deployment
- **Jupyter Notebook –** Exploratory Data Analysis (EDA) & model experiments
- **VS Code –** Primary development environment
- **Git & GitHub –** Version control & project repository
- **Docker (Future Scope) –** Containerization for deployment
- **FastAPI/Flask (Future Scope) –** API development for model serving

----

## License
This project is licensed under the MIT License. You are free to use, modify, and share this project with proper attribution.

