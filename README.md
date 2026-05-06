# Fake_Data_Detection
# A Modular Machine Learning Framework for Detecting Fake, Synthetic, and Anomalous Data

## Overview

This project presents a modular machine learning framework designed to detect fake, synthetic, manipulated, and anomalous data across multiple domains. The system primarily focuses on AI-generated and synthetic text detection using Natural Language Processing (NLP) and Machine Learning techniques.

The framework is designed with a plug-and-play modular architecture, allowing seamless integration of new datasets, models, and detection modules without rebuilding the complete system.

Developed as a Minor Project for the Department of Computer Science and Engineering at The National Institute of Engineering (NIE), Mysuru.

---

## Abstract

With the rapid growth of AI-generated content and synthetic data, distinguishing authentic information from fabricated data has become increasingly difficult. Fake and manipulated data can reduce trust in analytical systems and negatively impact machine learning models.

This project proposes a modular framework capable of:

* Extracting linguistic and statistical features
* Detecting synthetic and anomalous data
* Performing authenticity verification using NLP
* Classifying data as real or fake using machine learning models

The framework achieved approximately 97% accuracy on the IMDB movie review dataset using human-written and GPT-generated reviews.

---

## Features

* Fake text detection
* Synthetic review classification
* Modular ML architecture
* NLP-based feature extraction
* Logistic Regression classifier
* Real-time prediction support
* Streamlit/Flask based interface
* Extensible plug-and-play design

---

## Technologies Used

### Programming Languages

* Python

### Libraries & Frameworks

* Scikit-learn
* Pandas
* NumPy
* Transformers
* Torch
* tqdm
* joblib

### Tools

* VS Code
* Jupyter Notebook

---

## System Architecture

The system consists of four major modules:

1. **Data Input Module**

   * Loads and preprocesses datasets

2. **Feature Extraction Module**

   * Extracts linguistic and statistical features

3. **Model Training Module**

   * Trains machine learning classifiers

4. **Detection Module**

   * Predicts whether input data is real or synthetic

---

## Dataset

The project uses:

* IMDB Movie Review Dataset
* GPT-generated synthetic reviews

Note:
Large datasets and trained models are excluded from GitHub because of GitHub file size limitations.

---

## Performance Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 97%    |
| Precision | 0.97   |
| Recall    | 0.97   |
| F1-Score  | 0.97   |
| ROC-AUC   | 0.9799 |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Srirang050/Fake_Data_Detection.git
```

### Navigate to Project Folder

```bash
cd Fake_Data_Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python app.py
```

---

## Project Structure

```bash
Fake_Data_Detection/
│
├── api/
├── data/
├── gatekeeper/
├── scripts/
├── ui/
├── app.py
├── text_detector.py
├── requirements.txt
└── README.md
```

---

## Future Enhancements

* Support for image and audio fake detection
* Transformer-based advanced detection models
* Real-time streaming detection
* Cloud deployment support
* Improved explainability and visualization

---

## Contributors

* Sriranga V
* Vignap Sanjoth
* Yajnith K
* Yashas Gowda SH

---

## Institution

Department of Computer Science & Engineering
The National Institute of Engineering (NIE), Mysuru

---

## License

This project is developed for academic and educational purposes.

# Fake_Data_Detection