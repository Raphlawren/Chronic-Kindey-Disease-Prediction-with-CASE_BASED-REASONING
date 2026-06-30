# Chronic Kidney Disease Prediction with Case-Based Reasoning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Case--Based%20Reasoning-orange)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare%20AI-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-lightgrey)

## Overview

This project applies **Case-Based Reasoning (CBR)** to predict Chronic Kidney Disease outcomes using structured patient health records.

Unlike typical black-box machine learning classifiers, this project focuses on a more interpretable reasoning approach: instead of only returning a prediction, the system retrieves similar historical patient cases and uses them to infer the most likely outcome for a new patient.

The project includes:

* Exploratory data analysis of CKD patient records
* Data preparation for binary and multiclass prediction tasks
* A custom Case-Based Reasoning model implemented in Python
* Similarity-based patient case retrieval
* Feature weighting and optimization
* Binary prediction of CKD progression
* Multiclass classification of CKD stage
* Evaluation using classification reports, confusion matrices, and ROC-AUC

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis or medical decision-making.

---

## Problem Statement

Chronic Kidney Disease is a progressive condition where early risk identification can support better monitoring and intervention planning.

The goal of this project is to build an interpretable AI system that can answer two predictive questions:

1. **Binary classification:** Is a patient likely to experience CKD progression?
2. **Multiclass classification:** What CKD stage does the patient belong to?

The project uses Case-Based Reasoning to make predictions by comparing a new patient record with similar historical cases.

---

## Why Case-Based Reasoning?

Case-Based Reasoning is useful in healthcare-style problems because it mirrors how domain experts often reason:

> “This patient looks similar to previous patients with these characteristics, so their likely outcome may be similar.”

In this project, CBR provides:

* **Interpretability:** predictions are based on retrieved similar cases
* **Transparency:** similarity scores can be inspected
* **Flexibility:** supports numerical and categorical clinical features
* **Reasoning-based prediction:** uses past cases instead of relying only on learned coefficients or tree splits

---

## Dataset

The project uses a structured CKD dataset containing **1,138 patient records** and **23 clinical features**.

Examples of features used include:

* Age
* Sex
* Systolic pressure
* BMI
* Hemoglobin
* Albumin
* Creatinine
* eGFR
* Protein creatinine ratio
* CKD stage
* CKD risk
* Proteinuria
* Hypertension
* Diabetes
* Previous cardiovascular disease
* RAAS inhibitor usage
* Calcium channel blocker usage
* Diuretics usage
* CKD progression

---

## Project Structure

```text
.
├── README.md
└── Raphael Lawrence Farodoye COM804 Test/
    ├── cbr.py                    # Custom Case-Based Reasoning model
    ├── EDA.ipynb                 # Exploratory data analysis notebook
    ├── run_cbr_ckd.ipynb         # Model training and evaluation notebook
    ├── ckd.csv                   # CKD dataset
    ├── Binary_data.csv           # Processed binary classification dataset
    └── CHRONIC KIDNEY DISEASE.pdf
```

---

## Methodology

### 1. Exploratory Data Analysis

The EDA notebook investigates:

* Dataset shape and structure
* Missing values
* Clinical feature distributions
* Target variable distribution
* Numerical and categorical variables
* Relationships between CKD indicators and progression/stage outcomes

### 2. Data Preparation

The dataset was prepared for two prediction tasks:

#### Binary Classification

Target variable:

```text
CKD_Progression
```

Classes:

```text
0 = No Progression
1 = Progression
```

#### Multiclass Classification

Target variable:

```text
CKD_Stage
```

Classes:

```text
Stage 1
Stage 2
Stage 3
Stage 4
Stage 5
```

### 3. Case-Based Reasoning Model

The custom `CBR` class implements the main CBR workflow:

1. **Retrieve** similar historical cases
2. **Reuse** their solutions through voting
3. **Revise** predictions through evaluation
4. **Retain** cases in the case base

The model supports:

* Continuous feature similarity
* Categorical feature similarity
* Min-max normalization
* Gaussian similarity
* Feature weighting
* Top-k case retrieval
* Majority voting
* Weighted voting
* Distance-weighted voting
* Feature-weight optimization using gradient descent

---

## Modeling Approach

### Similarity Calculation

For each new patient case, the model compares the patient’s features against historical cases.

For numerical features, similarity is calculated using normalized distance.

For categorical features, similarity is based on exact category matches.

The final similarity score is a weighted average across all features.

```text
Global Similarity = Weighted average of local feature similarities
```

### Prediction

After retrieving the top-k most similar cases, the model predicts the outcome using voting.

Supported voting strategies include:

* Majority voting
* Weighted voting
* Distance-weighted voting

---

## Model Performance

### Binary Classification: CKD Progression

The binary model predicts whether a patient is likely to experience CKD progression.

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |   0.84 |
| ROC-AUC           |  0.865 |
| Weighted F1-score |   0.84 |
| Macro F1-score    |   0.73 |

Classification summary:

| Class          | Precision | Recall | F1-score |
| -------------- | --------: | -----: | -------: |
| No Progression |      0.89 |   0.92 |     0.90 |
| Progression    |      0.61 |   0.53 |     0.57 |

Confusion matrix:

```text
[[215  19]
 [ 27  30]]
```

### Interpretation

The model performs strongly at identifying patients without CKD progression, while the lower recall for progression cases suggests that future improvements should focus on reducing false negatives.

In a healthcare setting, recall for the positive progression class is especially important because missing high-risk patients may be more costly than false alarms.

---

### Multiclass Classification: CKD Stage

The multiclass model predicts CKD stage.

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |   0.65 |
| Weighted F1-score |   0.61 |
| Macro F1-score    |   0.48 |

Classification summary:

| CKD Stage | Precision | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| Stage 2   |      1.00 |   0.11 |     0.20 |
| Stage 3   |      0.66 |   0.88 |     0.75 |
| Stage 4   |      0.62 |   0.64 |     0.63 |
| Stage 5   |      0.58 |   0.21 |     0.31 |

Confusion matrix:

```text
[[  3  23   1   0]
 [  0 114  15   1]
 [  0  32  65   4]
 [  0   3  23   7]]
```

### Interpretation

The multiclass model performs best on Stage 3 and Stage 4 cases but struggles with minority or boundary classes such as Stage 2 and Stage 5. This highlights a common healthcare ML challenge: class imbalance and overlapping clinical profiles between disease stages.

---

## Key Technical Skills Demonstrated

This project demonstrates practical experience in:

* Python programming
* Healthcare data analysis
* Exploratory data analysis
* Data preprocessing
* Case-Based Reasoning
* Similarity-based machine learning
* Binary classification
* Multiclass classification
* Model evaluation
* ROC-AUC analysis
* Confusion matrix interpretation
* Feature weighting
* Interpretable AI for healthcare-style data

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Raphlawren/Chronic-Kindey-Disease-Prediction-with-CASE_BASED-REASONING.git
```

### 2. Navigate into the project folder

```bash
cd Chronic-Kindey-Disease-Prediction-with-CASE_BASED-REASONING
cd "Raphael Lawrence Farodoye COM804 Test"
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy jupyter
```

### 5. Run the notebooks

```bash
jupyter notebook
```

Open:

```text
EDA.ipynb
run_cbr_ckd.ipynb
```

---

## Main Files

### `cbr.py`

Contains the custom Case-Based Reasoning implementation.

Main components include:

* Case-base construction
* Feature range calculation
* Local similarity calculation
* Global similarity ranking
* Top-k similar case retrieval
* Solution reuse through voting
* Probability estimation
* Gradient-descent feature-weight optimization

### `EDA.ipynb`

Contains exploratory data analysis of the CKD dataset.

### `run_cbr_ckd.ipynb`

Runs the CBR workflow for:

* Binary CKD progression prediction
* Multiclass CKD stage classification
* Model evaluation
* Confusion matrix analysis
* ROC-AUC calculation for binary classification

---

## What I Learned

Through this project, I strengthened my understanding of:

* How similarity-based reasoning can be applied to healthcare prediction tasks
* How to separate continuous and categorical features for custom ML logic
* How top-k retrieval can be used for interpretable predictions
* How class imbalance affects healthcare classification problems
* Why model evaluation should go beyond accuracy
* How to interpret recall, precision, F1-score, ROC-AUC, and confusion matrices in a medical-risk context

---

## Limitations

The current implementation has several limitations:

* The model is not validated for clinical use
* The positive CKD progression class has lower recall than desired
* The multiclass model struggles with some CKD stages
* No external validation dataset was used
* No deployment interface is currently included
* Additional baselines such as logistic regression, random forest, XGBoost, or SVM would help compare performance
* Feature importance and clinical interpretation could be improved further

---

## Future Improvements

Planned improvements include:

* Add cross-validation for more robust evaluation
* Compare CBR with traditional ML models such as logistic regression, random forest, XGBoost, and SVM
* Improve minority-class performance using class weighting or resampling
* Add feature selection experiments
* Add model calibration for risk probabilities
* Improve explainability by displaying the most similar retrieved patient cases
* Build a Streamlit or Gradio demo for interactive prediction
* Package the model into a reusable Python module
* Add a `requirements.txt` file
* Add unit tests for the CBR class
* Create a cleaner production-style project structure

---

## Portfolio Value

This project is valuable because it goes beyond applying a standard machine learning algorithm. It demonstrates the ability to:

* Understand a healthcare prediction problem
* Build a custom machine learning model from scratch
* Design a similarity-based reasoning system
* Evaluate both binary and multiclass classification tasks
* Communicate model strengths and limitations clearly
* Think critically about interpretability and risk in healthcare AI

---

## Author

**Raphael Farodoye**
Data Scientist | Machine Learning | Healthcare AI | Python

GitHub: [@Raphlawren](https://github.com/Raphlawren)
# Chronic Kidney Disease Prediction with Case-Based Reasoning (CBR)

The vectorized_CBR is easy to understand and implement: [CODE NOTEBOOK](https://github.com/Raphlawren/Chronic-Kindey-Disease-Prediction-with-CASE_BASED-REASONING/blob/main/Vectorized_CBR.ipynb).

This repository contains my COM804 project on predicting **Chronic Kidney Disease (CKD)** outcomes using a custom **Case-Based Reasoning (CBR)** system in Python. The dataset has **1,138 patient cases** and **23 features**, with two targets:
- **CKD_Progression** (binary: progressing vs not progressing)
- **CKD_Stage** (multiclass: stages 2–5)

---

## What I built

I implemented a CBR pipeline following the CBR cycle:
**Retrieve → Reuse → Revise → Retain**, encapsulated in a `CBR` class that stores past cases, computes similarity, predicts labels, and learns feature weights. 

### Similarity + retrieval (Retrieve)
- **Categorical features:** similarity = 1 if values match, else 0  
- **Continuous features:** similarity is based on normalized distance (Min–Max), converted to a 0–1 similarity score
- A query case is compared against the whole case base and the **top-k most similar** cases are retrieved. 

### Prediction (Reuse)
I implemented multiple voting options:
- Majority voting
- Similarity-weighted voting
- Distance-weighted voting (squared weights)

### Learning feature weights with gradient descent (Revise/Retain)
A key part of this project is automatic feature-weight learning using **gradient descent** on a validation set:
- predict using current weights  
- compute error against an “ideal” similarity (1 for correct-label neighbors, 0 otherwise)  
- take gradients and update weights to improve accuracy 

---

## Data preparation (summary)

### Missing data
Missingness was analyzed with missingness plots, and the pattern suggested **Not Missing at Random (NMAR)** due to incomplete records.

### Handling missing data
- Dropped **88 rows** with missing values in key columns (CKD_Risk, Protein_Creatinine_Ratio, UPCR_Severity), which also reduced missingness in other correlated features. 
- Used **median imputation** for skewed continuous features.
- Used **class-conditional mode imputation** for selected categorical features (mode computed within each CKD stage). 

### Outliers
Creatinine outliers were removed using the **IQR method** to avoid dominating distance calculations in CBR. 

### Train/test split
Data was split into **70% train / 30% test** with **stratified sampling** to preserve class distribution.
---

## Results

### 1) CKD_Progression (Binary)
- Trained for **200 epochs** to learn feature weights. 
- **Test Accuracy:** **86%**  
- **ROC-AUC:** **0.87** 
- Note: the dataset is imbalanced (around **80.3%** “not progressing” vs **19.7%** “progressing”). 

### 2) CKD_Stage (Multiclass)
- **300 epochs**, training accuracy improved from **57% → 70%** (loss decreased).  


---

## How to run (typical workflow)

1. Clone the repo
2. Create a virtual environment and install requirements
3. Run the notebook/script in the project folder and point it to the dataset path used in the code

(Exact commands depend on how you structured the repo files.)

---

## Notes
This was implemented as a “from scratch” CBR system (not using a ready-made CBR library), including similarity design, multiple voting rules, and gradient-descent weight learning.
