


# Titanic Survival Predictor

**A machine learning mini-project to predict survival outcomes for Titanic passengers based on their details.**

## 📌 Overview
This project uses a **Random Forest Classifier** to predict whether a passenger would have survived the Titanic disaster, based on features like:
- Passenger class (Pclass)
- Sex
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Port of embarkation (Embarked)

The project includes:
- A **Jupyter Notebook/Python script** for model training and saving the model/encoders.
- A **Streamlit web app** for interactive predictions.

---

## 📂 Project Structure
```
.
├── train.csv                # Titanic dataset (from Kaggle)
├── titanic_model.pkl        # Trained model (saved)
├── label_encoders.pkl       # Label encoders for categorical features
├── app.py                   # Streamlit web app
├── model.py                 # creating model
└── README.md                # This file
```

---

## 🛠 Setup & Installation

### 1. Download the Dataset
- Download the [Titanic dataset](https://www.kaggle.com/competitions/titanic/data) from Kaggle.
- Place `train.csv` in the project directory.

---

## 🚀 Running the Project

### 1. Train the Model
Run the training script (or notebook) to generate the model and encoders:
```bash
python model.py
```
*(This will create `titanic_model.pkl` and `label_encoders.pkl`.)*

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
- The app will open in your default browser.
- Enter passenger details and click **"Predict Survival Outcome"** to see the result.

---

## 📊 Model Information
- **Algorithm:** Random Forest Classifier
- **Features Used:** Pclass, Sex, Age, SibSp, Parch, Embarked
- **Accuracy:** (You can add your model’s accuracy here after evaluation)

---

## 📝 Notes
- This is an **educational/demonstration** project.
- The model is trained on historical data and may not reflect real-world survival probabilities.
- For best results, ensure all input fields are filled accurately.

---
