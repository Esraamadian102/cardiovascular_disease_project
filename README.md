# Cardiovascular Disease Prediction Project

## Overview
This project aims to predict cardiovascular disease using machine learning techniques. The model is trained on a dataset of 70,000 patient records with various health metrics and lifestyle factors.

## Dataset
The dataset contains the following features:
- Age (in years)
- Gender (1: Female, 2: Male)
- Height (in cm)
- Weight (in kg)
- Systolic blood pressure (ap_hi)
- Diastolic blood pressure (ap_lo)
- Cholesterol (1: normal, 2: above normal, 3: well above normal)
- Glucose (1: normal, 2: above normal, 3: well above normal)
- Smoking status (0: non-smoker, 1: smoker)
- Alcohol intake (0: no alcohol, 1: alcohol consumption)
- Physical activity (0: not active, 1: active)
- Target variable: Presence of cardiovascular disease (0: no, 1: yes)

## Project Structure
- `cardio_train.csv`: Original dataset
- `processed_data.csv`: Cleaned and preprocessed dataset
- `eda_notebook.py`: Exploratory data analysis script
- `model_building.py`: Model training and evaluation script
- `app.py`: Streamlit web application for prediction
- `plots/`: Directory containing EDA visualizations
- `model_plots/`: Directory containing model performance visualizations
- `scaler.pkl`: Saved StandardScaler for preprocessing new data
- `best_model.pkl`: Saved best-performing model
- `predict_function.pkl`: Saved prediction function for the web app

## Exploratory Data Analysis
The EDA revealed several important insights:
1. The dataset is balanced with approximately 50% of patients having cardiovascular disease
2. Age is strongly correlated with cardiovascular disease risk
3. Blood pressure (both systolic and diastolic) shows strong correlation with cardiovascular disease
4. BMI is higher on average for patients with cardiovascular disease
5. Cholesterol and glucose levels show clear associations with cardiovascular disease
6. Lifestyle factors (smoking, alcohol, physical activity) show some relationship with cardiovascular disease

## Models Evaluated
1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The Gradient Boosting model performed best with approximately 74% accuracy and was further optimized through hyperparameter tuning.

## Web Application
The project includes a Streamlit web application (`app.py`) that allows users to:
1. Input patient information
2. View calculated health metrics (BMI, hypertension status)
3. Get a prediction of cardiovascular disease risk
4. See key risk factors and recommendations

## Running the Application
To run the web application:
```
pip install streamlit
streamlit run app.py
```
## video 
https://www.canva.com/design/DAGpnTrlPnY/8_vr65oGTwq_FMDOAo48tA/edit?utm_content=DAGpnTrlPnY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
## Requirements
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib
- dill

## Future Improvements
1. Collect additional data on lifestyle factors and medical history
2. Experiment with more advanced models like neural networks
3. Develop personalized risk reduction strategies based on identified risk factors
4. Incorporate more detailed medical guidelines for recommendations

## Disclaimer
This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
