import prediction
import streamlit as st

# Add side bar
page = st.sidebar.selectbox('Page', ['Home', 'Predict Dementia Status'])

st.sidebar.markdown('# About')

# Introduction
st.sidebar.write('''Dementia Prediction System is an interactive tool designed to assist in the early detection and prediction of dementia progression. By utilizing clinical data, including MRI scan details and dementia ratings, this system applies machine learning models to analyze patient information and predict cognitive decline. It aims to support healthcare professionals in making informed decisions about dementia management, offering real-time insights into patient conditions. The system features a user-friendly interface where clinical data can be inputted, and predictions are generated with ease.''')

# Features
st.sidebar.write('''### Key Features:
- **Early Detection of Dementia**: The system uses machine learning models to predict the likelihood of a patient being diagnosed as demented or non-demented based on clinical data.
- **Real-Time Predictions**: After inputting patient data, users receive immediate predictions, which can be used to monitor patient progression or for early intervention.
- **Easy-to-Use Interface**: The system provides a straightforward, interactive interface, making it accessible to healthcare professionals with varying levels of technical expertise.
- **Support for Clinical Decision-Making**: The system helps healthcare providers to make more informed decisions by analyzing patient information and providing insights on possible cognitive decline.''')

# Benefit
st.sidebar.write('''### Who can benefit?
- **Healthcare Professionals**: Doctors, clinicians, and healthcare providers who are involved in diagnosing and managing dementia can use the tool to support their clinical decisions.
- **Medical Institutions**: Hospitals, clinics, and research centers can incorporate this system into their diagnostic workflow to aid in dementia-related diagnoses and patient monitoring
- **Caregivers and Families**: While primarily intended for professionals, the system can also provide families and caregivers with insights into a patient’s condition, helping them understand and manage dementia progression.''')

def home():
  # Create title and introduction
  st.title('Welcome to Dementia Prediction System')
  st.write('''This page provides essential information about the key components used during the development of the model. To use the system, please select the **Dementia Prediction System** page from the left pane.''')

  # Add dataset info
  st.write('#### **Dataset**')
  st.markdown('''The dataset used for this project was obtained from Kaggle, providing comprehensive clinical and MRI data relevant to dementia classification. For more details, please visit [MRI and Alzheimer's Kaggle Dataset](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers?select=oasis_longitudinal.csv)''')

  # Add objectives
  st.write('### **Objectives**')
  st.markdown('''
  The primary objective of this project is to develop a reliable Machine Learning model for predicting dementia status based on clinical and MRI data. The system aims to support early detection and assist healthcare professionals in making informed decisions about dementia management.
  Specific goals include:
  1. **Data Understanding & Preprocessing**
  - Collect and organize relevant data.
  - Clean and preprocess data for analysis.
  2. **Exploratory Data Analysis (EDA)**
  - Analyze the dataset through visualizations.
  - Identify patterns and correlations among variables.
  3. **Feature Engineering**
  - Select important features to enhance model accuracy.
  4. **Model Development**
  - Test various Machine Learning models
  - Select the best-performing model based on evaluation metrics
  5. **Model Evaluation**
  - Assess model performance using metrics like accuracy, precision, recall, and F1 score
  6. **Model Deployment**
  - Develop an interactive system for real-time predictions
  - Ensure the system is user-friendly and accessible''')

  # Add machine learning part
  st.write('### **Machine Learning Models Used')
  st.markdown('''
  During the development of the Dementia Prediction System, various machine learning models were tested and evaluated to select the most appropriate one for accurate dementia classification. The models used include:
  1. **Support Vector Classifier (SVC)**: A powerful classification model that works well for high-dimensional datasets like clinical and MRI data. **SVC was chosen as the final model** due to its high performance and accuracy in classifying dementia status.
  2. **Random Forest**: An ensemble learning method that combines multiple decision trees to improve classification accuracy and handle overfitting.
  3. **Decision Tree**: A simple yet effective model that splits the data based on feature values, making decisions based on the most relevant information.
  4. **XGBoost**: An advanced gradient boosting algorithm that has been highly effective in classification tasks, especially when the data is imbalanced.
  5. **Multilayer Perceptron (MLP)**: A deep learning model that applies multiple layers of neurons to capture complex patterns in the data.
  After testing and evaluating the models, **SVC** demonstrated the highest accuracy and reliability, leading to its selection as the final model for deployment.''')

  # Tools utilized
  st.write('### **Libraries Used**')
  st.markdown('''
  Several key libraries were utilized in the development of the **Dementia Prediction System** to handle data processing, model building, and deployment:
  - **Pandas**: For data manipulation and cleaning, particularly in handling structured datasets.
  - **NumPy**: For numerical operations and handling arrays and matrices.
  - **Scikit-Learn**: For implementing machine learning algorithms, model training, and evaluation.
  - **XGBoost**: For implementing the XGBoost model.
  - **Matplotlib & Seaborn**: For data visualization and exploring correlations in the dataset.
  - **Streamlit**: For developing the interactive web application to deploy the prediction tool.
  - **Pickle**: For saving and loading trained models, ensuring that the trained models are persistable and reusable.''')


# Select page
if page == 'Home':
  home()
elif page == 'Predict Dementia Status':
  prediction.run()
