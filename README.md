# Bank Risk Controller - Classification Model

#### Problem Statement:
#### We have been provided with loan application details of a financial institution.  The dataset contains features which describe about a loan applicant’s personal, professional, properties held, salary, credit liability etc.,There is also a “TARGET” feature which provides information about whether the loan was processed. Now, the goal of this project is to analyze all the features, understand their relationship, then train a machine learning model to understand these relationships and predict the outcome (i.e.) process/reject a loan application. Banks and financial institutions can use the model to assess the risk of potential borrowers defaulting on loans, helping in decision-making for loan approvals. 

### Setting up the conda environment 
```conda create -p env python==3.10```

### Activate the conda environment
```conda activate env\```

### Install all the requirements 
```pip install -r requirements.txt```

### Classification Model 
#### Training - Regression 
#### Path : \Bank Risk Controller
```python class_training.py```
#### Model Training would be completed and the following pickle files would be generated 
#### pickle file path : \Bank Risk Controller\pickle_files
#### boxcox_params.pkl, capping_bounds.pkl, label_encoders.pkl, rf_model.pkl, scaler.pkl 


### Model Testing 
### Run the Streamlit app, pass the required inputs and click on Predict
### In order to test the Regression Model click on Regressoon
#### Path : \Singapore Flat Price
```streamlit run app.py```

