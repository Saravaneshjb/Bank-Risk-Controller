import streamlit as st
import pandas as pd
from class_testing import ClassificationTestingPipeline

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Home", "Data", "EDA", "Prediction"])

if page == "Home":
    st.title("Bank Risk Controller")
    st.write("""
        ### Description of the Problem Statement:
        We have been provided with loan application details of a financial institution.  The dataset contains features which describe about a loan applicant’s personal, professional, properties held, salary, credit liability etc.,
        There is also a “TARGET” feature which provides information about whether the loan was processed. 
        Now, the goal of this project is to analyze all the features, understand their relationship, then train a machine learning model to understand these relationships and predict the outcome (i.e.) process/reject a loan application. 
        Banks and financial institutions can use the model to assess the risk of potential borrowers defaulting on loans, helping in decision-making for loan approvals. 
    """)

elif page == "Data":
    st.title("Bank Risk Controller - Sample Data & Model Details")

elif page == "EDA":
    st.title("Bank Risk Controller - Exploratory Data Analysis")
    
elif page == "Prediction":
    st.title("Bank Risk Controller - Model Prediction")

    # Option to choose between uploading an Excel file or manual data entry
    input_option = st.selectbox("Choose input method", ["Upload Excel File", "Manual Entry"])

    if input_option == "Upload Excel File":
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write("Uploaded Data:")
            st.write(df)
            
            # Process each row in the uploaded Excel file
            if st.button("Predict"):
                pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Final_Project\\pickle_files'
                test_pipeline = ClassificationTestingPipeline(pickle_folder_path)
                
                predictions = []
                for _, row in df.iterrows():
                    input_data = row.to_dict()
                    # Process input_data as in manual entry
                    input_data['DAYS_BIRTH'] = int(input_data['DAYS_BIRTH'])
                    input_data['DAYS_ID_PUBLISH'] = int(input_data['DAYS_ID_PUBLISH'])
                    input_data['DAYS_REGISTRATION'] = float(input_data['DAYS_REGISTRATION'])
                    input_data['DAYS_LAST_PHONE_CHANGE'] = float(input_data['DAYS_LAST_PHONE_CHANGE'])
                    input_data['DAYS_EMPLOYED'] = int(input_data['DAYS_EMPLOYED'])
                    input_data['AMT_CREDIT_x'] = float(input_data['AMT_CREDIT_x'])
                    input_data['REGION_POPULATION_RELATIVE'] = float(input_data['REGION_POPULATION_RELATIVE'])
                    input_data['AMT_INCOME_TOTAL'] = float(input_data['AMT_INCOME_TOTAL'])
                    input_data['HOUR_APPR_PROCESS_START_x'] = int(input_data['HOUR_APPR_PROCESS_START_x'])
                    input_data['CNT_FAM_MEMBERS'] = int(input_data['CNT_FAM_MEMBERS'])
                    input_data['CNT_CHILDREN'] = int(input_data['CNT_CHILDREN'])
                    input_data['EXT_SOURCE_2'] = float(input_data['EXT_SOURCE_2'])
                    input_data['AMT_ANNUITY_x'] = float(input_data['AMT_ANNUITY_x'])
                    input_data['AMT_GOODS_PRICE_x'] = float(input_data['AMT_GOODS_PRICE_x'])
                    input_data['EXT_SOURCE_3'] = float(input_data['EXT_SOURCE_3'])
                    input_data['EXT_SOURCE_1'] = float(input_data['EXT_SOURCE_1'])
                    input_data['BASEMENTAREA_AVG'] = float(input_data['BASEMENTAREA_AVG'])
                    input_data['TOTALAREA_MODE'] = float(input_data['TOTALAREA_MODE'])
                    input_data['LANDAREA_AVG'] = float(input_data['LANDAREA_AVG'])
                    input_data['OBS_30_CNT_SOCIAL_CIRCLE'] = float(input_data['OBS_30_CNT_SOCIAL_CIRCLE'])
                    input_data['OBS_60_CNT_SOCIAL_CIRCLE'] = float(input_data['OBS_60_CNT_SOCIAL_CIRCLE'])
                    input_data['AMT_REQ_CREDIT_BUREAU_YEAR'] = float(input_data['AMT_REQ_CREDIT_BUREAU_YEAR'])
                    input_data['ENTRANCES_MODE'] = float(input_data['ENTRANCES_MODE'])
                    input_data['ELEVATORS_MODE'] = float(input_data['ELEVATORS_MODE'])
                    input_data['FLOORSMAX_MODE'] = float(input_data['FLOORSMAX_MODE'])
                    input_data['AMT_REQ_CREDIT_BUREAU_QRT'] = float(input_data['AMT_REQ_CREDIT_BUREAU_QRT'])
                    input_data['NFLAG_INSURED_ON_APPROVAL'] = int(input_data['NFLAG_INSURED_ON_APPROVAL'])
                    # The other string-type inputs remain as is
                    
                    # Proceed with prediction
                    preprocessed_df = test_pipeline.preprocess(input_data)
                    prediction = test_pipeline.predict(preprocessed_df)
                    predictions.append(prediction[0])
                
                # Display predictions
                df['Prediction'] = predictions
                st.write("Predictions:")
                st.write(df)

    elif input_option == "Manual Entry":
        input_data = {
            'DAYS_BIRTH': st.text_input("DAYS_BIRTH (e.g., -9461)"),
            'DAYS_ID_PUBLISH': st.text_input("DAYS_ID_PUBLISH (e.g., -2120)"),
            'DAYS_REGISTRATION': st.text_input("DAYS_REGISTRATION (e.g., -3648)"),
            'DAYS_LAST_PHONE_CHANGE': st.text_input("DAYS_LAST_PHONE_CHANGE (e.g., -1134)"),
            'DAYS_EMPLOYED': st.text_input("DAYS_EMPLOYED (e.g., -637)"),
            'AMT_CREDIT_x': st.text_input("AMT_CREDIT_x (e.g., 406597.5)"),
            'REGION_POPULATION_RELATIVE': st.text_input("REGION_POPULATION_RELATIVE (e.g., 0.018801)"),
            'AMT_INCOME_TOTAL': st.text_input("AMT_INCOME_TOTAL (e.g., 202500)"),
            'ORGANIZATION_TYPE': st.text_input("ORGANIZATION_TYPE (e.g., Business Entity Type 3)"),
            'HOUR_APPR_PROCESS_START_x': st.text_input("HOUR_APPR_PROCESS_START_x (e.g., 10)"),
            'WEEKDAY_APPR_PROCESS_START_x': st.text_input("WEEKDAY_APPR_PROCESS_START_x (e.g., MONDAY)"),
            'NAME_FAMILY_STATUS': st.text_input("NAME_FAMILY_STATUS (e.g., Married)"),
            'HOUR_APPR_PROCESS_START_y': st.text_input("HOUR_APPR_PROCESS_START_y (e.g., 10)"),
            'CNT_FAM_MEMBERS': st.text_input("CNT_FAM_MEMBERS (e.g., 2)"),
            'CNT_CHILDREN': st.text_input("CNT_CHILDREN (e.g., 0)"),
            'EXT_SOURCE_2': st.text_input("EXT_SOURCE_2 (e.g., 0.262949)"),
            'AMT_ANNUITY_x': st.text_input("AMT_ANNUITY_x (e.g., 24700.5)"),
            'AMT_GOODS_PRICE_x': st.text_input("AMT_GOODS_PRICE_x (e.g., 351000)"),
            'EXT_SOURCE_3': st.text_input("EXT_SOURCE_3 (e.g., 0.139376)"),
            'EXT_SOURCE_1': st.text_input("EXT_SOURCE_1 (e.g., 0.083037)"),
            'BASEMENTAREA_AVG': st.text_input("BASEMENTAREA_AVG (e.g., 0.0369)"),
            'TOTALAREA_MODE': st.text_input("TOTALAREA_MODE (e.g., 0.041)"),
            'LANDAREA_AVG': st.text_input("LANDAREA_AVG (e.g., 0.0369)"),
            'OBS_30_CNT_SOCIAL_CIRCLE': st.text_input("OBS_30_CNT_SOCIAL_CIRCLE (e.g., 2)"),
            'OBS_60_CNT_SOCIAL_CIRCLE': st.text_input("OBS_60_CNT_SOCIAL_CIRCLE (e.g., 1)"),
            'AMT_REQ_CREDIT_BUREAU_YEAR': st.text_input("AMT_REQ_CREDIT_BUREAU_YEAR (e.g., 1)"),
            'ENTRANCES_MODE': st.text_input("ENTRANCES_MODE (e.g., 0.5)"),
            'ELEVATORS_MODE': st.text_input("ELEVATORS_MODE (e.g., 0.25)"),
            'FLOORSMAX_MODE': st.text_input("FLOORSMAX_MODE (e.g., 0.3)"),
            'AMT_REQ_CREDIT_BUREAU_QRT': st.text_input("AMT_REQ_CREDIT_BUREAU_QRT (e.g., 1)"),
            'NFLAG_INSURED_ON_APPROVAL': st.text_input("NFLAG_INSURED_ON_APPROVAL (e.g., 0)"),
            'PRODUCT_COMBINATION': st.text_input("PRODUCT_COMBINATION (e.g., Cash loans)"),
            'NAME_TYPE_SUITE_x': st.text_input("NAME_TYPE_SUITE_x (e.g., Unaccompanied)"),
            'OCCUPATION_TYPE': st.text_input("OCCUPATION_TYPE (e.g., Laborers)"),
            'WALLSMATERIAL_MODE': st.text_input("WALLSMATERIAL_MODE (e.g., Panel)")
        }

        if st.button("Predict"):
            # Convert inputs to appropriate types for manual entry
            input_data['DAYS_BIRTH'] = int(input_data['DAYS_BIRTH'])
            input_data['DAYS_ID_PUBLISH'] = int(input_data['DAYS_ID_PUBLISH'])
            input_data['DAYS_REGISTRATION'] = float(input_data['DAYS_REGISTRATION'])
            input_data['DAYS_LAST_PHONE_CHANGE'] = float(input_data['DAYS_LAST_PHONE_CHANGE'])
            input_data['DAYS_EMPLOYED'] = int(input_data['DAYS_EMPLOYED'])
            input_data['AMT_CREDIT_x'] = float(input_data['AMT_CREDIT_x'])
            input_data['REGION_POPULATION_RELATIVE'] = float(input_data['REGION_POPULATION_RELATIVE'])
            input_data['AMT_INCOME_TOTAL'] = float(input_data['AMT_INCOME_TOTAL'])
            input_data['HOUR_APPR_PROCESS_START_x'] = int(input_data['HOUR_APPR_PROCESS_START_x'])
            input_data['CNT_FAM_MEMBERS'] = int(input_data['CNT_FAM_MEMBERS'])
            input_data['CNT_CHILDREN'] = int(input_data['CNT_CHILDREN'])
            input_data['EXT_SOURCE_2'] = float(input_data['EXT_SOURCE_2'])
            input_data['AMT_ANNUITY_x'] = float(input_data['AMT_ANNUITY_x'])
            input_data['AMT_GOODS_PRICE_x'] = float(input_data['AMT_GOODS_PRICE_x'])
            input_data['EXT_SOURCE_3'] = float(input_data['EXT_SOURCE_3'])
            input_data['EXT_SOURCE_1'] = float(input_data['EXT_SOURCE_1'])
            input_data['BASEMENTAREA_AVG'] = float(input_data['BASEMENTAREA_AVG'])
            input_data['TOTALAREA_MODE'] = float(input_data['TOTALAREA_MODE'])
            input_data['LANDAREA_AVG'] = float(input_data['LANDAREA_AVG'])
            input_data['OBS_30_CNT_SOCIAL_CIRCLE'] = float(input_data['OBS_30_CNT_SOCIAL_CIRCLE'])
            input_data['OBS_60_CNT_SOCIAL_CIRCLE'] = float(input_data['OBS_60_CNT_SOCIAL_CIRCLE'])
            input_data['AMT_REQ_CREDIT_BUREAU_YEAR'] = float(input_data['AMT_REQ_CREDIT_BUREAU_YEAR'])
            input_data['ENTRANCES_MODE'] = float(input_data['ENTRANCES_MODE'])
            input_data['ELEVATORS_MODE'] = float(input_data['ELEVATORS_MODE'])
            input_data['FLOORSMAX_MODE'] = float(input_data['FLOORSMAX_MODE'])
            input_data['AMT_REQ_CREDIT_BUREAU_QRT'] = float(input_data['AMT_REQ_CREDIT_BUREAU_QRT'])
            input_data['NFLAG_INSURED_ON_APPROVAL'] = int(input_data['NFLAG_INSURED_ON_APPROVAL'])
            # The other string-type inputs remain as is

            # Proceed with prediction
            pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Final_Project\\pickle_files'
            test_pipeline = ClassificationTestingPipeline(pickle_folder_path)
            preprocessed_df = test_pipeline.preprocess(input_data)
            prediction = test_pipeline.predict(preprocessed_df)
            st.write(f"Prediction: {prediction[0]}")
