import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.special import boxcox1p
import pickle


class ClassificationTestingPipeline:
    def __init__(self, pickle_folder_path):
        self.pickle_folder_path=pickle_folder_path
        self.feature_order = ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
       'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED', 'AMT_CREDIT_x',
       'REGION_POPULATION_RELATIVE', 'AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE',
       'HOUR_APPR_PROCESS_START_x', 'WEEKDAY_APPR_PROCESS_START_x',
       'NAME_FAMILY_STATUS', 'HOUR_APPR_PROCESS_START_y', 'CNT_FAM_MEMBERS',
       'CNT_CHILDREN', 'EXT_SOURCE_2', 'AMT_ANNUITY_x', 'AMT_GOODS_PRICE_x',
       'EXT_SOURCE_3', 'EXT_SOURCE_1', 'BASEMENTAREA_AVG', 'TOTALAREA_MODE',
       'LANDAREA_AVG', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'AMT_REQ_CREDIT_BUREAU_YEAR', 'ENTRANCES_MODE', 'ELEVATORS_MODE',
       'FLOORSMAX_MODE', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'NFLAG_INSURED_ON_APPROVAL', 'PRODUCT_COMBINATION', 'NAME_TYPE_SUITE_x',
       'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE'] 
        self.capping_bounds = self.pickle_load(os.path.join(self.pickle_folder_path, 'capping_bounds.pkl'))
        self.label_encoders = self.pickle_load(os.path.join(self.pickle_folder_path, 'label_encoders.pkl'))
        self.scaler = self.pickle_load(os.path.join(self.pickle_folder_path, 'scaler.pkl'))
        self.boxcox_params = self.pickle_load(os.path.join(self.pickle_folder_path, 'boxcox_params.pkl'))
        self.target_mean_encoders = self.pickle_load(os.path.join(self.pickle_folder_path, 'target_mean_encoders.pkl'))
        # self.rf_model = self.pickle_load(os.path.join(self.pickle_folder_path, 'rf_model2_undersampling.pkl'))
        self.rf_model = self.pickle_load(os.path.join(self.pickle_folder_path, 'rf_model.pkl'))
    
    def preprocess(self, input_data):
       
        # 1. Read the input data 
        df = pd.DataFrame([input_data])

        Continuous_features=['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'AMT_ANNUITY_x', 'DAYS_EMPLOYED', 'AMT_CREDIT_x', 'BASEMENTAREA_AVG', 'TOTALAREA_MODE', 'LANDAREA_AVG', 'REGION_POPULATION_RELATIVE', 'AMT_GOODS_PRICE_x', 'AMT_INCOME_TOTAL']

        print("The box cox params : " ,self.boxcox_params.keys())
        print("The capping params : " ,self.capping_bounds.keys())

        # 2. Apply Box Cox Transformation
        for col in Continuous_features:
            if col in self.boxcox_params.keys():
                params = self.boxcox_params[col]
                if isinstance(params, tuple):  # Data was shifted during training
                    lambda_param, min_value = params
                    df[col] = df[col] - min_value + 1  # Apply the same shift
                else:
                    lambda_param = params
                df[col] = boxcox1p(df[col], lambda_param)

        # 3. Outlier Capping
        for col in Continuous_features:
            if col in self.capping_bounds.keys():
                lower_bound, upper_bound = self.capping_bounds[col]
                # df[col] = df[col].apply(lambda x: np.clip(x, lower_bound, upper_bound) if x < lower_bound or x > upper_bound else x)
                # df[col] = np.where(df[col] < lower_bound, lower_bound, np.where(df[col] > upper_bound, upper_bound, df[col]))
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # Debugging: Print the data before scaling
        print("Data before scaling:\n", df.head())
        print("Shape of data before scaling:", df[['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1','DAYS_BIRTH','DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION',
        'AMT_ANNUITY_x', 'DAYS_EMPLOYED', 'AMT_CREDIT_x','REGION_POPULATION_RELATIVE', 'LANDAREA_AVG', 'AMT_INCOME_TOTAL', 'TOTALAREA_MODE', 'BASEMENTAREA_AVG', 'AMT_GOODS_PRICE_x']].shape)
        
        # 4. Apply Standard Scaler
        try:
            transformed_data = self.scaler.transform(df[Continuous_features])
            df[Continuous_features] = transformed_data
        except ValueError as e:
            print("Error during scaling:", str(e))
            # return None  # Handle the error as needed

        # 5. Label Encoding- This is not needed as none of the features fall under the less cardinal category
        for col in ['WEEKDAY_APPR_PROCESS_START_x','NAME_FAMILY_STATUS','WALLSMATERIAL_MODE','NAME_TYPE_SUITE_x']:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])
        
        # 6. Target guided Mean Encoding
        for col in ['ORGANIZATION_TYPE','OCCUPATION_TYPE','PRODUCT_COMBINATION']:
            df[col]=df[col].map(self.target_mean_encoders[col])
        
        # 7. Ensure the features are in the same order as training
        df=df[self.feature_order]
        
        return df
    
    def predict(self, preprocessed_df):
        return self.rf_model.predict(preprocessed_df)
    
    def pickle_load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
