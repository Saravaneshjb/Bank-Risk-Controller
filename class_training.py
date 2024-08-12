import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import boxcox
import pickle
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,f1_score,precision_score,recall_score,confusion_matrix

class ClassificationTrainingPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.label_encoders = {}
        self.capping_bounds = {}
        self.boxcox_params = {}
        self.scaler = StandardScaler()
        self.target_mean_encoders = {} 
        self.rf_model = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 10})
    
    def preprocess(self):
        # 1. Read the csv file into a dataframe
        fraud_df = pd.read_csv(self.filepath)

        # 2.Choose the top 20 features
        top_35_features=['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'AMT_ANNUITY_x', 'DAYS_EMPLOYED', 'AMT_CREDIT_x', 'BASEMENTAREA_AVG', 'TOTALAREA_MODE', 'LANDAREA_AVG', 'REGION_POPULATION_RELATIVE', 'AMT_GOODS_PRICE_x', 'AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE', 'HOUR_APPR_PROCESS_START_x', 'OCCUPATION_TYPE', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'ENTRANCES_MODE', 'WEEKDAY_APPR_PROCESS_START_x', 'ELEVATORS_MODE', 'FLOORSMAX_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT', 'NAME_FAMILY_STATUS', 'HOUR_APPR_PROCESS_START_y', 'CNT_FAM_MEMBERS', 'WALLSMATERIAL_MODE', 'PRODUCT_COMBINATION', 'NAME_TYPE_SUITE_x', 'CNT_CHILDREN', 'NFLAG_INSURED_ON_APPROVAL','TARGET'] 
        
        fraud_df=fraud_df[top_35_features]

        # Drop the duplicates
        fraud_df.drop_duplicates(inplace=True)
        
        # 3.Identify quantitative and qualitative features
        Quantitative_features = [col for col in fraud_df.columns if fraud_df[col].dtype in ['int64', 'float64']]
        Qualitative_features = [col for col in fraud_df.columns if fraud_df[col].dtype not in ['int64', 'float64']]

        # 4.Further segregation of quantitative features into continuous and discrete
        Continuous_features = []
        Discrete_features = []

        for col in Quantitative_features:
            unique_values = fraud_df[col].nunique()
            if fraud_df[col].dtype == 'float64':
                if unique_values > 33:  # Threshold for considering discrete vs continuous (you can adjust this)
                    Continuous_features.append(col)
                else:
                    Discrete_features.append(col)
            elif fraud_df[col].dtype == 'int64':
                if unique_values > 33:  # Threshold for considering discrete vs continuous (you can adjust this)
                    Continuous_features.append(col)
                else:
                    Discrete_features.append(col)
        
        # Assuming fraud_df is your DataFrame and Continuous_features is your list of feature names
        fraud_df[Continuous_features] = fraud_df[Continuous_features].astype('float64')

        #5. Identify the outliers to perform Box Cox Transformation
        outliers_dict,no_outliers_dict=self.outlier_detection(Continuous_features,fraud_df)

        #6. Apply Box-Cox transformation
        for feature in outliers_dict.keys():
            # Ensure the feature is positive and exclude NaNs
            valid_data = fraud_df[feature].dropna()

            if (valid_data > 0).all():
                # Apply Box-Cox transformation
                transformed_data, lambda_param = boxcox(valid_data)

                # Fill the transformed data back into the original dataframe
                fraud_df.loc[valid_data.index, feature] = transformed_data

                # Store the lambda parameter instead of the transformed data
                self.boxcox_params[feature] = lambda_param
            else:
                # Shift the data to be positive and exclude NaNs
                min_value = valid_data.min()
                shifted_data = valid_data - min_value + 1
                transformed_data, lambda_param = boxcox(shifted_data)

                # Fill the transformed data back into the original dataframe
                fraud_df.loc[valid_data.index, feature] = transformed_data

                # Store the lambda parameter instead of the transformed data
                self.boxcox_params[feature] = (lambda_param, min_value)
        
        #7. Identify the outliers to apply Winsorization
        outliers_dict,no_outliers_dict=self.outlier_detection(Continuous_features,fraud_df)
        
        #8. Capping or Winsorization 
        for feature in outliers_dict.keys():
            fraud_df[feature],lower_bound,upper_bound = self.iqr_capping(fraud_df[feature])
            self.capping_bounds[feature]=(lower_bound,upper_bound)
        
        #9.Handling NaN values in the dataset
        less_missing_continuous_features = [features for features in Continuous_features if fraud_df[features].isna().sum()<=10000 and fraud_df[features].isna().sum()>0]
        more_missing_continuous_features = [features for features in Continuous_features if fraud_df[features].isna().sum()>10000]

        less_missing_discrete_features = [features for features in Discrete_features if fraud_df[features].isna().sum()<=10000 and fraud_df[features].isna().sum()>0]
        more_missing_discrete_features = [features for features in Discrete_features if fraud_df[features].isna().sum()>10000] 

        less_missing_categorical_features = [features for features in Qualitative_features if fraud_df[features].isna().sum()<=10000 and fraud_df[features].isna().sum()>0]
        more_missing_categorical_features = [features for features in Qualitative_features if fraud_df[features].isna().sum()>10000]
        
        # Custom imputer for categorical features
        def fill_missing_with_category(X):
            return X.fillna('missing')
        
        # Define pipelines for preprocessing
        less_missing_continuous_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))])

        more_missing_continuous_pipeline = Pipeline(steps=[
            ('imputer', IterativeImputer())])
        
        less_missing_discrete_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))])

        more_missing_discrete_pipeline = Pipeline(steps=[
            ('imputer', IterativeImputer())])

        less_missing_categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))])

        more_missing_categorical_pipeline = Pipeline(steps=[
            ('imputer', FunctionTransformer(fill_missing_with_category, validate=False))])
        
        # Combine all pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('less_missing_continuous', less_missing_continuous_pipeline, less_missing_continuous_features),
                ('more_missing_continuous', more_missing_continuous_pipeline, more_missing_continuous_features),
                ('less_missing_discrete', less_missing_discrete_pipeline, less_missing_discrete_features),
                ('more_missing_discrete', more_missing_discrete_pipeline, more_missing_discrete_features),
                ('less_missing_categorical', less_missing_categorical_pipeline, less_missing_categorical_features),
                ('more_missing_categorical', more_missing_categorical_pipeline, more_missing_categorical_features)
            ])

        # Fit and transform the data
        preprocessed_data = preprocessor.fit_transform(fraud_df)

        # Convert to DataFrame for easier viewing
        preprocessed_columns = (
            less_missing_continuous_features +
            more_missing_continuous_features +
            less_missing_discrete_features +
            more_missing_discrete_features +
            less_missing_categorical_features +
            more_missing_categorical_features)

        preprocessed_df = pd.DataFrame(preprocessed_data, columns=preprocessed_columns, index=fraud_df.index)

        # Drop the preprocessed columns from the fraud_df dataframe 
        fraud_df.drop(columns=preprocessed_columns, axis=1, inplace=True)

        # Concatenate the preprocessed DataFrame with the original DataFrame
        fraud_df = pd.concat([fraud_df, preprocessed_df], axis=1)

        # Perform Feature Encoding 
        Qualitative_dict=dict()
        Qualitative_dict['less_cardinal']=list()
        Qualitative_dict['high_cardinal']=list()
        for features in Qualitative_features:
            if len(fraud_df[features].unique()) <= 8:
                Qualitative_dict['less_cardinal'].append(features)
            else:
                Qualitative_dict['high_cardinal'].append(features)
        
        # Label Encoding
        for feature in Qualitative_dict['less_cardinal']:
            le = LabelEncoder()
            fraud_df[feature] = le.fit_transform(fraud_df[feature])
            self.label_encoders[feature]=le
            print("label encoder created for the feature :", feature)
        
        # Target Guided Mean Encoding
        for feature in Qualitative_dict['high_cardinal']:
            # Calculate the mean of the target for each category
            mean_encoding = fraud_df.groupby(feature)['TARGET'].mean()

            # Map the mean values to the original feature
            fraud_df[feature] = fraud_df[feature].map(mean_encoding)
            self.target_mean_encoders[feature] = mean_encoding
            print("Target guided mean encoding created for :", feature)
        
        ## Converting all the columns to numeric dtype
        fraud_df=fraud_df.apply(pd.to_numeric, errors='coerce')

        #Apply Standard Scalar 
        print("Order of Continuous Features before scaling :",Continuous_features)
        fraud_df.loc[:, Continuous_features] = self.scaler.fit_transform(fraud_df[Continuous_features])


        self.preprocessed_df = fraud_df

        print("The shape of preprocessed dataframe :",fraud_df.shape)
        print("The columns in preprocessed dataframe :",fraud_df.columns)
        print(fraud_df.head(1))
    
    
    # function to perform the IQR capping 
    def iqr_capping(self,series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(series, lower_bound, upper_bound),lower_bound,upper_bound
    
    # function to detect the outlier 
    def outlier_detection(self,Continuous_features,fraud_df1):
        outliers_dict={}
        no_outliers_dict={}
        for features in Continuous_features:
            Q1=fraud_df1[features].quantile(0.25)
            Q3=fraud_df1[features].quantile(0.75)
            IQR=Q3-Q1
            lower_bound=Q1-(1.5*IQR)
            upper_bound=Q3+(1.5*IQR)

            outliers_count=len(fraud_df1[(fraud_df1[features]<lower_bound) | (fraud_df1[features]>upper_bound)])

            if outliers_count>0:
                outliers_dict[features]=outliers_count
            else:
                no_outliers_dict[features]=outliers_count

        return outliers_dict, no_outliers_dict
        
    def build_model(self):
        # 2. Split the dataframe into train and test
        X = self.preprocessed_df.drop('TARGET', axis=1)
        y = self.preprocessed_df['TARGET']
        print("Columns order in X to be used in testing : ",X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
        
        # Initialize the RandomUnderSampler
        undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)

        # Fit and apply the undersampler to the training data 
        X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
        
        # 3. Build the Random Forest model
        self.rf_model.fit(X_train_res, y_train_res)

        # 4. Checking the model performance
        y_test_pred=self.rf_model.predict(X_test)
        print("accuracy score :",accuracy_score(y_test,y_test_pred))
        print("precision score :",precision_score(y_test,y_test_pred))
        print("recall score :",recall_score(y_test,y_test_pred))
        print("f1-score score :",f1_score(y_test,y_test_pred))

    
    def pickle_dump(self, obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    
    def save_objects(self,folder_path):
        # Ensure the folder exists, if not, create it
        os.makedirs(folder_path, exist_ok=True)

        self.pickle_dump(self.capping_bounds, os.path.join(folder_path,'capping_bounds.pkl'))
        self.pickle_dump(self.label_encoders, os.path.join(folder_path,'label_encoders.pkl'))
        self.pickle_dump(self.scaler, os.path.join(folder_path,'scaler.pkl'))
        self.pickle_dump(self.boxcox_params, os.path.join(folder_path,'boxcox_params.pkl'))
        self.pickle_dump(self.target_mean_encoders, os.path.join(folder_path,'target_mean_encoders.pkl'))
        self.pickle_dump(self.rf_model, os.path.join(folder_path,'rf_model.pkl'))



if __name__ == "__main__":
    train_pipeline = ClassificationTrainingPipeline('D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Final_Project\\Data\\loan_data.csv')
    print("Starting the Preprocess workflow")
    train_pipeline.preprocess()
    print("Preprocess workflow - completed")
    print("Starting the model building workflow")
    train_pipeline.build_model()
    print("model building workflow-completed")

    # Specify the folder path where you want to save the pickle files
    pickle_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Final_Project\\pickle_files'
    print("Starting Pickling workflow")
    train_pipeline.save_objects(pickle_path)
    print("Pickling workflow completed")