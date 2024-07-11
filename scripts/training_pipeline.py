# training.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from preprocessing_pipeline import run_data_pipeline
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_COMPANY = "historical_info_ISA_Interconnection_Electric.csv"
DATA_FOLDER = "data/"
FINAL_FILES_PATH = os.path.join(DATA_FOLDER, DATA_COMPANY)

def load_and_preprocess_data(path_file: str) -> pd.DataFrame:    
    """
    Load and preprocess the dataset.

    Parameters:
    path_file (str): Path to the dataset file.

    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    logging.info(f'Leyendo archivo: {DATA_COMPANY}') 

    data_transformed = run_data_pipeline(path_file)
    return data_transformed

def split_data(dataset: pd.DataFrame, target_column: str):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    dataset (pd.DataFrame): The preprocessed dataset.
    target_column (str): The name of the target column.

    Returns:
    Tuple containing train, validation, and test sets.
    """
    logging.info("Training, testing and validation sets preparing!")    

    X = dataset[['Apertura', 'Máximo', 'Mínimo', 'Vol.', '% var.']]
    y = dataset[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logging.info("Training, testing and validation sets partition has finished!")    

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation labels.

    Returns:
    XGBRegressor: The trained XGBoost model.
    """
    logging.info("Training XGBoost model has started!")    

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.

    Parameters:
    model (XGBRegressor): The trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.

    Returns:
    float: Mean Squared Error of the model on the test set.
    """
    logging.info("Training XGBoost Evaluation model has started!")    

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def main():
    """
    Main function to run the training pipeline.
    """
    dataset = load_and_preprocess_data(FINAL_FILES_PATH)
    target_column = 'Último'
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset, target_column)
    
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error on the test set: {mse}")
    
    # Save the model
    model.save_model('../models/xgboost_model.json')

if __name__ == "__main__":
    main()
