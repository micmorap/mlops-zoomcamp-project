#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from ydata_profiling import ProfileReport
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_COMPANY = "historical_info_ISA_Interconnection_Electric.csv"
DATA_FOLDER = "../data/"
FINAL_FILES_PATH = os.path.join(DATA_FOLDER, DATA_COMPANY)
PROFILING_REPORTS_PATH = "../profiling_reports/"


def read_data_stock(name_file: str) -> pd.DataFrame:
    """
    Reads a CSV file containing stock data into a Pandas DataFrame.

    Parameters:
    name_file (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The data from the CSV file.
    """
    logging.info('Iniciando el proceso de lectura de datos...')
    primary_dataset = pd.read_csv(name_file)
    logging.info('Proceso de lectura de datos completado.')
    return primary_dataset


def profiling_eda_report(dataset: pd.DataFrame) -> ProfileReport:
    """
    Generates an exploratory data analysis (EDA) report for the given dataset using ydata_profiling.

    Parameters:
    dataset (pd.DataFrame): The dataset to profile.

    Returns:
    ProfileReport: The generated EDA report.
    """
    logging.info('Generando reporte de EDA...')
    report = ProfileReport(dataset, tsmode=True, sortby="Fecha", title="Profiling Report Stock prices")
    report.to_file(f"{PROFILING_REPORTS_PATH}report_stock_prices.html")
    logging.info('Reporte de EDA generado.')
    return report


def transform_type_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the data types of specific columns in the dataset.

    - Converts date columns to a consistent format.
    - Converts price columns to floats.
    - Converts volume column to floats and handles 'K' and 'M' suffixes.
    - Converts percentage variation column to floats.

    Parameters:
    dataset (pd.DataFrame): The dataset to transform.

    Returns:
    pd.DataFrame: The transformed dataset.
    """
    date_columns = 'Fecha'
    price_columns = ['Último', 'Apertura', 'Máximo', 'Mínimo']
    volume_column = 'Vol.'
    variation_column = '% var.'

    logging.info('Transformando tipos de columnas...')
    dataset = dataset.drop(columns=[date_columns])
    # dataset[date_columns] = dataset[date_columns].str.replace(".", "-", regex=False)

    for col in price_columns:
        dataset[col] = dataset[col].str.replace(".", "").str.replace(",", ".").astype(float)

    dataset[volume_column] = dataset[volume_column].str.replace(",", ".")
    dataset[volume_column] = dataset[volume_column].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x
                                                          else float(x.replace('M', '')) * 1000000 if 'M' in x
                                                          else float(x))

    dataset[variation_column] = dataset[variation_column].astype(str).str.replace('%', '').str.replace(',', '.').astype(float) / 100
    logging.info('Transformación de tipos de columnas completada.')
    return dataset


def run_data_pipeline(path_file: str) -> pd.DataFrame:
    """
    Executes the entire data pipeline: reading the data, generating a profiling report,
    and transforming the data.

    Parameters:
    path_file (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The transformed dataset.
    """
    initial_dataset_csv = read_data_stock(path_file)
    profiling_report = profiling_eda_report(initial_dataset_csv)
    dataset_transformed = transform_type_columns(initial_dataset_csv)
    return dataset_transformed
