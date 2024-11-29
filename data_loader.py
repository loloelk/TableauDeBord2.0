# data_loader.py

import pandas as pd
import logging
import yaml
import os

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

PATIENT_DATA_CSV = config['paths']['patient_data']
NURSE_INPUTS_CSV = config['paths']['nurse_inputs']
SIMULATED_EMA_CSV = config['paths']['simulated_ema_data']

def load_patient_data(csv_file: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='utf-8')
        logging.debug(f"Patient data loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='latin1')
            logging.debug(f"Patient data loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load patient data from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load patient data from {csv_file}: {e}")
        return pd.DataFrame()

def validate_patient_data(data: pd.DataFrame):
    if 'ID' not in data.columns:
        logging.error("The 'ID' column is missing in the patient data.")
        raise ValueError("La colonne 'ID' est manquante dans le fichier CSV des patients.")

    if data['ID'].isnull().any():
        logging.error("There are empty entries in the 'ID' column.")
        raise ValueError("Certaines entrées de la colonne 'ID' sont vides. Veuillez les remplir.")

    if data['ID'].duplicated().any():
        logging.error("There are duplicate IDs in the 'ID' column.")
        raise ValueError("Il y a des IDs dupliqués dans la colonne 'ID'. Veuillez assurer l'unicité.")

    logging.debug("Patient data validation passed.")

def load_simulated_ema_data(csv_file: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(csv_file, dtype={'PatientID': str}, encoding='utf-8')
        logging.debug(f"Simulated EMA data loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'PatientID': str}, encoding='latin1')
            logging.debug(f"Simulated EMA data loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load simulated EMA data from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load simulated EMA data from {csv_file}: {e}")
        return pd.DataFrame()
