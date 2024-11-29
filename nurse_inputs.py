# nurse_inputs.py

import pandas as pd
import logging
import yaml
import os
from typing import Dict

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

NURSE_INPUTS_CSV = config['paths']['nurse_inputs']

def load_nurse_data(csv_file: str) -> pd.DataFrame:
    """
    Load nurse inputs from a CSV file with appropriate encoding.
    If the file does not exist, initialize it with necessary columns.
    """
    if not os.path.exists(csv_file):
        # Initialize the CSV with necessary columns
        df = pd.DataFrame(columns=["ID", "objectives", "tasks", "comments"])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logging.debug(f"Nurse inputs CSV initialized at {csv_file}.")
        return df
    try:
        data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='utf-8')
        logging.debug(f"Nurse inputs loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='latin1')
            logging.debug(f"Nurse inputs loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load nurse inputs from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load nurse inputs from {csv_file}: {e}")
        return pd.DataFrame()

def get_nurse_inputs(patient_id: str, nurse_data: pd.DataFrame) -> Dict[str, str]:
    """
    Retrieve nurse inputs for a specific patient.
    """
    row = nurse_data[nurse_data["ID"] == patient_id]
    if not row.empty:
        return row.iloc[0][["objectives", "tasks", "comments"]].fillna("").to_dict()
    else:
        return {"objectives": "", "tasks": "", "comments": ""}

def save_nurse_inputs(patient_id: str, objectives: str, tasks: str, comments: str, nurse_data: pd.DataFrame, csv_file: str):
    """
    Save nurse inputs for a specific patient.
    """
    try:
        if patient_id in nurse_data["ID"].values:
            nurse_data.loc[nurse_data["ID"] == patient_id, ["objectives", "tasks", "comments"]] = [objectives, tasks, comments]
        else:
            new_entry = pd.DataFrame([{"ID": patient_id, "objectives": objectives, "tasks": tasks, "comments": comments}])
            nurse_data = pd.concat([nurse_data, new_entry], ignore_index=True)
        nurse_data.to_csv(csv_file, index=False, encoding='utf-8')
        logging.debug(f"Nurse inputs saved successfully for Patient ID {patient_id}.")
    except Exception as e:
        logging.error(f"Failed to save nurse inputs for Patient ID {patient_id}: {e}")
        raise e

# Additional nurse input management functions can be added here as needed.
