# create_symptom_networks.py

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import re
import os

# Import the function to generate symptom networks
# Assuming network_visualization.py is in the same directory
from network_visualization import generate_person_specific_network

# Path to the simulated EMA data
EMA_DATA_CSV = 'simulated_ema_data.csv'

# Load the simulated EMA data
ema_df = pd.read_csv(EMA_DATA_CSV)

# Define the symptom variables
MADRS_ITEMS = [f'madrs_{i}' for i in range(1, 11)]      # madrs_1 to madrs_10
ANXIETY_ITEMS = [f'anxiety_{i}' for i in range(1, 6)]  # anxiety_1 to anxiety_5
SLEEP = 'sleep'
ENERGY = 'energy'
STRESS = 'stress'

SYMPTOMS = MADRS_ITEMS + ANXIETY_ITEMS + [SLEEP, ENERGY, STRESS]

# Function to extract patient IDs
def get_patient_ids(df):
    return df['PatientID'].unique()

# Function to generate and save network plots for each patient
def generate_and_save_networks(df, output_dir='patient_networks', threshold=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    patient_ids = get_patient_ids(df)
    for patient_id in patient_ids:
        patient_df = df[df['PatientID'] == patient_id].sort_values('Timestamp')
        if patient_df.empty:
            print(f"No data for patient {patient_id}. Skipping.")
            continue
        
        # Generate the network figure
        fig = generate_person_specific_network(patient_df, patient_id, SYMPTOMS, threshold=threshold)
        
        # Save the figure as an HTML file
        output_path = os.path.join(output_dir, f'{patient_id}_symptom_network.html')
        fig.write_html(output_path)
        print(f"Symptom network for {patient_id} saved to {output_path}.")

# Run the network generation
generate_and_save_networks(ema_df, output_dir='patient_networks', threshold=0.3)
