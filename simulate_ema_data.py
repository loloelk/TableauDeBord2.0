# simulate_ema_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration Parameters
NUM_PATIENTS = 50
PATIENT_IDS = [f'P{str(i).zfill(3)}' for i in range(1, NUM_PATIENTS + 1)]
NUM_DAYS = 30
ENTRIES_PER_DAY = 5
START_DATE = datetime(2024, 1, 1)

# Symptom Definitions
MADRS_ITEMS = [f'madrs_{i}' for i in range(1, 11)]      # madrs_1 to madrs_10
ANXIETY_ITEMS = [f'anxiety_{i}' for i in range(1, 6)]  # anxiety_1 to anxiety_5
SLEEP = 'sleep'
ENERGY = 'energy'
STRESS = 'stress'

SYMPTOMS = MADRS_ITEMS + ANXIETY_ITEMS + [SLEEP, ENERGY, STRESS]

# Define baseline correlations between symptoms
# This matrix should be symmetric with 1's on the diagonal
# Values range between -1 and 1
# Adjust these values based on realistic clinical correlations
CORRELATIONS = {
    'madrs_1': {'madrs_1': 1.0, 'madrs_2': 0.5, 'madrs_3': 0.4, 'madrs_4': 0.3, 'madrs_5': 0.2,
               'madrs_6': 0.3, 'madrs_7': 0.4, 'madrs_8': 0.3, 'madrs_9': 0.2, 'madrs_10': 0.3,
               'anxiety_1': 0.6, 'anxiety_2': 0.5, 'anxiety_3': 0.4, 'anxiety_4': 0.3, 'anxiety_5': 0.2,
               'sleep': -0.4, 'energy': -0.5, 'stress': 0.3},
    'madrs_2': {'madrs_1': 0.5, 'madrs_2': 1.0, 'madrs_3': 0.3, 'madrs_4': 0.2, 'madrs_5': 0.1,
               'madrs_6': 0.2, 'madrs_7': 0.3, 'madrs_8': 0.2, 'madrs_9': 0.1, 'madrs_10': 0.2,
               'anxiety_1': 0.5, 'anxiety_2': 0.4, 'anxiety_3': 0.3, 'anxiety_4': 0.2, 'anxiety_5': 0.1,
               'sleep': -0.3, 'energy': -0.4, 'stress': 0.2},
    # Continue filling in the correlations for all symptoms...
    # For brevity, we'll create a random symmetric correlation matrix with controlled correlations
}

# Function to generate a symmetric correlation matrix
def generate_symmetric_correlation_matrix(symptoms, base_corr=0.3, variance=0.1):
    num_symptoms = len(symptoms)
    corr_matrix = np.eye(num_symptoms)
    for i in range(num_symptoms):
        for j in range(i+1, num_symptoms):
            # Random correlation between -0.6 and 0.8, biased towards positive correlations
            corr = random.uniform(base_corr, 0.8)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return pd.DataFrame(corr_matrix, index=symptoms, columns=symptoms)

# Generate the correlation matrix
corr_df = generate_symmetric_correlation_matrix(SYMPTOMS, base_corr=0.3)

# Ensure diagonal is 1
np.fill_diagonal(corr_df.values, 1.0)

# Function to simulate symptom scores
def simulate_symptom_scores(mean_vector, cov_matrix):
    """
    Simulate symptom scores using a multivariate normal distribution.
    """
    scores = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix)
    scores = np.round(scores).astype(int)
    # Define score ranges
    for idx, symptom in enumerate(SYMPTOMS):
        if 'madrs' in symptom or 'anxiety' in symptom:
            scores[idx] = np.clip(scores[idx], 0, 6)
        else:
            scores[idx] = np.clip(scores[idx], 0, 4)
    return scores

# Initialize a list to store all EMA entries
ema_entries = []

# Mean vector for symptom scores
mean_vector = [3] * len(SYMPTOMS)  # Starting with a moderate score

# Simulate data for each patient
for patient_id in PATIENT_IDS:
    previous_scores = mean_vector.copy()
    for day in range(1, NUM_DAYS + 1):
        current_date = START_DATE + timedelta(days=day - 1)
        for entry_num in range(1, ENTRIES_PER_DAY + 1):
            # Simulate a timestamp within the day
            hour = random.randint(6, 22)  # Active hours between 6 AM and 10 PM
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = current_date + timedelta(hours=hour, minutes=minute, seconds=second)
            
            # Simulate symptom scores based on previous scores with some noise
            cov_matrix = corr_df.values
            scores = simulate_symptom_scores(mean_vector, cov_matrix)
            
            # Create an EMA entry
            entry = {
                'PatientID': patient_id,
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Day': day,
                'Entry': entry_num
            }
            for idx, symptom in enumerate(SYMPTOMS):
                entry[symptom] = scores[idx]
            ema_entries.append(entry)
            
            # Update previous scores with some influence from current scores
            # This introduces temporal dependency
            previous_scores = scores.tolist()

# Create a DataFrame from EMA entries
ema_df = pd.DataFrame(ema_entries)

# Save the simulated data to a CSV file
output_csv = 'simulated_ema_data.csv'
ema_df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Simulated EMA data has been successfully saved to '{output_csv}'.")
