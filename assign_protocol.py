# assign_protocol.py

import pandas as pd
import numpy as np

# Configuration
PATIENT_DATA_CSV = '/Users/laurentelkrief/Desktop/Neuromod/Research/TableaudeBord/patient_dashbord/patient_data.csv'  # Remplacez par le chemin réel de votre fichier
OUTPUT_CSV = '/Users/laurentelkrief/Desktop/Neuromod/Research/TableaudeBord/patient_dashbord/patient_data_with_protocol.csv'  # Chemin pour le fichier de sortie

# Options de protocole
protocol_options = ['HF - 10Hz', 'iTBS', 'BR - 18Hz']

# Charger les données patients
df = pd.read_csv(PATIENT_DATA_CSV)

# Vérifier si la colonne 'protocol' existe déjà
if 'protocol' in df.columns:
    print("La colonne 'protocol' existe déjà. Elle sera mise à jour avec de nouvelles assignations.")
else:
    print("Ajout de la colonne 'protocol'.")

# Assigner aléatoirement un protocole à chaque patient
df['protocol'] = np.random.choice(protocol_options, size=len(df))

# Sauvegarder le fichier mis à jour
df.to_csv(OUTPUT_CSV, index=False)
print(f"Les données mises à jour avec 'protocol' ont été sauvegardées dans '{OUTPUT_CSV}'.")
