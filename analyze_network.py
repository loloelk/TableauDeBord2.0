# analyze_network.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données simulées
ema_df = pd.read_csv('simulated_ema_data.csv')

# Définir les symptômes
SYMPTOMS = MADRS_ITEMS + ANXIETY_ITEMS + [SLEEP, ENERGY, STRESS]

# Fonction pour ajuster le modèle et extraire les coefficients
def fit_multilevel_models(df, symptom, predictors):
    # Définir les variables lag
    for predictor in predictors:
        df[f'{predictor}_lag'] = df[predictor].shift(1)
    
    # Supprimer les premières entrées avec NaN
    df_model = df.dropna()
    
    # Définir la formule
    formula = f"{symptom} ~ " + " + ".join([f"{pred}_lag" for pred in predictors])
    
    # Ajuster le modèle à effets mixtes
    model = mixedlm(formula, df_model, groups=df_model['PatientID'])
    result = model.fit()
    return result

# Construire une matrice de coefficients pour un patient spécifique
def build_coef_matrix(patient_id, df, symptoms):
    df_patient = df[df['PatientID'] == patient_id].sort_values('Date')
    coef_matrix = pd.DataFrame(index=symptoms, columns=symptoms, dtype=float)
    
    for symptom in symptoms:
        predictors = symptoms
        try:
            result = fit_multilevel_models(df_patient, symptom, predictors)
            coef = result.params.filter(regex='_lag$').values
            coef_matrix.loc[symptom, symptoms] = coef
        except Exception as e:
            print(f"Erreur pour le symptôme {symptom} du patient {patient_id}: {e}")
            coef_matrix.loc[symptom, symptoms] = 0
    
    return coef_matrix

# Construire et visualiser le réseau pour un patient
def visualize_patient_network(patient_id, df, symptoms, threshold=0.3):
    coef_matrix = build_coef_matrix(patient_id, df, symptoms)
    G = construct_network(coef_matrix, threshold=threshold)
    fig = plot_network(G, title=f"Symptom Network for Patient {patient_id}")
    fig.show()

# Définir les fonctions construct_network et plot_network comme précédemment

# Exemple d'utilisation
patient_id = 'P001'
visualize_patient_network(patient_id, ema_df, SYMPTOMS, threshold=0.3)
