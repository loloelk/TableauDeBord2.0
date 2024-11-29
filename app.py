# app.py

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import logging
import yaml
import networkx as nx  # Assurez-vous d'avoir NetworkX installé

from data_loader import load_patient_data, validate_patient_data, load_simulated_ema_data
from nurse_inputs import load_nurse_data, get_nurse_inputs, save_nurse_inputs
from network_visualization import generate_person_specific_network  # Importé mis à jour

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

PATIENT_DATA_CSV = config['paths']['patient_data']
NURSE_INPUTS_CSV = config['paths']['nurse_inputs']
SIMULATED_EMA_CSV = config['paths']['simulated_ema_data']
MADRS_ITEMS_MAPPING = config['mappings']['madrs_items']
PID5_DIMENSIONS_MAPPING = config['mappings']['pid5_dimensions']

# Define a pastel color palette for plots
PASTEL_COLORS = px.colors.qualitative.Pastel

# Page configuration for better aesthetics
st.set_page_config(
    page_title="Tableau de Bord des Patients",
    page_icon=":hospital:",
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded",
)

# Apply custom CSS for styling (compatible with light and dark modes)
st.markdown("""
    <style>
        /* Light Mode Styles */
        @media (prefers-color-scheme: light) {
            /* Typography */
            body {
                font-family: 'Roboto', sans-serif;
            }
            h1, h2, h3, h4 {
                color: #2c3e50;
                font-weight: 600;
            }
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #ecf0f1;
                padding: 2rem;
            }
            /* Main Content */
            .reportview-container .main .block-container {
                padding-top: 3rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            /* DataFrame Tables */
            .dataframe th, .dataframe td {
                padding: 0.5rem;
                text-align: left;
                background-color: #ffffff;
                color: #000000;
            }
            .dataframe thead th {
                background-color: #34495e !important;
                color: #ffffff !important;
            }
            /* Form Elements */
            textarea, input, select {
                background-color: #ffffff;
                color: #000000;
            }
            /* Buttons */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        }

        /* Dark Mode Styles */
        @media (prefers-color-scheme: dark) {
            /* Typography */
            body {
                font-family: 'Roboto', sans-serif;
            }
            h1, h2, h3, h4 {
                color: #ffffff;
                font-weight: 600;
            }
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #1e1e1e;
                padding: 2rem;
            }
            /* Main Content */
            .reportview-container .main .block-container {
                padding-top: 3rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            /* DataFrame Tables */
            .dataframe th, .dataframe td {
                padding: 0.5rem;
                text-align: left;
                color: #ffffff;
                background-color: #2c2c2c;
            }
            .dataframe thead th {
                background-color: #34495e !important;
                color: #ffffff !important;
            }
            /* Form Elements */
            textarea, input, select {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            /* Buttons */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        }

        /* Universal Styles */
        /* Ensure tables are responsive */
        .dataframe {
            width: 100%;
            overflow-x: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Load and validate patient data
final_data = load_patient_data(PATIENT_DATA_CSV)
if final_data.empty:
    st.error("Aucune donnée patient chargée. Veuillez vérifier le fichier CSV.")
    st.stop()

try:
    validate_patient_data(final_data)
except ValueError as ve:
    st.error(str(ve))
    st.stop()

logging.debug("Colonnes des données finales : %s", final_data.columns.tolist())
logging.debug("Échantillon des données finales :\n%s", final_data.head())

# Initialize nurse_data in st.session_state to manage it across the app
if 'nurse_data' not in st.session_state:
    st.session_state.nurse_data = load_nurse_data(NURSE_INPUTS_CSV)

# Load simulated EMA data
simulated_ema_data = load_simulated_ema_data(SIMULATED_EMA_CSV)

# Définir les variables de Symptômes
MADRS_ITEMS = [f'madrs_{i}' for i in range(1, 11)]      # madrs_1 à madrs_10
ANXIETY_ITEMS = [f'anxiety_{i}' for i in range(1, 6)]  # anxiety_1 à anxiety_5
SLEEP = 'sleep'
ENERGY = 'energy'
STRESS = 'stress'

SYMPTOMS = MADRS_ITEMS + ANXIETY_ITEMS + [SLEEP, ENERGY, STRESS]

# Function to merge simulated EMA data with final_data if needed
def merge_simulated_data(final_df: pd.DataFrame, ema_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge simulated EMA data with the final patient data.
    """
    if ema_df.empty:
        logging.warning("Simulated EMA data is empty. Skipping merge.")
        return final_df
    # Assuming 'ID' in final_data corresponds to 'PatientID' in simulated_ema_data
    merged_df = final_df.merge(ema_df, how='left', left_on='ID', right_on='PatientID')
    return merged_df

final_data = merge_simulated_data(final_data, simulated_ema_data)

# Function to extract numeric part of patient ID for sorting
def extract_number(id_str):
    match = re.search(r'\d+', id_str)
    return int(match.group()) if match else float('inf')

# Sidebar Configuration
with st.sidebar:
    st.title("Tableau de Bord des Patients")
    st.markdown("---")
    st.header("Navigation")
    page = st.radio("Aller à", ["Tableau de Bord du Patient", "Entrées Infirmières", "Détails PID-5"])

    st.markdown("---")
    st.header("Sélectionner un Patient")
    # Combine existing and simulated patient IDs
    existing_patient_ids = final_data['ID'].unique().tolist()
    simulated_patient_ids = simulated_ema_data['PatientID'].unique().tolist() if not simulated_ema_data.empty else []
    all_patient_ids = sorted(list(set(existing_patient_ids + simulated_patient_ids)), key=extract_number)
    selected_patient_id = st.selectbox("Sélectionner l'ID du Patient", all_patient_ids) if all_patient_ids else None

    # Display the number of loaded patients for debugging
    st.write(f"Nombre de patients chargés : {len(all_patient_ids)}")

# Function to get EMA data for a patient
def get_patient_ema_data(patient_id: str, ema_data: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve EMA data for a specific patient.
    """
    patient_ema = ema_data[ema_data['PatientID'] == patient_id].sort_values(by='Timestamp')
    return patient_ema

# Page "Tableau de Bord du Patient"
def patient_dashboard():
    if not selected_patient_id:
        st.warning("Aucun patient sélectionné.")
        return

    # Retrieve patient data
    patient_row = final_data[final_data["ID"] == selected_patient_id]
    if patient_row.empty:
        st.error("Données du patient non trouvées.")
        return
    patient_data = patient_row.iloc[0]

    # Retrieve EMA data if available
    patient_ema = get_patient_ema_data(selected_patient_id, simulated_ema_data)

    # Aperçu du Patient et Objectifs SMART
    st.header("Aperçu du Patient")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Informations du Patient")
            st.write(f"**Âge :** {patient_data.get('age', 'N/A')}")
            sex_numeric = patient_data.get('sexe', 'N/A')
            # Correction de l'affichage de "Sexe"
            sex = "Homme" if sex_numeric == '1' else "Femme" if sex_numeric == '2' else "Autre" if sex_numeric else "N/A"
            st.write(f"**Sexe :** {sex}")
            education_years = patient_data.get('annees_education_bl', 'N/A')
            st.write(f"**Années d'éducation (Baseline) :** {education_years}")
            revenu_bl = patient_data.get('revenu_bl', 'N/A')
            # Ajout du symbole '$' devant le revenu et formatage avec des virgules
            try:
                revenu_int = int(revenu_bl)
                revenu_formate = f"${revenu_int:,}"
            except:
                revenu_formate = f"${revenu_bl}" if revenu_bl != 'N/A' else 'N/A'
            st.write(f"**Revenu (Baseline) :** {revenu_formate}")
        with col2:
            st.subheader("Objectifs SMART")
            # Charger les entrées infirmières pour ce patient depuis st.session_state
            nurse_inputs = get_nurse_inputs(selected_patient_id, st.session_state.nurse_data)
            objectives = nurse_inputs.get("objectives", "N/A")
            tasks = nurse_inputs.get("tasks", "N/A")
            comments = nurse_inputs.get("comments", "N/A")
            st.write(f"**Objectifs :** {objectives}")
            st.write(f"**Tâches :** {tasks}")
            st.write(f"**Commentaires :** {comments}")

    st.markdown("---")

    # Données Démographiques et Cliniques
    st.header("Données Démographiques et Cliniques")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Données Démographiques")
            demog_labels = ["Sexe", "Âge", "Années d'éducation (Baseline)", "Revenu (Baseline)"]
            demog_values = [
                sex,
                patient_data.get('age', 'N/A'),
                education_years,
                revenu_formate
            ]
            demog_df = pd.DataFrame({
                "Paramètre": demog_labels,
                "Valeur": demog_values
            })
            st.table(demog_df)
        with col2:
            st.subheader("Données Cliniques")
            clin_labels = ["Comorbidités", "Enceinte", "Cigarettes (Baseline)", "Alcool (Baseline)", "Cocaïne (Baseline)"]
            # Correction de l'affichage de "Enceinte"
            pregnant_val = patient_data.get('pregnant', 'N/A')
            if pregnant_val == '1':
                pregnant_display = "Oui"
            elif pregnant_val == '0':
                pregnant_display = "Non"
            else:
                pregnant_display = "N/A"
            clin_values = [
                patient_data.get('comorbidities', 'N/A'),
                pregnant_display,
                patient_data.get('cigarette_bl', 'N/A'),
                patient_data.get('alcool_bl', 'N/A'),
                patient_data.get('cocaine_bl', 'N/A')
            ]
            clin_df = pd.DataFrame({
                "Paramètre": clin_labels,
                "Valeur": clin_values
            })
            st.table(clin_df)

    st.markdown("---")

    # Scores MADRS
    st.header("Scores MADRS")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Score Total MADRS")
            madrs_total = {
                "Baseline": patient_data.get("madrs_score_bl", 0),
                "Jour 30": patient_data.get("madrs_score_fu", 0)
            }
            fig_madrs = px.bar(
                x=list(madrs_total.keys()),
                y=list(madrs_total.values()),
                labels={"x": "Temps", "y": "Score MADRS"},
                color=list(madrs_total.keys()),
                color_discrete_sequence=PASTEL_COLORS,
                title="Score Total MADRS"
            )
            st.plotly_chart(fig_madrs, use_container_width=True)
        with col2:
            st.subheader("Scores par Item MADRS")
            # Assuming MADRS item columns are named like 'madrs_1_bl', 'madrs_1_fu', etc.
            madrs_items = patient_data.filter(regex=r"^madrs[_.]\d+[_.](bl|fu)$")
            if madrs_items.empty:
                st.warning("Aucun score par item MADRS trouvé pour ce patient.")
            else:
                madrs_items_df = madrs_items.to_frame().T
                madrs_long = madrs_items_df.melt(var_name="Item", value_name="Score").dropna()
                madrs_long["Temps"] = madrs_long["Item"].str.extract("_(bl|fu)$")[0]
                madrs_long["Temps"] = madrs_long["Temps"].map({"bl": "Baseline", "fu": "Jour 30"})
                madrs_long["Item_Number"] = madrs_long["Item"].str.extract(r"madrs[_.](\d+)_")[0].astype(int)
                madrs_long["Item"] = madrs_long["Item_Number"].map(MADRS_ITEMS_MAPPING)
                madrs_long.dropna(subset=["Item"], inplace=True)

                if madrs_long.empty:
                    st.warning("Tous les scores par item MADRS sont NaN.")
                else:
                    fig = px.bar(
                        madrs_long,
                        x="Item",
                        y="Score",
                        color="Temps",
                        barmode="group",
                        title="Scores par Item MADRS",
                        template="plotly_white",
                        color_discrete_sequence=PASTEL_COLORS
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scores d'Évaluation (PID-5 et PHQ-9)
    st.header("Scores d'Évaluation")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            # PID-5 Handling (Assuming columns like 'pid5_1_bl', 'pid5_1_fu', etc.)
            has_pid5 = any(col.startswith('pid5_') for col in final_data.columns)
            if has_pid5:
                st.subheader("Scores PID-5")
                pid5_columns_bl = []
                pid5_columns_fu = []
                for dimension, items in PID5_DIMENSIONS_MAPPING.items():
                    pid5_columns_bl += [f'pid5_{item}_bl' for item in items]
                    pid5_columns_fu += [f'pid5_{item}_fu' for item in items]

                if not set(pid5_columns_bl + pid5_columns_fu).issubset(final_data.columns):
                    st.warning("Données PID-5 incomplètes pour ce patient.")
                else:
                    dimension_scores_bl = {}
                    dimension_scores_fu = {}
                    for dimension, items in PID5_DIMENSIONS_MAPPING.items():
                        baseline_score = patient_data[[f'pid5_{item}_bl' for item in items]].sum()
                        followup_score = patient_data[[f'pid5_{item}_fu' for item in items]].sum()
                        dimension_scores_bl[dimension] = baseline_score.sum()
                        dimension_scores_fu[dimension] = followup_score.sum()

                    categories = list(PID5_DIMENSIONS_MAPPING.keys())
                    values_bl = list(dimension_scores_bl.values())
                    values_fu = list(dimension_scores_fu.values())

                    # Close the radar chart
                    categories += [categories[0]]
                    values_bl += [values_bl[0]]
                    values_fu += [values_fu[0]]

                    fig_spider = go.Figure()
                    fig_spider.add_trace(go.Scatterpolar(
                        r=values_bl,
                        theta=categories,
                        fill='toself',
                        name='Baseline',
                        line_color=PASTEL_COLORS[0]
                    ))
                    fig_spider.add_trace(go.Scatterpolar(
                        r=values_fu,
                        theta=categories,
                        fill='toself',
                        name='Jour 30',
                        line_color=PASTEL_COLORS[1]
                    ))
                    fig_spider.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=False,  # Hide radial axis labels and ticks
                                range=[0, 15]   # Limit the radial axis to 15
                            )
                        ),
                        showlegend=True,
                        title="Scores par Dimension PID-5",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_spider, use_container_width=True)
            else:
                st.info("Les données PID-5 ne sont pas disponibles.")
        with col2:
            # PHQ-9 Handling (Assuming columns like 'phq9_day1_item1', etc.)
            has_phq9 = any(col.startswith('phq9_') for col in final_data.columns)
            if has_phq9:
                st.subheader("Progression PHQ-9")
                phq9_days = [5, 10, 15, 20, 25, 30]
                phq9_scores = {}
                missing_phq9 = False
                for day in phq9_days:
                    item_columns = [f'phq9_day{day}_item{item}' for item in range(1, 10)]
                    if not set(item_columns).issubset(final_data.columns):
                        missing_phq9 = True
                        break
                    phq9_score = patient_data[item_columns].sum()
                    phq9_scores[f'Jour {day}'] = phq9_score

                if missing_phq9:
                    st.warning("Données PHQ-9 incomplètes pour ce patient.")
                else:
                    phq9_df = pd.DataFrame(list(phq9_scores.items()), columns=["Jour", "Score"])
                    fig_phq9 = px.line(
                        phq9_df,
                        x="Jour",
                        y="Score",
                        markers=True,
                        title="Progression PHQ-9",
                        template="plotly_white",
                        color_discrete_sequence=[PASTEL_COLORS[0]]  # Use a different pastel color
                    )
                    fig_phq9.update_layout(xaxis_title="Jour", yaxis_title="Score PHQ-9")
                    st.plotly_chart(fig_phq9, use_container_width=True)
            else:
                st.info("Les données PHQ-9 ne sont pas disponibles.")

    st.markdown("---")

    # Symptom Network Visualization
    st.header("Réseau de Symptômes")
    with st.container():
        if patient_ema.empty:
            st.warning("Aucune donnée EMA disponible pour ce patient. Le réseau ne peut pas être généré.")
        else:
            # Générer le réseau de symptômes idiographique avec un seuil abaissé à 0.3
            try:
                fig_network = generate_person_specific_network(patient_ema, selected_patient_id, SYMPTOMS, threshold=0.3)
                st.plotly_chart(fig_network, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors de la génération du réseau de symptômes: {e}")

    st.markdown("---")

    # Entrées Infirmières (Dashboard Section for Nurses to Input Objectives)
    st.header("Entrées Infirmières - Dashboard")
    with st.form(key='nursing_inputs_dashboard_form'):
        objectives_input = st.text_area("Objectifs SMART", height=100, value=nurse_inputs.get("objectives", ""))
        tasks_input = st.text_area("Tâches d'Activation Comportementale", height=100, value=nurse_inputs.get("tasks", ""))
        comments_input = st.text_area("Commentaires", height=100, value=nurse_inputs.get("comments", ""))
        submit_button = st.form_submit_button(label='Sauvegarder')
        
        if submit_button:
            try:
                save_nurse_inputs(selected_patient_id, objectives_input, tasks_input, comments_input, st.session_state.nurse_data, NURSE_INPUTS_CSV)
                # Reload nurse data after saving
                st.session_state.nurse_data = load_nurse_data(NURSE_INPUTS_CSV)
                st.success("Entrées infirmières sauvegardées avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde des entrées infirmières: {e}")

    st.markdown("---")

    # Display Saved Nurse Inputs on Dashboard
    st.subheader("Entrées Infirmières Sauvegardées - Dashboard")
    if objectives_input or tasks_input or comments_input:
        st.write(f"**Objectifs :** {objectives_input if objectives_input else 'N/A'}")
        st.write(f"**Tâches :** {tasks_input if tasks_input else 'N/A'}")
        st.write(f"**Commentaires :** {comments_input if comments_input else 'N/A'}")
    else:
        st.write("Aucune entrée sauvegardée.")

# Page "Entrées Infirmières"
def nurse_inputs_page():
    if not selected_patient_id:
        st.warning("Aucun patient sélectionné.")
        return

    st.header("Entrées Infirmières")
    nurse_inputs = get_nurse_inputs(selected_patient_id, st.session_state.nurse_data)
    with st.form(key='nursing_inputs_form_page'):
        objectives_input = st.text_area("Objectifs SMART", height=100, value=nurse_inputs.get("objectives", ""))
        tasks_input = st.text_area("Tâches d'Activation Comportementale", height=100, value=nurse_inputs.get("tasks", ""))
        comments_input = st.text_area("Commentaires", height=100, value=nurse_inputs.get("comments", ""))
        submit_button = st.form_submit_button(label='Sauvegarder')

        if submit_button:
            try:
                save_nurse_inputs(selected_patient_id, objectives_input, tasks_input, comments_input, st.session_state.nurse_data, NURSE_INPUTS_CSV)
                # Reload nurse data after saving
                st.session_state.nurse_data = load_nurse_data(NURSE_INPUTS_CSV)
                st.success("Entrées infirmières sauvegardées avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde des entrées infirmières: {e}")

    st.markdown("---")

    # Display Saved Nurse Inputs on Page
    st.subheader("Entrées Infirmières Sauvegardées")
    if nurse_inputs:
        st.write(f"**Objectifs :** {nurse_inputs.get('objectives', 'N/A')}")
        st.write(f"**Tâches :** {nurse_inputs.get('tasks', 'N/A')}")
        st.write(f"**Commentaires :** {nurse_inputs.get('comments', 'N/A')}")
    else:
        st.write("Aucune entrée sauvegardée.")

# Page "Détails PID-5"
def details_pid5_page():
    if not selected_patient_id:
        st.warning("Aucun patient sélectionné.")
        return

    if not any(col.startswith('pid5_') for col in final_data.columns):
        st.info("Les données PID-5 ne sont pas disponibles.")
        return

    patient_row = final_data[final_data["ID"] == selected_patient_id]
    if patient_row.empty:
        st.error("Données du patient non trouvées.")
        return
    patient_data = patient_row.iloc[0]

    pid5_columns = []
    for dimension, items in PID5_DIMENSIONS_MAPPING.items():
        pid5_columns += [f'pid5_{item}_bl' for item in items] + [f'pid5_{item}_fu' for item in items]

    if not set(pid5_columns).issubset(final_data.columns):
        st.warning("Données PID-5 incomplètes pour ce patient.")
        return

    dimension_scores_bl = {}
    dimension_scores_fu = {}
    for dimension, items in PID5_DIMENSIONS_MAPPING.items():
        baseline_score = patient_data[[f'pid5_{item}_bl' for item in items]].sum()
        followup_score = patient_data[[f'pid5_{item}_fu' for item in items]].sum()
        dimension_scores_bl[dimension] = baseline_score.sum()
        dimension_scores_fu[dimension] = followup_score.sum()

    # Prepare data for the table
    table_data = []
    for dimension in PID5_DIMENSIONS_MAPPING.keys():
        table_data.append({
            "Domaine": dimension,
            "Total Baseline": f"{dimension_scores_bl[dimension]:,}",
            "Total Jour 30": f"{dimension_scores_fu[dimension]:,}"
        })

    pid5_df = pd.DataFrame(table_data)

    st.subheader("Scores PID-5 par Domaine")
    st.table(pid5_df)

    # Create the spider (radar) chart without reference lines
    categories = list(PID5_DIMENSIONS_MAPPING.keys())
    values_bl = list(dimension_scores_bl.values())
    values_fu = list(dimension_scores_fu.values())

    # Close the radar chart
    categories += [categories[0]]
    values_bl += [values_bl[0]]
    values_fu += [values_fu[0]]

    fig_spider = go.Figure()
    fig_spider.add_trace(go.Scatterpolar(
        r=values_bl,
        theta=categories,
        fill='toself',
        name='Baseline',
        line_color=PASTEL_COLORS[0]
    ))
    fig_spider.add_trace(go.Scatterpolar(
        r=values_fu,
        theta=categories,
        fill='toself',
        name='Jour 30',
        line_color=PASTEL_COLORS[1]
    ))
    fig_spider.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,  # Hide radial axis labels and ticks
                range=[0, 15]   # Limit the radial axis to 15
            )
        ),
        showlegend=True,
        title="Scores par Dimension PID-5",
        template="plotly_white"
    )
    st.plotly_chart(fig_spider, use_container_width=True)

# Main Application Logic
if page == "Tableau de Bord du Patient":
    patient_dashboard()
elif page == "Entrées Infirmières":
    nurse_inputs_page()
elif page == "Détails PID-5":
    details_pid5_page()
else:
    st.error("Page non reconnue.")
