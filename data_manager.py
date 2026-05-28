import os
import pandas as pd
import numpy as np

def load_and_clean_data(filepath='Comprehensive_Traffic_Data_2025.csv'):
    """
    Loads the traffic data CSV, handles missing values, standardizes text columns,
    handles age anomalies, and creates the target variable.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file at '{filepath}' was not found. Please check the path.")

    # Load the raw CSV and explicitly copy to prevent SettingWithCopy warnings
    df = pd.read_csv(filepath).copy()
    
    # --- 1. REMOVE MISSING CITIES & CRITICAL VALUES ---
    df = df.dropna(subset=['City'])
    
    # --- 2. BASIC CLEANING ---
    df['Damage_Cost_PHP'] = pd.to_numeric(df['Damage_Cost_PHP'], errors='coerce')
    df['Severity'] = df['Severity'].astype(str).str.extract(r'(\d+)').astype(float)
    df = df.dropna(subset=['Severity', 'Damage_Cost_PHP'])

    # --- 3. STANDARDIZATION ---
    for col in ['City', 'Vehicle_Type']:
        df[col] = df[col].astype(str).str.strip().str.title()
        
    for col in ['Weather_Condition', 'Traffic_Volume']:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    weather_map = {'Rainy': 'Rain', 'Stormy': 'Storm'}
    df['Weather_Condition'] = df['Weather_Condition'].replace(weather_map)

    # --- 4. ANOMALY HANDLING (Driver Age) ---
    df['Driver_Age'] = pd.to_numeric(df['Driver_Age'], errors='coerce')
    df.loc[(df['Driver_Age'] < 0) | (df['Driver_Age'] > 100), 'Driver_Age'] = np.nan
    
    mean_age = df['Driver_Age'].mean()
    df['Driver_Age'] = df['Driver_Age'].fillna(mean_age)
    
    # --- 5. TARGET DEFINITION ---
    df['High_Risk_Target'] = (df['Severity'] > 3).astype(int)
    
    return df.reset_index(drop=True)