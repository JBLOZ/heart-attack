import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# 1. Cargar datos
def load_data(path):
    df = pd.read_csv(path)
    return df

# 2. Generar variables derivadas si no están presentes
def derive_lifestyle_variables(df):
    # Índice de Masa Corporal (IMC) si existe peso y altura
    if 'Peso_kg' in df.columns and 'Altura_m' in df.columns:
        df['IMC'] = df['Peso_kg'] / (df['Altura_m'] ** 2)
    # Actividad física: ejemplo de MET-minutos semanales
    if 'Minutos_actividad_semanal' in df.columns:
        df['Actividad_Fisica'] = df['Minutos_actividad_semanal'] / 150  # Normalizar vs recomendación WHO
    # Calidad de la dieta: suponga un puntaje 0-100
    if 'Puntaje_dieta' in df.columns:
        df['Calidad_Dieta'] = df['Puntaje_dieta'] / 100
    # Tabaquismo: índice de paquetes-año
    if 'Cigarrillos_por_dia' in df.columns and 'Anios_fumando' in df.columns:
        df['Tabaquismo'] = (df['Cigarrillos_por_dia'] / 20) * df['Anios_fumando']
    # Consumo de alcohol: unidades estándar por semana
    if 'Copas_por_semana' in df.columns:
        df['Consumo_Alcohol'] = df['Copas_por_semana'] / 14  # Definición de consumo moderado
    # Nivel de estrés: escala 1-10
    if 'Escala_estres' in df.columns:
        df['Estres'] = df['Escala_estres'] / 10
    # Ansiedad y Depresión: asuma escala 0-21 (GAD-7, PHQ-9)
    for col, max_score in [('Ansiedad', 21), ('Depresion', 27)]:
        if col in df.columns:
            df[col] = df[col] / max_score
    return df

# 3. Seleccionar variables relevantes
def select_variables(df):
    cols = [
        'Presion_Sistolica', 'Presion_Diastolica', 'Colesterol_LDL', 'Colesterol_HDL',
        'Glucemia', 'IMC', 'FC_reposo', 'Trigliceridos', 'Colesterol_Total',
        'hs_CRP', 'Lipoproteina_a', 'Homocisteina', 'Actividad_Fisica', 'Calidad_Dieta',
        'Tabaquismo', 'Consumo_Alcohol', 'Historia_Familiar', 'Score_Poligenico',
        'Estres', 'Ansiedad', 'Depresion', 'Edad', 'Sexo'
    ]
    present = [c for c in cols if c in df.columns]
    return df[present]

# 4. Normalizar datos

def normalize_data(df, method='minmax'):
    scaler = MinMaxScaler() if method=='minmax' else StandardScaler()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = scaler.fit_transform(df[numeric])
    # Dejar variables categóricas sin cambios (Sexo, etc.)
    return df, scaler

# 5. Graficar distribuciones

def plot_distributions(df):
    numeric = df.select_dtypes(include=[np.number]).columns
    n = len(numeric)
    cols = 3
    nrows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(5*cols,4*nrows))
    axes = axes.flatten()
    for i, col in enumerate(numeric):
        axes[i].hist(df[col].dropna(), bins=30)
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()

# 6. Correlation heatmap

def plot_correlation(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10,8))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Mapa de calor de correlaciones')
    plt.tight_layout()
    plt.show()

# Función principal
def preprocess(path):
    df = load_data(path)
    df = derive_lifestyle_variables(df)
    df_sel = select_variables(df)
    df_norm, scaler = normalize_data(df_sel)
    plot_distributions(df_norm)
    plot_correlation(df_norm)
    return df_norm, scaler

if __name__ == '__main__':
    ruta = 'datos_cardio.csv'
    df_normalizado, scaler_usado = preprocess(ruta)
    df_normalizado.to_csv('datos_normalizados.csv', index=False)
    print("Preprocesamiento y normalización completados.")
