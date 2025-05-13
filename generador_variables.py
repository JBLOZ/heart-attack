import pandas as pd
import numpy as np
import re

# ——————————————————————
# Mapping functions para categorías de la encuesta
# ——————————————————————
def map_weekly_freq(cat: str) -> float:
    if pd.isna(cat): return 0.0
    cat = cat.lower()
    if 'nunca' in cat: return 0.0
    if 'casi nunca' in cat: return 0.1
    if 'todos los días' in cat or 'a diario' in cat: return 7.0
    if '1-2' in cat: return 1.5
    if '3-4' in cat: return 3.5
    if '5 o más' in cat or '5 o mas' in cat: return 6.0
    # fallback: extraer primer número
    nums = re.findall(r"(\d+)", cat)
    if nums:
        return float(nums[0])
    return 0.0

def map_tobacco_consumption(cat: str) -> float:
    """
    Mapea categorías de consumo de tabaco a un valor numérico de 0 a 1
    0 = No consume
    0.1 = Casi nunca
    0.5 = Consumo ocasional (1-2 veces por semana)
    0.75 = Consumo moderado (3-4 veces por semana o diario pero menos de 5)
    1.0 = Consumo intenso (diario 5+ cigarrillos)
    """
    if pd.isna(cat): return 0.0
    cat = cat.lower()
    if 'nunca' in cat: return 0.0
    if 'casi nunca' in cat: return 0.1
    if 'cada día' in cat and 'menos de 5' in cat: return 0.75
    if 'cada día' in cat and '5 o más' in cat: return 1.0
    if '1-2' in cat: return 0.5
    if '3-4' in cat: return 0.75
    # Para cualquier otro caso que indique consumo frecuente
    if 'diario' in cat or 'a diario' in cat: return 0.9
    return 0.0

def parse_heart_rate(cat: str) -> float:
    if pd.isna(cat): return np.nan
    cat = cat.lower()
    if 'menor a 60' in cat: return 55.0
    if 'entre 60' in cat and '70' in cat: return 65.0
    if 'entre 70' in cat and '80' in cat: return 75.0
    if 'entre 80' in cat and '90' in cat: return 85.0
    nums = re.findall(r"(\d+)", cat)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    return np.nan

def map_diet_quality(cat: str) -> float:
    if pd.isna(cat): return 0.5
    cat = cat.lower()
    if 'saludable' in cat: return 1.0
    if 'moderada' in cat: return 0.5
    if 'pobre' in cat: return 0.0
    return 0.5

def map_yes_no_maybe(cat: str) -> float:
    if pd.isna(cat): return 0.0
    cat = cat.lower()
    if 'sí' in cat: return 1.0
    if 'tal vez' in cat or 'quizá' in cat: return 0.5
    return 0.0

# ——————————————————————
# Cálculos biomédicos
# ——————————————————————
def compute_bmi(weight: float, height_cm: float) -> float:
    h = height_cm / 100.0
    return weight / (h * h) if h > 0 else np.nan

def normalize_bmi_series(bmi_series: pd.Series) -> pd.Series:
    mi, ma = bmi_series.min(), bmi_series.max()
    return (bmi_series - mi) / (ma - mi)

def compute_systolic_bp(age: float, bmi_norm: float, salt_wk: float,
                        diet_q: float, activity_min: float) -> float:
    base = 110.0
    val = base + age * 0.5 + bmi_norm * 15 - activity_min / 200 + salt_wk * 2 + (1 - diet_q) * 5
    return val + np.random.normal(0, 5)

def compute_ldl(bmi_norm: float, satfat_wk: float, diet_q: float) -> float:
    base = 100.0
    val = base + bmi_norm * 20 + satfat_wk * 5 + (1 - diet_q) * 10
    return val + np.random.normal(0, 10)

def compute_hdl(bmi_norm: float, activity_min: float, diet_q: float) -> float:
    base = 50.0
    val = base - bmi_norm * 10 + activity_min / 200 + diet_q * 5
    return val + np.random.normal(0, 5)

def compute_glucose(bmi_norm: float, sugary_wk: float, activity_min: float) -> float:
    base = 90.0
    val = base + bmi_norm * 20 + sugary_wk * 2 - activity_min / 120
    return val + np.random.normal(0, 5)

def compute_resting_hr(parsed_hr: float, stress_pct: float,
                       anxiety_pct: float, activity_min: float) -> float:
    val = parsed_hr + stress_pct * 0.1 + anxiety_pct * 0.05 - activity_min / 100
    return val + np.random.normal(0, 3)

# ——————————————————————
# Cálculo combinado de calidad de dieta
# ——————————————————————
def compute_diet_quality(row: pd.Series) -> float:
    # 1) Auto-evaluación subjetiva
    self_q = map_diet_quality(
        row['Calidad de dieta: ¿Cómo calificaría la calidad de su alimentación?']
    )
    # 2) Frecuencias semanales de consumos "no saludables"
    salt    = map_weekly_freq(
        row['Uso de sal en la dieta: ¿Con qué frecuencia usa sal de adición para las comidas o ingiere snacks salados?']
    )
    sugary  = map_weekly_freq(
        row['Bebidas azucaradas: ¿Cuántas veces por semana consume refrescos, jugos envasados o bebidas energéticas azucaradas?']
    )
    satfat  = map_weekly_freq(
        row['Consumo de grasas saturadas: ¿Con qué frecuencia consume alimentos ricos en grasas saturadas (hamburguesas, patatas fritas, carnes rojas con grasa, quesos curados, etc.)?']
    )
    alcohol = map_weekly_freq(
        row['Consumo de alcohol: ¿Con qué frecuencia consume bebidas alcohólicas (cerveza, vino, licores)?']
    )
    # 3) Normalizar a [0,1] dividiendo por 7
    max_freq = 7.0
    salt_n    = np.clip(salt    / max_freq, 0, 1)
    sugary_n  = np.clip(sugary  / max_freq, 0, 1)
    satfat_n  = np.clip(satfat  / max_freq, 0, 1)
    alcohol_n = np.clip(alcohol / max_freq, 0, 1)
    # 4) Índice hábitos no saludables
    unhealthy = np.mean([salt_n, sugary_n, satfat_n, alcohol_n])
    # 5) Combinar subjetivo y objetivo
    diet_q = 0.3 * self_q + 0.7 * (1 - unhealthy)
    return float(np.clip(diet_q, 0, 1))

# ——————————————————————
# Cálculo combinado de hábitos nocivos
# ——————————————————————
def compute_harmful_habits_index(row: pd.Series) -> float:
    """
    Calcula un índice de hábitos nocivos basado en:
    1. Consumo de tabaco
    2. Uso de cigarrillos electrónicos/cachimba
    3. Consumo de estupefacientes
    
    Retorna un valor entre 0 (sin hábitos nocivos) y 1 (máximo de hábitos nocivos)
    """
    # 1) Consumo de tabaco (mayor peso: 50%)
    tobacco = map_tobacco_consumption(
        row['Frecuencia de consumo de tabaco: ¿Con qué frecuencia consume tabaco (cigarros, puros, etc.)?']
    )
    
    # 2) Consumo de cigarrillos electrónicos/vapeo (peso: 30%)
    vaping = map_tobacco_consumption(
        row['Frecuencia de consumo de cigarrillos electrónicos o cachimba : ¿Con qué frecuencia utiliza dispositivos como cigarrillos electrónicos o cachimba?']
    )
    
    # 3) Consumo de estupefacientes (peso: 20%)
    drugs = map_tobacco_consumption(
        row['Frecuencia de consumo de estupefacientes: ¿Con qué frecuencia consume sustancias psicoactivas?']
    )
    
    # 4) Combinación ponderada
    harmful_index = 0.5 * tobacco + 0.2 * vaping + 0.3 * drugs
    
    return float(np.clip(harmful_index, 0, 1))


if __name__ == '__main__':
    # 1) Carga y limpieza básica
    df = pd.read_csv('encuesta.csv', sep=';', encoding='utf-8-sig')
    df['Peso'] = (
        df['Rango de peso: ¿Cuál es su peso actual en kg? (solo el numero, sin decimales)']
        .astype(str).str.replace(',', '.')
        .replace({'-': np.nan, '': np.nan})
        .astype(float)
    )
    df['Estatura'] = (
        df['Rango de estatura: ¿Cuál es su estatura actual en cm? (solo el numero, sin comas son cm)']
        .astype(str).str.replace(',', '.')
        .replace({'-': np.nan, '': np.nan})
        .astype(float)
    )
    df['Edad'] = df['Edad:']
    df['Sexo_num'] = df['Sexo:'].map({'Masculino': 1, 'Femenino': 0}).fillna(0)
    df['Diabetes'] = df['Diabetes: ¿Padece de diabetes?'].map(map_yes_no_maybe)
    df['Hipertension'] = df[
        'Hipertensión arterial: ¿Padece o ha padecido hipertensión arterial (tensión alta)?'
    ].map(map_yes_no_maybe)

    # 2) Lifestyle
    df['Activity_days'] = df[
        'Actividad física semanal: ¿Cuántos días a la semana realiza, al menos, 30 minutos de actividad física (moderada o intensa: caminar, nadar, etc.)?'
    ].map(map_weekly_freq)
    df['Activity_min_wk'] = df['Activity_days'] * 30.0

    # Consumo de alcohol (columna corregida)
    df['Alcohol_wk'] = df[
        'Consumo de alcohol: ¿Con qué frecuencia consume bebidas alcohólicas (cerveza, vino, licores)?'
    ].map(map_weekly_freq)

    # Dieta y otros consumos
    df['Diet_q']    = df.apply(compute_diet_quality, axis=1)
    df['SatFat_wk'] = df[
        'Consumo de grasas saturadas: ¿Con qué frecuencia consume alimentos ricos en grasas saturadas (hamburguesas, patatas fritas, carnes rojas con grasa, quesos curados, etc.)?'
    ].map(map_weekly_freq)
    df['Sugary_wk'] = df[
        'Bebidas azucaradas: ¿Cuántas veces por semana consume refrescos, jugos envasados o bebidas energéticas azucaradas?'
    ].map(map_weekly_freq)
    df['Salt_wk']   = df[
        'Uso de sal en la dieta: ¿Con qué frecuencia usa sal de adición para las comidas o ingiere snacks salados?'
    ].map(map_weekly_freq)
    
    # Hábitos nocivos (tabaco, vapeo, estupefacientes)
    df['Harmful_habits'] = df.apply(compute_harmful_habits_index, axis=1)
    
    # Crear columnas individuales para cada componente
    df['Tobacco'] = df[
        'Frecuencia de consumo de tabaco: ¿Con qué frecuencia consume tabaco (cigarros, puros, etc.)?'
    ].map(map_tobacco_consumption)
    
    df['Vaping'] = df[
        'Frecuencia de consumo de cigarrillos electrónicos o cachimba : ¿Con qué frecuencia utiliza dispositivos como cigarrillos electrónicos o cachimba?'
    ].map(map_tobacco_consumption)
    
    df['Drugs'] = df[
        'Frecuencia de consumo de estupefacientes: ¿Con qué frecuencia consume sustancias psicoactivas?'
    ].map(map_tobacco_consumption)

    # 3) Antecedentes familiares y estrés/ansiedad
    df['FamHyper']   = df[
        'Historial familiar: ¿Tiene antecedentes familiares de hipertensión arterial (tensión alta)?'
    ].map(map_yes_no_maybe)
    df['FamInfarct'] = df[
        'Historial familiar: ¿Tiene antecedentes familiares de infarto de miocardio?'
    ].map(map_yes_no_maybe)
    df['Anxiety_pct']= df[
        'Ansiedad: En una escala de 0 a 10, ¿padece o ha padecido ansiedad en el último año?'
    ].astype(float) * 10.0
    df['Stress_pct'] = df[
        'Estrés: En una escala de 0 a 10, ¿padece o ha padecido estrés en el último año?'
    ].astype(float) * 10.0

    # 4) Variables derivadas de salud
    df['BMI']       = df.apply(lambda r: compute_bmi(r['Peso'], r['Estatura']), axis=1)
    df['BMI_norm']  = normalize_bmi_series(df['BMI'])
    df['BP_systolic']= df.apply(
        lambda r: compute_systolic_bp(r['Edad'], r['BMI_norm'], r['Salt_wk'], r['Diet_q'], r['Activity_min_wk']),
        axis=1
    )
    df['LDL']       = df.apply(lambda r: compute_ldl(r['BMI_norm'], r['SatFat_wk'], r['Diet_q']), axis=1)
    df['HDL']       = df.apply(lambda r: compute_hdl(r['BMI_norm'], r['Activity_min_wk'], r['Diet_q']), axis=1)
    df['Glucemia']  = df.apply(lambda r: compute_glucose(r['BMI_norm'], r['Sugary_wk'], r['Activity_min_wk']), axis=1)
    df['HR_base']   = df[
        'Frecuencia cardiaca: Si conoce el valor de sus pulsaciones en reposo, marque la opción más adecuada'
    ].map(parse_heart_rate)
    df['HR_rest']   = df.apply(
        lambda r: compute_resting_hr(r['HR_base'], r['Stress_pct'], r['Anxiety_pct'], r['Activity_min_wk']),
        axis=1
    )
    df['Blood_sugar'] = df['Glucemia']

    # 5) Selección y exportación
    out = df[[
        'Edad','Sexo_num','Diabetes','Hipertension','BMI','BMI_norm',
        'BP_systolic','LDL','HDL','Glucemia','Blood_sugar','HR_rest',
        'FamHyper','FamInfarct','Stress_pct','Anxiety_pct',
        'Activity_min_wk','Alcohol_wk','Diet_q','SatFat_wk','Sugary_wk','Salt_wk',
        'Harmful_habits','Tobacco','Vaping','Drugs'
    ]]
    out.to_csv('variables_generadas.csv', index=False)
