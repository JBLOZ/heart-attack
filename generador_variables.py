import pandas as pd
import numpy as np
import re

# Mapping functions for survey categories
def map_weekly_freq(cat:str)->float:
    if pd.isna(cat): return 0.0
    cat = cat.lower()
    if 'nunca' in cat: return 0.0
    if 'casi nunca' in cat: return 0.1
    if 'todos los días' in cat or 'a diario' in cat: return 7.0
    if '1-2' in cat or '1-2' in cat: return 1.5
    if '3-4' in cat or '3-4' in cat: return 3.5
    if '5 o más' in cat or '5 o mas' in cat: return 6.0
    # fallback: extract numbers
    nums = re.findall(r"(\d+)", cat)
    if nums:
        return float(nums[0])
    return 0.0


def parse_heart_rate(cat:str)->float:
    if pd.isna(cat): return np.nan
    cat = cat.lower()
    if 'menor a 60' in cat: return 55.0
    if 'entre 60' in cat and '70' in cat: return 65.0
    if 'entre 70' in cat and '80' in cat: return 75.0
    if 'entre 80' in cat and '90' in cat: return 85.0
    # fallback midpoint
    nums = re.findall(r"(\d+)", cat)
    if len(nums)>=2:
        return (float(nums[0])+float(nums[1]))/2
    return np.nan


def map_diet_quality(cat:str)->float:
    if pd.isna(cat): return 0.5
    cat = cat.lower()
    if 'saludable' in cat: return 1.0
    if 'moderada' in cat: return 0.5
    if 'pobre' in cat: return 0.0
    return 0.5


def map_yes_no_maybe(cat:str)->float:
    if pd.isna(cat): return 0.0
    cat = cat.lower()
    if 'sí' in cat: return 1.0
    if 'tal vez' in cat or 'quizá' in cat: return 0.5
    return 0.0

# Biomedical variable computations
def compute_bmi(weight: float, height_cm: float) -> float:
    h = height_cm / 100.0
    return weight / (h * h) if h > 0 else np.nan

def normalize_bmi_series(bmi_series: pd.Series) -> pd.Series:
    mi, ma = bmi_series.min(), bmi_series.max()
    return (bmi_series - mi) / (ma - mi)


def compute_systolic_bp(age:float, bmi_norm:float, salt_wk:float, diet_q:float, activity_min:float)->float:
    base = 110.0
    val = base + age*0.5 + bmi_norm*15 - activity_min/200 + salt_wk*2 + (1-diet_q)*5
    return val + np.random.normal(0,5)


def compute_ldl(bmi_norm:float, satfat_wk:float, diet_q:float)->float:
    base = 100.0
    val = base + bmi_norm*20 + satfat_wk*5 + (1-diet_q)*10
    return val + np.random.normal(0,10)


def compute_hdl(bmi_norm:float, activity_min:float, diet_q:float)->float:
    base = 50.0
    val = base - bmi_norm*10 + activity_min/200 + diet_q*5
    return val + np.random.normal(0,5)


def compute_glucose(bmi_norm:float, sugary_wk:float, activity_min:float)->float:
    base = 90.0
    val = base + bmi_norm*20 + sugary_wk*2 - activity_min/120
    return val + np.random.normal(0,5)


def compute_resting_hr(parsed_hr:float, stress_pct:float, anxiety_pct:float, activity_min:float)->float:
    val = parsed_hr + stress_pct*0.1 + anxiety_pct*0.05 - activity_min/100
    return val + np.random.normal(0,3)

# Main script
if __name__ == '__main__':
    df = pd.read_csv('encuesta.csv', sep=';')
    # clean numeric fields
    # clean peso: replace '-' or empty with NaN then convert
    df['Peso'] = df['Rango de peso: ¿Cuál es su peso actual en kg? (solo el numero, sin decimales)']\
        .astype(str).str.replace(',', '.')\
        .replace({'-': np.nan, '': np.nan})\
        .astype(float)
    # clean estatura: replace '-' or empty with NaN then convert
    df['Estatura'] = df['Rango de estatura: ¿Cuál es su estatura actual en cm? (solo el numero, sin comas son cm)']\
        .astype(str).str.replace(',', '.')\
        .replace({'-': np.nan, '': np.nan})\
        .astype(float)
    df['Edad'] = df['Edad:']
    df['Sexo_num'] = df['Sexo:'].map({'Masculino':1, 'Femenino':0}).fillna(0)
    df['Diabetes'] = df['Diabetes: ¿Padece de diabetes?'].map(lambda x: map_yes_no_maybe(x))
    df['Hipertension'] = df['Hipertensión arterial: ¿Padece o ha padecido hipertensión arterial (tensión alta)?'].map(lambda x: map_yes_no_maybe(x))
    # lifestyle maps
    df['Activity_days'] = df['Actividad física semanal: ¿Cuántos días a la semana realiza, al menos, 30 minutos de actividad física (moderada o intensa: caminar, nadar, etc.)?']\
        .map(lambda x: map_weekly_freq(x))
    df['Activity_min_wk'] = df['Activity_days'] * 30.0
    df['Diet_q'] = df['Calidad de dieta: ¿Cómo calificaría la calidad de su alimentación?']\
        .map(lambda x: map_diet_quality(x))
    df['SatFat_wk'] = df['Consumo de grasas saturadas: ¿Con qué frecuencia consume alimentos ricos en grasas saturadas (hamburguesas, patatas fritas, carnes rojas con grasa, quesos curados, etc.)?']\
        .map(lambda x: map_weekly_freq(x))
    df['Sugary_wk'] = df['Bebidas azucaradas: ¿Cuántas veces por semana consume refrescos, jugos envasados o bebidas energéticas azucaradas?']\
        .map(lambda x: map_weekly_freq(x))
    df['Salt_wk'] = df['Uso de sal en la dieta: ¿Con qué frecuencia usa sal de adición para las comidas o ingiere snacks salados?']\
        .map(lambda x: map_weekly_freq(x))
    # genetic/psych
    df['FamHyper'] = df['Historial familiar: ¿Tiene antecedentes familiares de hipertensión arterial (tensión alta)?']\
        .map(lambda x: map_yes_no_maybe(x))
    df['FamInfarct'] = df['Historial familiar: ¿Tiene antecedentes familiares de infarto de miocardio?']\
        .map(lambda x: map_yes_no_maybe(x))
    df['Anxiety_pct'] = df['Ansiedad: En una escala de 0 a 10, ¿padece o ha padecido ansiedad en el último año?']\
        .astype(float) * 10.0
    df['Stress_pct'] = df['Estrés: En una escala de 0 a 10, ¿padece o ha padecido estrés en el último año?']\
        .astype(float) * 10.0
    # compute derived variables
    df['BMI']     = df.apply(lambda r: compute_bmi(r['Peso'], r['Estatura']), axis=1)
    df['BMI_norm']= normalize_bmi_series(df['BMI'])
    df['BP_systolic'] = df.apply(lambda r: compute_systolic_bp(r['Edad'], r['BMI_norm'], r['Salt_wk'], r['Diet_q'], r['Activity_min_wk']), axis=1)
    # compute LDL row-wise to include diet quality per row
    df['LDL'] = df.apply(lambda r: compute_ldl(r['BMI_norm'], r['SatFat_wk'], r['Diet_q']), axis=1)
    df['HDL'] = df.apply(lambda r: compute_hdl(r['BMI_norm'], r['Activity_min_wk'], r['Diet_q']), axis=1)
    df['Glucemia'] = df.apply(lambda r: compute_glucose(r['BMI_norm'], r['Sugary_wk'], r['Activity_min_wk']), axis=1)
    # resting HR parse and compute
    df['HR_base'] = df['Frecuencia cardiaca: Si conoce el valor de sus pulsaciones en reposo, marque la opción más adecuada']\
        .map(lambda x: parse_heart_rate(x))
    df['HR_rest'] = df.apply(lambda r: compute_resting_hr(r['HR_base'], r['Stress_pct'], r['Anxiety_pct'], r['Activity_min_wk']), axis=1)
    # alias blood sugar
    df['Blood_sugar'] = df['Glucemia']
    # select output columns
    out = df[['Edad','Sexo_num','Diabetes','Hipertension','BMI','BMI_norm',
              'BP_systolic','LDL','HDL','Glucemia','Blood_sugar','HR_rest',
              'FamHyper','FamInfarct','Stress_pct','Anxiety_pct',
              'Activity_min_wk','Diet_q', 'SatFat_wk','Sugary_wk']]
    # export
    out.to_csv('variables_generadas.csv', index=False)
