import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.patheffects as path_effects

# Leer datos
csv = 'variables_generadas.csv'
df = pd.read_csv(csv)

# Crear carpeta docs si no existe
os.makedirs('docs', exist_ok=True)

# Paleta de colores
palette = sns.color_palette('tab10', 10)

def pie_bonito(labels, counts, colors, title, filename):
    # Filtrar valores 0
    filtered = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if not filtered:
        print(f"No hay datos para mostrar en {title}")
        return
    labels, counts, colors = zip(*filtered)
    total = sum(counts)
    explode = [0.05]*len(counts)  # pequeño efecto de separación

    fig, ax = plt.subplots(figsize=(7,7))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle=60,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.65, edgecolor='white', linewidth=2),  # Más grosor, menos centro
        textprops=dict(color="black", fontsize=15, weight='bold')
    )
    # Mejorar los textos de porcentaje
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')
        autotext.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="black"),
            path_effects.Normal()
        ])
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('bold')
    ax.set_title(title, fontsize=20, weight='bold')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Edad
plt.figure(figsize=(7,5))
bins = np.linspace(df['Edad'].min(), df['Edad'].max(), 11)
n, bins, patches = plt.hist(df['Edad'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Distribución de la Edad')
plt.xlabel('Edad (años)')
plt.ylabel('Número de personas')
plt.xticks(bins.astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/edad.png')
plt.close()

# 2. Sexo (círculo)
labels = ['Femenino', 'Masculino']
counts = df['Sexo_num'].value_counts().sort_index()
pie_bonito(labels, counts, [palette[0], palette[1]], 'Distribución por Sexo', 'docs/sexo.png')


# 3. Diabetes (círculo)
diab_labels = ['No', 'Tal vez', 'Sí']
diab_counts = [sum(df['Diabetes'] == 0.0), sum(df['Diabetes'] == 0.5), sum(df['Diabetes'] == 1.0)]
pie_bonito(diab_labels, diab_counts, [palette[2], palette[3], palette[4]], 'Distribución de Diabetes', 'docs/diabetes.png')


# 4. Hipertensión (círculo)
hip_labels = ['No', 'Tal vez', 'Sí']
hip_counts = [sum(df['Hipertension'] == 0.0), sum(df['Hipertension'] == 0.5), sum(df['Hipertension'] == 1.0)]
pie_bonito(hip_labels, hip_counts, [palette[5], palette[6], palette[7]], 'Distribución de Hipertensión', 'docs/hipertension.png')


# 5. IMC
plt.figure(figsize=(7,5))
bins = np.linspace(df['BMI'].min(), df['BMI'].max(), 11)
n, bins, patches = plt.hist(df['BMI'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Distribución del IMC')
plt.xlabel('IMC (kg/m²)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/imc.png')
plt.close()

# 6. IMC normalizado
plt.figure(figsize=(7,5))
bins = np.arange(0, 1.1, 0.1)
n, bins, patches = plt.hist(df['BMI_norm'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('IMC Normalizado')
plt.xlabel('IMC normalizado')
plt.ylabel('Número de personas')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/imc_normalizado.png')
plt.close()

# 7. Presión arterial sistólica
plt.figure(figsize=(7,5))
bins = np.linspace(df['BP_systolic'].min(), df['BP_systolic'].max(), 11)
n, bins, patches = plt.hist(df['BP_systolic'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Presión Arterial Sistólica (mmHg)')
plt.xlabel('Presión arterial (mmHg)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/presion_arterial.png')
plt.close()

# 8. Colesterol LDL
plt.figure(figsize=(7,5))
bins = np.linspace(df['LDL'].min(), df['LDL'].max(), 11)
n, bins, patches = plt.hist(df['LDL'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Colesterol LDL (mg/dL)')
plt.xlabel('LDL (mg/dL)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/colesterol_ldl.png')
plt.close()

# 9. Colesterol HDL
plt.figure(figsize=(7,5))
bins = np.linspace(df['HDL'].min(), df['HDL'].max(), 11)
n, bins, patches = plt.hist(df['HDL'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Colesterol HDL (mg/dL)')
plt.xlabel('HDL (mg/dL)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/colesterol_hdl.png')
plt.close()

# 10. Glucemia
plt.figure(figsize=(7,5))
bins = np.linspace(df['Glucemia'].min(), df['Glucemia'].max(), 11)
n, bins, patches = plt.hist(df['Glucemia'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Glucemia en Ayunas (mg/dL)')
plt.xlabel('Glucemia (mg/dL)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/glucemia.png')
plt.close()

# 11. Azúcar en sangre (alias)
plt.figure(figsize=(7,5))
bins = np.linspace(df['Blood_sugar'].min(), df['Blood_sugar'].max(), 11)
n, bins, patches = plt.hist(df['Blood_sugar'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Azúcar en Sangre (mg/dL)')
plt.xlabel('Azúcar en sangre (mg/dL)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/azucar_sangre.png')
plt.close()

# 12. Frecuencia cardíaca en reposo
plt.figure(figsize=(7,5))
bins = np.linspace(df['HR_rest'].min(), df['HR_rest'].max(), 11)
n, bins, patches = plt.hist(df['HR_rest'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Frecuencia Cardíaca en Reposo (lpm)')
plt.xlabel('Frecuencia cardíaca (lpm)')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/frecuencia_cardiaca.png')
plt.close()

# 13. Historia familiar hipertensión (círculo)
hist_hip_labels = ['No', 'Tal vez', 'Sí']
hist_hip_counts = [sum(df['FamHyper'] == 0.0), sum(df['FamHyper'] == 0.5), sum(df['FamHyper'] == 1.0)]
pie_bonito(hist_hip_labels, hist_hip_counts, [palette[8], palette[9], palette[0]], 'Historia Familiar de Hipertensión', 'docs/historia_hipertension.png')


# 14. Historia familiar infarto (círculo)
hist_inf_labels = ['No', 'Tal vez', 'Sí']
hist_inf_counts = [sum(df['FamInfarct'] == 0.0), sum(df['FamInfarct'] == 0.5), sum(df['FamInfarct'] == 1.0)]
pie_bonito(hist_inf_labels, hist_inf_counts, [palette[1], palette[2], palette[3]], 'Historia Familiar de Infarto', 'docs/historia_infarto.png')


# 15. Estrés (%)
plt.figure(figsize=(7,5))
bins = np.arange(0, 101, 10)
n, bins, patches = plt.hist(df['Stress_pct'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Estrés (%)')
plt.xlabel('Estrés (%)')
plt.ylabel('Número de personas')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/estres.png')
plt.close()

# 16. Ansiedad (%)
plt.figure(figsize=(7,5))
bins = np.arange(0, 101, 10)
n, bins, patches = plt.hist(df['Anxiety_pct'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Ansiedad (%)')
plt.xlabel('Ansiedad (%)')
plt.ylabel('Número de personas')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/ansiedad.png')
plt.close()

# 17. Actividad física semanal (min)
plt.figure(figsize=(7,5))
bins = np.linspace(df['Activity_min_wk'].min(), df['Activity_min_wk'].max(), 11)
n, bins, patches = plt.hist(df['Activity_min_wk'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Actividad Física Semanal (min)')
plt.xlabel('Minutos/semana')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/actividad_fisica.png')
plt.close()

# 18. Calidad de la dieta
plt.figure(figsize=(7,5))
bins = np.arange(0, 1.1, 0.1)
n, bins, patches = plt.hist(df['Diet_q'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Calidad de la Dieta (escala)')
plt.xlabel('Calidad de dieta')
plt.ylabel('Número de personas')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/calidad_dieta.png')
plt.close()

# 19. Consumo de grasas saturadas
plt.figure(figsize=(7,5))
bins = np.linspace(df['SatFat_wk'].min(), df['SatFat_wk'].max(), 11)
n, bins, patches = plt.hist(df['SatFat_wk'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Consumo de Grasas Saturadas (veces/semana)')
plt.xlabel('Veces por semana')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/grasas_saturadas.png')
plt.close()

# 20. Consumo de bebidas azucaradas
plt.figure(figsize=(7,5))
bins = np.linspace(df['Sugary_wk'].min(), df['Sugary_wk'].max(), 11)
n, bins, patches = plt.hist(df['Sugary_wk'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Consumo de Bebidas Azucaradas (veces/semana)')
plt.xlabel('Veces por semana')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/bebidas_azucaradas.png')
plt.close()

# 21. Consumo de alcohol
plt.figure(figsize=(7,5))
bins = np.linspace(df['Alcohol_wk'].min(), df['Alcohol_wk'].max(), 11)
n, bins, patches = plt.hist(df['Alcohol_wk'], bins=bins, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(palette[i % len(palette)])
plt.title('Consumo de Alcohol (veces/semana)')
plt.xlabel('Veces por semana')
plt.ylabel('Número de personas')
plt.xticks(bins.round(0).astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/alcohol.png')
plt.close()

