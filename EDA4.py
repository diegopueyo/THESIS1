# eda_4_more.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# ————————————————
# 1) Carga tu DataFrame (ajusta la ruta a donde tengas tu CSV)
file_path = r"C:\Users\diego\OneDrive\Escritorio\TESIS\datasets\UserBehavior.csv"
column_names = ['user_id','item_id','category_id','behavior_type','timestamp']
df = pd.read_csv(file_path, names=column_names)

# 2) Prepara datetime y filtra sólo page‐views
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
pv = df[df['behavior_type']=='pv'].sort_values(['user_id','datetime'])
# ————————————————
# 1) Duración de sesiones
pv = pv.sort_values(['user_id','datetime'])
pv['gap'] = pv.groupby('user_id')['datetime'] \
              .diff().dt.total_seconds().fillna(0)
pv['new_sess'] = (pv['gap'] > 30*60).astype(int)
pv['session_id'] = pv.groupby('user_id')['new_sess'].cumsum()
sess_len = pv.groupby(['user_id','session_id']) \
             .size().reset_index(name='len')
plt.figure(figsize=(6,4))
sns.histplot(sess_len['len'], bins=50, log_scale=(False, True), color='steelblue')
plt.title("Distribución de longitud de sesiones")
plt.xlabel("Acciones por sesión")
plt.ylabel("Frecuencia (eje-y log)")
plt.show()

# 2) Hora del día y día de la semana
pv['hour'] = pv['datetime'].dt.hour
pv['weekday'] = pv['datetime'].dt.day_name()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
sns.countplot(x='hour', data=pv, ax=ax1, color='teal')
sns.countplot(x='weekday', data=pv, ax=ax2,
              order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
              color='teal')
ax1.set_title("Clicks por hora del día")
ax2.set_title("Clicks por día de la semana")
plt.tight_layout()
plt.show()

# 3) Top‐10 categorías (usuarios top 1%)
top_1pct = pv['user_id'].value_counts().nlargest(
    int(pv['user_id'].nunique()*0.01)
).index
top10cats = pv[pv['user_id'].isin(top_1pct)]['category_id'] \
               .value_counts().nlargest(10).index
plt.figure(figsize=(6,4))
sns.countplot(y='category_id',
              data=pv[pv['user_id'].isin(top_1pct)],
              order=top10cats,
              palette='viridis',
              hue=pv['category_id'])  # fija hue para evitar warning
plt.title("Top 10 categorías vistas por usuarios top 1%")
plt.xlabel("Conteo de clicks")
plt.ylabel("Categoría")
plt.legend([],[], frameon=False)
plt.show()

# 4) Diversidad de cada sesión
sess_div = pv.groupby(['user_id','session_id'])['category_id'] \
             .nunique().reset_index(name='diversity')
plt.figure(figsize=(6,4))
sns.histplot(sess_div['diversity'], bins=30, color='orange')
plt.title("Diversidad (# categorías) por sesión")
plt.xlabel("Número de categorías distintas")
plt.ylabel("Frecuencia")
plt.show()

# 5) Delay entre acciones
delays = pv['gap'][pv['gap']>0]
print("Percentiles de delay (s):",
      delays.quantile([0.5, 0.75, 0.9, 0.99]).to_dict())
plt.figure(figsize=(6,4))
sns.kdeplot(delays, log_scale=True, fill=True)
plt.title("Densidad de delay entre acciones (eje-x log)")
plt.xlabel("Segundos entre clicks")
plt.ylabel("Densidad")
plt.show()

# 6) Clicks vs ítems únicos por usuario
u_stats = pv.groupby('user_id').agg(
    total_clicks=('item_id','size'),
    unique_items=('item_id','nunique')
).reset_index()
plt.figure(figsize=(6,6))
sample_users = u_stats.sample(5000, random_state=42)
sns.scatterplot(x='unique_items', y='total_clicks',
                data=sample_users, alpha=0.3)
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Ítems únicos (log)")
plt.ylabel("Total clicks (log)")
plt.title("Clicks vs Ítems únicos (5k usuarios muestreados)")
plt.show()

# 7) Evolución semanal de nuevas categorías
pv['week'] = pv['datetime'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_frac = pv.groupby('week')['category_id'] \
                .apply(lambda s: s.nunique()/s.count()) \
                .reset_index(name='frac')
plt.figure(figsize=(8,4))
sns.lineplot(x='week', y='frac', data=weekly_frac, marker='o')
plt.xticks(rotation=45)
plt.title("Ratio categorías únicas / clicks por semana")
plt.xlabel("Semana")
plt.ylabel("Fracción únicas")
plt.tight_layout()
plt.show()

# 8) Heatmap usuarios top50 vs hora
top50 = pv['user_id'].value_counts().head(50).index
hm = pv[pv['user_id'].isin(top50)].pivot_table(
    index='user_id', columns='hour', aggfunc='size', fill_value=0
)
plt.figure(figsize=(8,10))
sns.heatmap(hm, cmap='Blues')
plt.title("Heatmap clicks: TOP 50 usuarios vs hora")
plt.xlabel("Hora del día")
plt.ylabel("Usuario ID")
plt.show()

# 9) Retención de cohortes (semanas ISO 1–10 de 2017)
pv['year'] = pv['datetime'].dt.year
pv17 = pv[pv['year']==2017].copy()
pv17['signup_wk'] = pv17.groupby('user_id')['week'] \
                         .transform('min')
pv17['lag_wk'] = ((pv17['week'] - pv17['signup_wk'])
                   .dt.days // 7).astype(int)
# Filtrar primeras 10 semanas de signup y lag<10
mask = (pv17['signup_wk'].dt.isocalendar().week.between(1,10)) & \
       (pv17['lag_wk'].between(0,9))
coh = pv17[mask].groupby(
    [pv17['signup_wk'].dt.isocalendar().week, 'lag_wk']
)['user_id'].nunique().unstack(fill_value=0)
coh = coh.divide(coh.iloc[:,0], axis=0)
plt.figure(figsize=(8,6))
sns.heatmap(coh, cmap='Reds', annot=True, fmt=".2f")
plt.title("Retención cohortes 2017 (semanas 1–10)")
plt.xlabel("Semanas desde signup")
plt.ylabel("Semana ISO de signup")
plt.show()

# 10) Top-10 pares consecutivos (muestra 200K filas)
pv_samp = pv.sample(200_000, random_state=42) \
            .sort_values(['user_id','datetime'])
pv_samp['next_item'] = pv_samp.groupby('user_id') \
                               ['item_id'].shift(-1)
pairs = (pv_samp.dropna(subset=['next_item'])
         .groupby(['item_id','next_item']).size()
         .reset_index(name='cnt'))
print("Top 10 pares consecutivos:\n",
      pairs.nlargest(10,'cnt'))