# eda_3_transitions_churn_lifespan.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ————————————————
# 0) Ajusta esta ruta a tu CSV:
file_path = r"C:\Users\diego\OneDrive\Escritorio\TESIS\datasets\UserBehavior.csv"

# 1) Cargar y preparar el DataFrame
column_names = ['user_id','item_id','category_id','behavior_type','timestamp']
df = pd.read_csv(file_path, names=column_names)

# 2) Crear columna datetime y filtrar solo page-views
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
pv = df[df['behavior_type']=='pv'].copy()
# ————————————————

# 1) MATRIZ DE TRANSICIONES CATEGORÍA → CATEGORÍA
pv.sort_values(['user_id','datetime'], inplace=True)
pv['next_cat'] = pv.groupby('user_id')['category_id'].shift(-1)

transitions = (
    pv[['category_id','next_cat']]
      .dropna()
      .groupby(['category_id','next_cat'])
      .size()
      .unstack(fill_value=0)
)
trans_probs = transitions.div(transitions.sum(axis=1), axis=0)

top_cats = pv['category_id'].value_counts().head(10).index
tp = trans_probs.loc[top_cats, top_cats]

plt.figure(figsize=(8,6))
sns.heatmap(tp, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title("Category Transition Probabilities (Top 10 Categories)")
plt.xlabel("Next Category")
plt.ylabel("Current Category")
plt.tight_layout()
plt.show()


# 2) CHURN SEMANAL: fracción de categorías nuevas cada semana
# Muestreamos 10k usuarios para no petar la memoria
sample_users = np.random.choice(pv['user_id'].unique(), size=10000, replace=False)
pv_samp = pv[pv['user_id'].isin(sample_users)].copy()
pv_samp['week'] = pv_samp['datetime'].dt.to_period('W')

fractions = []
for uid, grp in pv_samp.groupby('user_id'):
    seen = set()
    for week, sub in grp.groupby('week'):
        cats = set(sub['category_id'])
        if not cats:
            continue
        new = len(cats - seen)
        frac = new / len(cats)
        fractions.append(frac)
        seen |= cats

plt.figure(figsize=(8,4))
sns.histplot(fractions, bins=50, kde=True, color='teal')
plt.title("Fraction of New Categories per User/Week (sampled 10k users)")
plt.xlabel("New-Category Fraction")
plt.ylabel("Count of User-Weeks")
plt.tight_layout()
plt.show()


# 3) LIFESPAN DEL ÍTEM: días entre primera y última vista
first_last = pv.groupby('item_id')['datetime'].agg(['min','max'])
first_last['lifespan_days'] = (first_last['max'] - first_last['min']).dt.days + 1

# Clip a 365d para el histograma lineal
ll = first_last['lifespan_days']
ll_cap = ll.clip(upper=365)

# Histograma lineal con límite x=100 para zoom en el grueso
plt.figure(figsize=(8,4))
sns.histplot(ll_cap, bins=50, color='purple')
plt.xlim(0, 100)
plt.title("Item Lifespan (dias), cap 365d, zoom 0–100d")
plt.xlabel("Lifespan (días)")
plt.ylabel("Número de ítems")
plt.tight_layout()
plt.show()

# Histograma log escala para toda la cola
plt.figure(figsize=(8,4))
sns.histplot(ll_cap, bins=100, log_scale=(False, True), color='purple')
plt.title("Item Lifespan (días), cap 365d, escala y-log")
plt.xlabel("Lifespan (días)")
plt.ylabel("Número de ítems (log escala)")
plt.tight_layout()
plt.show()



t_xlabel("Días")
axes[0].set_ylabel("Nº ítems")

# b) Resto 8–365 días
sns.histplot(ll_cap[ll_cap>=8], bins=50, ax=axes[1], color='purple')
axes[1].set_xlim(8,365)
axes[1].set_title("Lifespan (días) cap 365, rango 8–365d")
axes[1].set_xlabel("Días")
axes[1].set_ylabel("Nº ítems")

plt.tight_layout()
plt.show()

# Ahora la vista log1p de todo el rango
plt.figure(figsize=(8,4))
sns.histplot(ll_log1p, bins=50, color='purple')
plt.title("Histograma log1p(Lifespan cap 365d)")
plt.xlabel("log1p(Días)")
plt.ylabel("Nº ítems")
plt.tight_layout()
plt.show()
