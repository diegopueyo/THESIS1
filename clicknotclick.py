# clicknotclick.py

import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Ruta a tu CSV original
DATA_FILE = r"C:\Users\diego\OneDrive\Escritorio\TESIS\datasets\UserBehavior.csv"

# 2) Carga completa (sin header)
df = pd.read_csv(
    DATA_FILE,
    names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
)

# 3) Definimos exposiciones = page-views (behavior_type == 'pv')
impressions = (
    df[df['behavior_type']=='pv']
    .loc[:, ['user_id','item_id','timestamp']]
    .drop_duplicates()
)

# 4) Definimos interacciones “positivas”
#    (compras, carrito o favoritos según tu dataset)
positive = (
    df[df['behavior_type'].isin(['buy','cart','fav'])]
    .loc[:, ['user_id','item_id']]
    .drop_duplicates()
    .assign(clicked=1)
)

# 5) Asociamos cada impresión con si luego hubo click o no
imp = impressions.merge(
    positive,
    on=['user_id','item_id'],
    how='left'
).fillna({'clicked': 0})

# 6) Separamos positivos / negativos
pos = imp[imp['clicked']==1][['user_id','item_id','clicked']]
neg = imp[imp['clicked']==0].sample(n=len(pos), random_state=42)[['user_id','item_id','clicked']]

data = pd.concat([pos,neg]).reset_index(drop=True)

# 7) Train/test split estratificado
train, test = train_test_split(
    data,
    test_size=0.2,
    stratify=data['clicked'],
    random_state=42
)

# 8) Guardamos los CSV
train.to_csv("train.csv", index=False)
test.to_csv ("test.csv",  index=False)

print(f"Positivos: {len(pos)}, Negativos muestreados: {len(neg)}")
print(f"Train: {train.shape}, Test: {test.shape}")
print("¡Hecho!")
