# recsys.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 1) Cargar train/test
# ----------------------------
train = pd.read_csv("train.csv")   # user_id,item_id,clicked
test  = pd.read_csv("test.csv")    # mismo formato

# nos interesa sólo los positivos de test para evaluación
test_pos = test[test.clicked==1].groupby("user_id")["item_id"].agg(set).to_dict()

# todos los usuarios y items en train
all_users = train.user_id.unique()
all_items = train.item_id.unique()

# mappings user/item → índice 0…N-1
u2i = {u:idx for idx,u in enumerate(all_users)}
i2u = {idx:u for u,idx in u2i.items()}
v2i = {v:idx for idx,v in enumerate(all_items)}
i2v = {idx:v for v,idx in v2i.items()}

n_users = len(all_users)
n_items = len(all_items)

# ----------------------------
# 2) Construir la matriz usuario×item
# ----------------------------
rows = train.user_id.map(u2i)
cols = train.item_id.map(v2i)
data = np.ones(len(train), dtype=np.float32)

# matriz sparse de entrenamiento
M = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

# ----------------------------
# 3) Calcular similitud item–item
# ----------------------------
# calculamos similitud columna a columna:
# shape result: (n_items, n_items)
print("→ Calculando similitud coseno item–item… esto puede tardar un poco")
# para ahorrar memoria, lo hacemos on-the-fly por bloques
sim = cosine_similarity(M.T, dense_output=False)  # devuelve sparse matrix

# ----------------------------
# 4) Función de recomendación
# ----------------------------
def recommend_for_user(u, K=10):
    """
    Para un usuario u (user_id original), devuelve lista de top-K item_ids
    basadas en item-based CF.
    """
    if u not in u2i:
        return []
    ui = u2i[u]
    # vector binario de items vistos
    user_row = M[ui].toarray().ravel()  # shape (n_items,)
    # score = Σ_{i en vistos} sim[i]  (producto matricial)
    scores = user_row @ sim  # shape (n_items,)
    # quitamos ya vistos
    scores[user_row.nonzero()] = -1
    # pedimos top-K índices
    topk_idx = np.argpartition(-scores, K)[:K]
    topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
    return [i2v[idx] for idx in topk_sorted]

# ----------------------------
# 5) Evaluación Precision@K, Recall@K
# ----------------------------
def precision_recall_at_k(K=10):
    precisions = []
    recalls = []
    # iteramos sólo usuarios que aparecen en test_pos
    for u, actual_items in test_pos.items():
        preds = recommend_for_user(u, K)
        if not preds:
            continue
        hit = len(set(preds) & actual_items)
        precisions.append(hit / K)
        recalls.append(hit / len(actual_items))
    return np.mean(precisions), np.mean(recalls)

# ----------------------------
# 6) Ejecutamos y mostramos resultados
# ----------------------------
for K in [5, 10, 20]:
    p, r = precision_recall_at_k(K)
    print(f"Precision@{K}: {p:.4f}   Recall@{K}: {r:.4f}")
