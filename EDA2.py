import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# 1) Load
file_path = r"C:\Users\diego\OneDrive\Escritorio\TESIS\datasets\UserBehavior.csv"
cols = ['user_id','item_id','category_id','behavior_type','timestamp']
df = pd.read_csv(file_path, names=cols)

# 2) Keep only page‐views
pv = df[df['behavior_type']=='pv']

# 3) Count (user,category) pairs
uc_counts = pv.groupby(['user_id','category_id']).size()

# 4) Compute entropy per user:
#    - normalize each user's category counts into probabilities
#    - apply scipy.stats.entropy on each probability vector
user_entropy = uc_counts.groupby(level=0).apply(
    lambda counts: entropy(counts/counts.sum(), base=2)
)

# 5) Compute Gini per user:
def gini(x):
    # x is a numpy array of non‐neg counts
    a = np.sort(x)
    n = len(a)
    idx = np.arange(1,n+1)
    return (2*np.sum(idx*a)/(n*a.sum())) - (n+1)/n

user_gini = uc_counts.groupby(level=0).apply(
    lambda counts: gini(counts.values)
)

# 6) Plot entropy distribution (log‐x)
plt.figure(figsize=(8,5))
sns.histplot(user_entropy, bins=50, kde=True, color='skyblue')
plt.xscale('log')
plt.title('User Category Entropy (log scale)')
plt.xlabel('Entropy (bits)')
plt.ylabel('Number of Users')
plt.show()

# 7) Plot Gini distribution
plt.figure(figsize=(8,5))
sns.histplot(user_gini, bins=50, kde=True, color='coral')
plt.title('User Category Gini')
plt.xlabel('Gini Coefficient')
plt.ylabel('Number of Users')
plt.show()






