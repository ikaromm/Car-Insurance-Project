# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# %%
train = pd.read_csv('train.csv')
# %%
train.isna().sum()
## Não possui valores nulos
# %%
numeric_columns_train = train.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(20, 15))

for i, column in enumerate(numeric_columns_train.columns):
    plt.subplot(5, 4, i + 1)
    sns.histplot(train[column], kde=False)
    plt.title(column)
    plt.tight_layout()

plt.show()

# %%
plt.figure(figsize=(20, 15))

# Iterar sobre as colunas numéricas
for i, column in enumerate(numeric_columns_train.columns):
    plt.subplot(5, 4, i + 1)
    sns.violinplot(x='is_claim', y=column, data=train)
    plt.title(column)
    plt.tight_layout()

# Exibir o gráfico
plt.show()
# %%
### Chart by Zubin Relia (From Kaggle)

fig, axes = plt.subplots(3,3, figsize=(30,24))
axes = axes.flatten()

cols_1=['area_cluster', 'model', 'fuel_type', 'is_speed_alert', 'ncap_rating', 'is_esc', 'is_parking_sensors', 'rear_brakes_type', 'is_brake_assist']

for i, column in enumerate(cols_1):
    ax = axes[i]  

  
    sns.countplot(data=train, x=column, ax=ax, hue='is_claim')
    ax.set_title(f'Countplot for {column}')
    
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=90)  
    
    

plt.tight_layout()


plt.show()
# %%
train.shape