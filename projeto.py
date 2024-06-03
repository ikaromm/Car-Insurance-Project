# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import pipeline
from sklearn import ensemble
from sklearn import model_selection

# %%
train = pd.read_csv("train.csv")
# %%
train.isna().sum()
## Não possui valores nulos
# %%
numeric_columns_train = train.select_dtypes(include=["float64", "int64"])

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
    sns.violinplot(x="is_claim", y=column, data=train)
    plt.title(column)
    plt.tight_layout()

# Exibir o gráfico
plt.show()
# %%
### Chart by Zubin Relia (From Kaggle)

fig, axes = plt.subplots(3, 3, figsize=(30, 24))
axes = axes.flatten()

cols_1 = [
    "area_cluster",
    "model",
    "fuel_type",
    "is_speed_alert",
    "ncap_rating",
    "is_esc",
    "is_parking_sensors",
    "rear_brakes_type",
    "is_brake_assist",
]

for i, column in enumerate(cols_1):
    ax = axes[i]

    sns.countplot(data=train, x=column, ax=ax, hue="is_claim")
    ax.set_title(f"Countplot for {column}")

    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90)


plt.tight_layout()


plt.show()
# %%
train.shape

# %%
heat = train[cols_1]

plt.figure(figsize=(25, 25))
sns.heatmap(numeric_columns_train.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# %%
train.dtypes

# Index(['policy_id', 'policy_tenure', 'age_of_car', 'age_of_policyholder',
#    'area_cluster', 'population_density', 'make', 'segment', 'model',
#    'fuel_type', 'max_torque', 'max_power', 'engine_type', 'airbags',
#    'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
#    'is_parking_camera', 'rear_brakes_type', 'displacement', 'cylinder',
#    'transmission_type', 'gear_box', 'steering_type', 'turning_radius',
#    'length', 'width', 'height', 'gross_weight', 'is_front_fog_lights',
#    'is_rear_window_wiper', 'is_rear_window_washer',
#    'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
#    'is_central_locking', 'is_power_steering',
#    'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
#    'is_ecw', 'is_speed_alert', 'ncap_rating', 'is_claim'],
#   dtype='object')
# %%
from funcs import Funcs

func = Funcs()

# %%
# train['segment'].value_counts()

func.group_less_2pct(train, "model")

train["model"].value_counts()
# %%

pipe = pipeline.Pipeline([
    ('MemSaver', func.reduce_mem_usage(train)),    
])


# %%

func.reduce_mem_usage(train)
# %%
