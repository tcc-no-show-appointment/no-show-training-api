## 📦 Imports
import numpy as np
import pandas as pd
from collections import Counter
import warnings

# Scikit-learn (modelagem e pré-processamento)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

# Encoders de categóricas
try:
    from category_encoders import TargetEncoder
except Exception as e:
    raise RuntimeError("Falta 'category_encoders'. Instale com: pip install category_encoders")

# Balanceamento
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)


## 📥 Carregar a Base 
CSV_PATH = "noshowappointments.csv"  # ajuste o caminho se necessário
df = pd.read_csv(CSV_PATH)
print("Dimensão:", df.shape)
df.head()

## 🧹 Limpeza & 🧪 Engenharia de Atributos
# PatientId para inteiro
try:
    df["PatientId"] = df["PatientId"].astype("int64")
except Exception:
    df["PatientId"] = df["PatientId"].astype(str).str.replace(r"\\.0$", "", regex=True)
    df["PatientId"] = pd.to_numeric(df["PatientId"], errors="coerce").astype("Int64")

# Alvo binário
if df["No-show"].dtype not in (np.int64, np.int32, np.int8):
    df["No-show"] = df["No-show"].map({"No": 0, "Yes": 1}).astype(int)

# Idade válida
df = df[(df["Age"] >= 0) & (df["Age"] <= 100)].copy()

# Datas e remoção de nulos
df["ScheduledDay"]   = pd.to_datetime(df["ScheduledDay"], errors="coerce")
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")
df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])

# waiting_days e remoção de negativos
df["waiting_days"] = (df["AppointmentDay"].dt.floor("D") - df["ScheduledDay"].dt.floor("D")).dt.days
neg_count = (df["waiting_days"] < 0).sum()
df = df[df["waiting_days"] >= 0].copy()

# Atributos temporais
df["scheduled_hour"] = df["ScheduledDay"].dt.hour
df["appointment_weekday"] = df["AppointmentDay"].dt.dayofweek

# Handcap binário
df["Handcap"] = (df["Handcap"] > 0).astype(int)

print("Registros removidos por waiting_days negativo:", neg_count)
df.head()

## 🧪 Novas Features
# Buckets de espera
bins = [-1, 0, 3, 7, 14, 10**9]
labels = ["wait_0", "wait_1_3", "wait_4_7", "wait_8_14", "wait_15p"]
df["waiting_days_bucket"] = pd.cut(df["waiting_days"], bins=bins, labels=labels)

# Weekend (0/1)
df["is_weekend"] = df["AppointmentDay"].dt.dayofweek.isin([5,6]).astype(int)

# Partes do dia
df["day_part_morning"]   = ((df["scheduled_hour"] >= 6)  & (df["scheduled_hour"] < 12)).astype(int)
df["day_part_afternoon"] = ((df["scheduled_hour"] >= 12) & (df["scheduled_hour"] < 18)).astype(int)
df["day_part_evening"]   = ((df["scheduled_hour"] >= 18) | (df["scheduled_hour"] < 6)).astype(int)

# Codificação cíclica
df["weekday_sin"] = np.sin(2*np.pi*df["appointment_weekday"]/7)
df["weekday_cos"] = np.cos(2*np.pi*df["appointment_weekday"]/7)

# Interação
df["wait_x_sms"] = df["waiting_days"] * df["SMS_received"]

## 🧩 Definir **Features** (X) e **Target** (y)

drop_cols = ["No-show", "PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"]
X = df.drop(columns=drop_cols, errors="ignore").copy()
y = df["No-show"].copy()

num_features      = ["Age", "waiting_days", "scheduled_hour", "weekday_sin", "weekday_cos", "wait_x_sms"]
cat_low_features  = ["Gender", "waiting_days_bucket"]
cat_high_features = ["Neighbourhood"]
bin_features      = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received",
                     "is_weekend", "day_part_morning", "day_part_afternoon", "day_part_evening"]

for c in num_features + cat_low_features + cat_high_features + bin_features:
    if c not in X.columns:
        raise KeyError(f"Coluna esperada não encontrada: {c}")

X.head()


## 🏗️ Pré-processamento & ✂️ Split
try:
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("gender_waitbucket", ohe, cat_low_features),
        ("neigh", TargetEncoder(), cat_high_features),
        ("bin", "passthrough", bin_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Distribuição (treino):", Counter(y_train))

## ⚖️ Otimização de Memória
from sklearn.preprocessing import FunctionTransformer
def to_float32(X):
    return X.astype(np.float32)

astype32 = FunctionTransformer(to_float32)

## 🎛️ Tuning Leve — **GridSearchCV** com **Cross-Validation**
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# RandomForest com class_weight (memória mais leve na CV)
rf_cv = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("astype32", astype32),
    ("clf", RandomForestClassifier(
        n_estimators=120, max_depth=None, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=1, random_state=42
    ))
])
rf_grid = {
    "clf__n_estimators": [100, 180],
    "clf__max_depth": [None, 12],
    "clf__min_samples_leaf": [10, 20],
}
rf_search = GridSearchCV(rf_cv, rf_grid, scoring="average_precision", cv=cv, n_jobs=1, verbose=0)
rf_search.fit(X_train, y_train)
y_pred_rf_cv = rf_search.predict(X_test)
y_prob_rf_cv = rf_search.predict_proba(X_test)[:, 1]
print("RF GridSearch — best params:", rf_search.best_params_)
print("RF GridSearch — PR-AUC:", average_precision_score(y_test, y_prob_rf_cv))

# Salvar o modelo treinado (RF Grid) em um arquivo .pkl
import pickle
rf_grid_model = rf_search.best_estimator_  # melhor pipeline do GridSearchCV
with open("rf_grid_model.pkl", "wb") as f:
    pickle.dump(rf_grid_model, f)
print("Modelo RF (Grid) salvo como rf_grid_model.pkl")