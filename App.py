import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Dashboard Churn - Marjane",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    with open(MODEL_DIR / "model_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)

    return model, scaler, features, model_info


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    required_columns = ["Recency", "Frequency", "Monetary"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier : " + ", ".join(missing_columns)
        )

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Les colonnes Recency, Frequency et Monetary doivent contenir des valeurs numériques."
        )

    data["Frequency"] = data["Frequency"].replace(0, np.nan)
    data["Montant_par_visite"] = data["Monetary"] / data["Frequency"]
    data["Recency_x_Frequency"] = data["Recency"] * data["Frequency"]
    data["Log_Montant"] = np.log1p(data["Monetary"])
    data["Log_Frequency"] = np.log1p(data["Frequency"])
    data = data.replace([np.inf, -np.inf], np.nan)

    if data.isna().any().any():
        raise ValueError(
            "Certaines lignes contiennent des valeurs invalides, par exemple Frequency = 0."
        )

    return data


def assign_risk(probability: float) -> str:
    if probability < 0.30:
        return "Faible"
    if probability < 0.60:
        return "Moyen"
    return "Élevé"


def risk_color(risk: str) -> str:
    colors = {
        "Faible": "#00A86B",
        "Moyen": "#F5A623",
        "Élevé": "#E74C3C",
    }
    return colors.get(risk, "#6B7280")


model, scaler, features, model_info = load_artifacts()

st.title("Dashboard Churn Client - Marjane")
st.caption(
    "Prédiction du risque de churn à partir des métriques RFM des clients."
)

with st.sidebar:
    st.header("Paramètres")
    st.write(f"Modèle utilisé : **{model_info.get('meilleur_modele', 'Modèle ML')}**")
    st.write(f"Seuil churn : **{model_info.get('seuil_churn_jours', 180)} jours**")
    st.divider()
    st.write("Colonnes obligatoires dans le fichier :")
    st.code("Recency, Frequency, Monetary")

uploaded_file = st.file_uploader(
    "Importer un fichier clients",
    type=["csv", "xlsx"],
    help="Le fichier doit contenir au minimum les colonnes Recency, Frequency et Monetary.",
)

if uploaded_file is None:
    st.info("Importe un fichier CSV ou Excel pour lancer la prédiction.")
    st.subheader("Exemple de format attendu")
    st.dataframe(
        pd.DataFrame(
            {
                "Customer ID": [12347, 12348, 12349],
                "Recency": [35, 71, 14],
                "Frequency": [7, 5, 4],
                "Monetary": [5408.50, 2019.40, 4428.69],
                "City": ["Rabat", "Kenitra", "Oujda"],
            }
        ),
        use_container_width=True,
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)
    else:
        raw_df = pd.read_csv(uploaded_file)

    prepared_df = prepare_features(raw_df)
    X = prepared_df[features]
    X_scaled = scaler.transform(X)

    prepared_df["Proba_Churn"] = model.predict_proba(X_scaled)[:, 1]
    prepared_df["Churn_Predit"] = model.predict(X_scaled)
    prepared_df["Risque"] = prepared_df["Proba_Churn"].apply(assign_risk)

except Exception as exc:
    st.error(str(exc))
    st.stop()

total_clients = len(prepared_df)
clients_churn = int(prepared_df["Churn_Predit"].sum())
taux_churn = clients_churn / total_clients if total_clients else 0
risque_eleve = int((prepared_df["Risque"] == "Élevé").sum())
proba_moyenne = prepared_df["Proba_Churn"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Clients analysés", f"{total_clients:,}".replace(",", " "))
kpi2.metric("Clients prédits churn", f"{clients_churn:,}".replace(",", " "))
kpi3.metric("Taux churn prédit", f"{taux_churn:.1%}")
kpi4.metric("Risque élevé", f"{risque_eleve:,}".replace(",", " "))

st.divider()

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    risk_filter = st.multiselect(
        "Risque",
        options=["Faible", "Moyen", "Élevé"],
        default=["Faible", "Moyen", "Élevé"],
    )

with filter_col2:
    city_options = (
        sorted(prepared_df["City"].dropna().unique().tolist())
        if "City" in prepared_df.columns
        else []
    )
    city_filter = st.multiselect("Ville", options=city_options, default=city_options)

with filter_col3:
    st.metric("Probabilité churn moyenne", f"{proba_moyenne:.1%}")

filtered_df = prepared_df[prepared_df["Risque"].isin(risk_filter)].copy()
if city_options:
    filtered_df = filtered_df[filtered_df["City"].isin(city_filter)]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Répartition par risque")
    risk_counts = (
        filtered_df["Risque"]
        .value_counts()
        .reindex(["Faible", "Moyen", "Élevé"])
        .fillna(0)
    )
    st.bar_chart(risk_counts)

with chart_col2:
    st.subheader("Probabilité de churn par client")
    display_index = (
        filtered_df["Customer ID"].astype(str)
        if "Customer ID" in filtered_df.columns
        else filtered_df.index.astype(str)
    )
    proba_chart = pd.DataFrame(
        {"Proba churn": filtered_df["Proba_Churn"].values},
        index=display_index,
    )
    st.line_chart(proba_chart)

st.subheader("Résultats de prédiction")

preferred_columns = [
    "Customer ID",
    "City",
    "Recency",
    "Frequency",
    "Monetary",
    "Proba_Churn",
    "Churn_Predit",
    "Risque",
]
visible_columns = [col for col in preferred_columns if col in filtered_df.columns]

st.dataframe(
    filtered_df[visible_columns].style.format(
        {
            "Monetary": "{:,.2f}",
            "Proba_Churn": "{:.1%}",
        }
    ),
    use_container_width=True,
)

csv_result = filtered_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Télécharger les résultats",
    data=csv_result,
    file_name="predictions_churn_marjane.csv",
    mime="text/csv",
)
