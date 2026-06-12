import base64
import json
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


st.set_page_config(
    page_title="Dashboard Churn - Marjane",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --marjane-green: #009E73;
        --marjane-blue: #005AA7;
        --risk-red: #E74C3C;
        --risk-orange: #F5A623;
        --soft-bg: #F5F8F6;
        --card-border: #DDE7E2;
        --text-main: #1F2933;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--soft-bg);
        color: var(--text-main);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 246, 0.92);
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        color: var(--text-main);
        border-right: 1px solid var(--card-border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    section[data-testid="stSidebar"] code {
        color: #0F172A;
        background: #F3F7F5;
        border: 1px solid #CFE3DA;
        border-left: 4px solid var(--marjane-green);
        border-radius: 8px;
        white-space: normal;
        font-size: 0.78rem;
        line-height: 1.45;
        padding: 0.75rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--marjane-blue);
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--card-border);
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }

    .brand-row {
        display: grid;
        grid-template-columns: 210px minmax(420px, 1fr) 210px;
        gap: 1.25rem;
        align-items: center;
        margin-bottom: 1.25rem;
    }

    .logo-slot {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-bottom: 4px solid var(--marjane-green);
        border-radius: 10px;
        min-height: 92px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        padding: 0.75rem;
    }

    .logo-slot strong {
        color: var(--marjane-green);
        font-size: 1.25rem;
        letter-spacing: 0;
    }

    .logo-slot span {
        color: #52616B;
        font-size: 0.72rem;
        display: block;
        text-align: center;
        margin-top: 0.15rem;
    }

    .logo-slot img {
        max-width: 175px;
        max-height: 70px;
        object-fit: contain;
    }

    .hero {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 6px solid var(--marjane-green);
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.09);
        margin-bottom: 0;
        text-align: center;
    }

    .hero h1 {
        color: var(--marjane-blue);
        font-size: 2.25rem;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0;
        line-height: 1.15;
    }

    .hero p {
        color: #52616B;
        margin: 0;
        font-size: 1rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: #EAF7F1;
        color: #006B4F;
        border: 1px solid #B8E2D0;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label p {
        color: #1F2933 !important;
        font-weight: 800 !important;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marjane-green);
        border-radius: 8px;
        padding: 1rem;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .kpi-label {
        color: #52616B;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .kpi-value {
        color: var(--text-main);
        font-size: 2rem;
        line-height: 1;
        font-weight: 800;
    }

    .status-box {
        background: #EAF7F1;
        border: 1px solid #B8E2D0;
        border-left: 5px solid var(--marjane-green);
        border-radius: 8px;
        color: #075E46;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1.2rem 0;
        font-weight: 700;
    }

    .schema-card {
        background: #F7FBF9;
        border: 1px solid #CFE3DA;
        border-left: 4px solid var(--marjane-green);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.65rem 0;
    }

    .schema-card strong {
        color: var(--marjane-blue) !important;
        display: block;
        margin-bottom: 0.25rem;
    }

    .schema-card span {
        color: #1F2933 !important;
        font-family: Consolas, monospace;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    div[data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 1px dashed var(--marjane-green);
        border-radius: 8px;
    }

    div[data-testid="stFileUploader"] section * {
        color: #1F2933 !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #52616B !important;
    }

    div[data-testid="stFileUploader"] button {
        background: var(--marjane-blue) !important;
        border: 1px solid var(--marjane-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--marjane-blue);
        border-radius: 8px;
        color: #1F2933 !important;
    }

    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
        color: #1F2933 !important;
    }

    div[data-testid="stFileUploader"] button[kind="icon"] {
        background: transparent !important;
        border: 0 !important;
        color: #52616B !important;
        min-width: 32px !important;
    }

    div[data-testid="stFileUploader"] button[kind="icon"] svg {
        color: #52616B !important;
        stroke: #52616B !important;
    }

    div[data-testid="stFileUploader"] svg {
        color: #52616B !important;
        fill: none !important;
        stroke: #52616B !important;
    }

    [data-testid="StyledFullScreenButton"] button,
    [data-testid="stElementToolbar"] button {
        background: #FFFFFF !important;
        border: 1px solid #CFE3DA !important;
        color: #1F2933 !important;
        border-radius: 6px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }

    [data-testid="StyledFullScreenButton"] svg,
    [data-testid="stElementToolbar"] svg {
        color: #1F2933 !important;
        stroke: #1F2933 !important;
    }

    [data-testid="stNotification"],
    div[data-baseweb="notification"] {
        background: #EAF7F1 !important;
        border: 1px solid #B8E2D0 !important;
        color: #075E46 !important;
        border-radius: 8px !important;
    }

    [data-testid="stNotification"] *,
    div[data-baseweb="notification"] * {
        color: #075E46 !important;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #CFE3DA !important;
        color: #1F2933 !important;
    }

    div[data-baseweb="tag"] {
        background: #EAF7F1 !important;
        border: 1px solid #B8E2D0 !important;
    }

    div[data-baseweb="tag"] span {
        color: #075E46 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: var(--marjane-green);
        border: 1px solid var(--marjane-green);
        color: #FFFFFF;
        border-radius: 6px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #007F5D;
        border-color: #007F5D;
        color: #FFFFFF;
    }

    h2, h3 {
        color: var(--marjane-blue);
        letter-spacing: 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--card-border);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    @media (max-width: 900px) {
        .brand-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def build_rfm_from_transactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    required_columns = ["Customer ID", "InvoiceDate", "Invoice", "Quantity", "Price"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour le fichier transactionnel : "
            + ", ".join(missing_columns)
        )

    data = data.dropna(subset=["Customer ID", "InvoiceDate"])
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    data = data[(data["Quantity"] > 0) & (data["Price"] > 0)]

    if data.empty:
        raise ValueError("Aucune transaction valide apres nettoyage du fichier.")

    data["TotalAmount"] = data["Quantity"] * data["Price"]
    analysis_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    aggregations = {
        "InvoiceDate": lambda x: (analysis_date - x.max()).days,
        "Invoice": "nunique",
        "TotalAmount": "sum",
    }
    if "City" in data.columns:
        aggregations["City"] = "first"

    rfm = data.groupby("Customer ID").agg(aggregations).reset_index()

    columns = ["Customer ID", "Recency", "Frequency", "Monetary"]
    if "City" in data.columns:
        columns.append("City")
    rfm.columns = columns
    rfm = rfm[rfm["Monetary"] > 0]

    return rfm


def detect_and_prepare_input(df: pd.DataFrame):
    rfm_columns = {"Recency", "Frequency", "Monetary"}
    transaction_columns = {"Customer ID", "InvoiceDate", "Invoice", "Quantity", "Price"}

    if rfm_columns.issubset(df.columns):
        return prepare_features(df), "Fichier RFM prepare"

    if transaction_columns.issubset(df.columns):
        rfm = build_rfm_from_transactions(df)
        return prepare_features(rfm), "Fichier transactionnel brut"

    raise ValueError(
        "Format non reconnu. Le fichier doit contenir soit Recency, Frequency, Monetary, "
        "soit Customer ID, InvoiceDate, Invoice, Quantity, Price."
    )


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


def logo_slot(file_name: str, label: str) -> str:
    logo_path = ASSETS_DIR / file_name
    if logo_path.exists():
        suffix = logo_path.suffix.lower().replace(".", "")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        return (
            '<div class="logo-slot">'
            f'<img src="data:image/{mime};base64,{encoded}" alt="{label}">'
            "</div>"
        )

    return (
        '<div class="logo-slot">'
        "<div>"
        f"<strong>{label}</strong>"
        "<span>emplacement logo</span>"
        "</div>"
        "</div>"
    )


model, scaler, features, model_info = load_artifacts()

st.markdown(
    f"""
    <div class="brand-row">
        {logo_slot("marjane_logo.png", "MARJANE")}
        <div class="hero">
            <div class="hero-badge">Pilotage churn client</div>
            <h1>Dashboard Churn Client - Marjane</h1>
            <p>Import d'un fichier magasin, calcul RFM automatique et prédiction des clients à risque.</p>
        </div>
        {logo_slot("marjane_group_logo.png", "MARJANE GROUP")}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Paramètres")
    st.write(f"Modèle utilisé : **{model_info.get('meilleur_modele', 'Modèle ML')}**")
    st.write(f"Seuil churn : **{model_info.get('seuil_churn_jours', 180)} jours**")
    st.divider()
    st.write("Colonnes obligatoires dans le fichier :")
    st.markdown(
        """
        <div class="schema-card">
            <strong>Mode RFM</strong>
            <span>Recency, Frequency, Monetary</span>
        </div>
        <div class="schema-card">
            <strong>Mode brut</strong>
            <span>Customer ID, InvoiceDate, Invoice, Quantity, Price</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "Importer un fichier clients",
    type=["csv", "xlsx"],
    help="Le fichier peut etre un fichier RFM prepare ou un fichier transactionnel brut.",
)

if uploaded_file is None:
    st.info("Importe un fichier CSV ou Excel pour lancer la prédiction.")
    st.subheader("Exemple de format attendu")
    st.write("Option 1 : fichier client deja prepare avec les metriques RFM.")
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
    st.write("Option 2 : fichier transactionnel brut, l'application calcule RFM automatiquement.")
    st.dataframe(
        pd.DataFrame(
            {
                "Invoice": ["INV001", "INV002", "INV003"],
                "Customer ID": [10001, 10001, 10002],
                "InvoiceDate": ["2026-01-10", "2026-02-15", "2025-08-20"],
                "Quantity": [2, 1, 5],
                "Price": [120.0, 350.0, 80.0],
                "City": ["Casablanca", "Casablanca", "Rabat"],
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

    prepared_df, input_mode = detect_and_prepare_input(raw_df)
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

def kpi_card(label: str, value: str, color: str = "#009E73"):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color: {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    kpi_card("Clients analysés", f"{total_clients:,}".replace(",", " "), "#005AA7")
with kpi2:
    kpi_card("Clients prédits churn", f"{clients_churn:,}".replace(",", " "), "#E74C3C")
with kpi3:
    kpi_card("Taux churn prédit", f"{taux_churn:.1%}", "#F5A623")
with kpi4:
    kpi_card("Risque élevé", f"{risque_eleve:,}".replace(",", " "), "#E74C3C")

st.divider()
st.markdown(
    f'<div class="status-box">Type de fichier détecté : {input_mode}</div>',
    unsafe_allow_html=True,
)

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
    kpi_card("Probabilité churn moyenne", f"{proba_moyenne:.1%}", "#005AA7")

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
        .reset_index()
    )
    risk_counts.columns = ["Risque", "Nombre de clients"]
    risk_chart = (
        alt.Chart(risk_counts)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Risque:N", sort=["Faible", "Moyen", "Élevé"], title="Risque"),
            y=alt.Y("Nombre de clients:Q", title="Nombre de clients"),
            color=alt.Color(
                "Risque:N",
                scale=alt.Scale(
                    domain=["Faible", "Moyen", "Élevé"],
                    range=["#00A86B", "#F5A623", "#E74C3C"],
                ),
                legend=None,
            ),
            tooltip=["Risque:N", "Nombre de clients:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(risk_chart, use_container_width=True)

with chart_col2:
    st.subheader("Probabilité de churn par client")
    display_index = (
        filtered_df["Customer ID"].astype(str)
        if "Customer ID" in filtered_df.columns
        else filtered_df.index.astype(str)
    )
    proba_chart = pd.DataFrame(
        {
            "Client": display_index,
            "Probabilité churn": filtered_df["Proba_Churn"].values,
            "Risque": filtered_df["Risque"].values,
        }
    )
    line_chart = (
        alt.Chart(proba_chart)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Client:N", title="Client"),
            y=alt.Y("Probabilité churn:Q", title="Probabilité churn", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Risque:N",
                scale=alt.Scale(
                    domain=["Faible", "Moyen", "Élevé"],
                    range=["#00A86B", "#F5A623", "#E74C3C"],
                ),
            ),
            tooltip=[
                "Client:N",
                alt.Tooltip("Probabilité churn:Q", format=".1%"),
                "Risque:N",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(line_chart, use_container_width=True)

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
