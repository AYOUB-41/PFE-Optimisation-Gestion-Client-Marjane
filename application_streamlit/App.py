import base64
import json
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

alt.renderers.set_embed_options(actions=False)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "modeles"
ASSETS_DIR = APP_DIR / "assets"

# Seuils metier utilises pour transformer la probabilite du modele
# en niveau de risque lisible par un manager.
RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.60,
}

# Couleurs conservees dans toute l'application pour garder une lecture stable.
RISK_COLORS = {
    "Faible": "#00A86B",
    "Moyen": "#F5A623",
    "Élevé": "#E74C3C",
}

st.set_page_config(
    page_title="Dashboard Churn - Marjane",
    page_icon="🛒",
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

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        color: var(--text-main) !important;
        border-right: 2px solid var(--marjane-green) !important;
        box-shadow: 4px 0 16px rgba(0, 158, 115, 0.08) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #FFFFFF !important;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--marjane-blue) !important;
        border-left: 4px solid var(--marjane-green);
        padding-left: 0.6rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--card-border) !important;
    }

    section[data-testid="stSidebar"] code {
        color: #0F172A !important;
        background: #F3F7F5 !important;
        border: 1px solid #CFE3DA !important;
        border-left: 4px solid var(--marjane-green) !important;
        border-radius: 8px !important;
        white-space: normal !important;
        font-size: 0.78rem !important;
        line-height: 1.45 !important;
        padding: 0.75rem !important;
    }

    /* ── Sidebar toggle button ── */
    [data-testid="collapsedControl"] {
        background: #FFFFFF !important;
        border: 2px solid var(--marjane-green) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0, 158, 115, 0.25) !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="collapsedControl"]:hover {
        background: #EAF7F1 !important;
        border-color: #007F5D !important;
    }

    [data-testid="collapsedControl"] svg {
        color: var(--marjane-green) !important;
        stroke: var(--marjane-green) !important;
        fill: none !important;
        width: 18px !important;
        height: 18px !important;
    }

    button[kind="header"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"],
    button[aria-label="Ouvrir le panneau latéral"],
    button[aria-label="Fermer le panneau latéral"] {
        background: #FFFFFF !important;
        border: 2px solid var(--marjane-green) !important;
        border-radius: 10px !important;
        color: var(--marjane-green) !important;
        box-shadow: 0 4px 16px rgba(0, 158, 115, 0.2) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    button[kind="header"] svg,
    button[aria-label="Open sidebar"] svg,
    button[aria-label="Close sidebar"] svg {
        color: var(--marjane-green) !important;
        stroke: var(--marjane-green) !important;
        fill: none !important;
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
        text-align: center;
    }

    .hero h1 {
        color: var(--marjane-blue);
        font-size: 2.25rem;
        margin: 0 0 0.35rem 0;
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

    /* ── Import zone ── */
    .import-zone {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marjane-green);
        border-radius: 10px;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    }

    .import-zone-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1rem;
    }

    .import-icon {
        background: #EAF7F1;
        border-radius: 50%;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }

    .import-title {
        font-weight: 800;
        color: var(--marjane-blue);
        font-size: 1.05rem;
        margin: 0;
    }

    .import-subtitle {
        color: #52616B;
        font-size: 0.82rem;
        margin: 0.15rem 0 0 0;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marjane-green);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        transition: box-shadow 0.2s;
    }

    .kpi-card:hover {
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    }

    .kpi-label {
        color: #52616B;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-value {
        color: var(--text-main);
        font-size: 2rem;
        line-height: 1;
        font-weight: 800;
    }

    /* ── Status box ── */
    .status-box {
        background: #EAF7F1;
        border: 1px solid #B8E2D0;
        border-left: 5px solid var(--marjane-green);
        border-radius: 8px;
        color: #075E46;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1.2rem 0;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Schema cards sidebar ── */
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

    /* ── Section titles ── */
    h2 {
        color: var(--marjane-blue) !important;
        border-left: 4px solid var(--marjane-green);
        padding-left: 0.75rem;
        margin-top: 1.5rem !important;
    }

    h3 {
        color: var(--marjane-blue) !important;
    }

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {
        background: var(--marjane-green);
        border: 1px solid var(--marjane-green);
        color: #FFFFFF;
        border-radius: 6px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #007F5D;
        border-color: #007F5D;
        color: #FFFFFF;
    }

    /* ── Multiselect tags ── */
    div[data-baseweb="tag"] {
        background: #EAF7F1 !important;
        border: 1px solid #009E73 !important;
        border-radius: 999px !important;
    }

    div[data-baseweb="tag"] span {
        color: #006B4F !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #CFE3DA !important;
        color: #1F2933 !important;
    }

    div[data-testid="stMultiSelect"] label,
    div[data-testid="stMultiSelect"] label p {
        color: #1F2933 !important;
        background: transparent !important;
        font-weight: 800 !important;
    }

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label p {
        color: #1F2933 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 2px dashed var(--marjane-green);
        border-radius: 8px;
    }

    div[data-testid="stFileUploader"] section * {
        color: #1F2933 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: var(--marjane-blue) !important;
        border: 1px solid var(--marjane-blue) !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        padding: 0.4rem 1.2rem !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #004A8A !important;
        border-color: #004A8A !important;
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

    div[data-testid="stFileUploader"] button[kind="icon"],
    div[data-testid="stFileUploader"] button[aria-label="Help"],
    div[data-testid="stFileUploader"] [data-testid="stTooltipIcon"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"],
    div[data-testid="stFileUploader"] button[kind="headerNoPadding"],
    div[data-testid="stFileUploader"] button[kind="minimal"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #52616B !important;
        min-width: 28px !important;
        width: 28px !important;
        height: 28px !important;
        padding: 0 !important;
    }

    div[data-testid="stFileUploader"] svg {
        color: #52616B !important;
        fill: none !important;
        stroke: #52616B !important;
    }

    /* ── Charts ── */
    div[data-testid="stVegaLiteChart"] {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marjane-green);
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        overflow: hidden;
    }

    div[data-testid="stVegaLiteChart"] canvas,
    div[data-testid="stVegaLiteChart"] svg {
        background: #FFFFFF !important;
        max-width: 100% !important;
    }

    /* Toolbar graphiques visible et stylisée */
[data-testid="stElementToolbar"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE7E2 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0,158,115,0.12) !important;
    padding: 2px !important;
}

[data-testid="stElementToolbar"] button {
    background: #FFFFFF !important;
    border: 1px solid #CFE3DA !important;
    color: #005AA7 !important;
    border-radius: 6px !important;
    margin: 1px !important;
}

[data-testid="stElementToolbar"] button:hover {
    background: #EAF7F1 !important;
    border-color: #009E73 !important;
}

[data-testid="stElementToolbar"] svg {
    color: #005AA7 !important;
    stroke: #005AA7 !important;
}

    /* ── DataFrames ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--card-border);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    /* ── Notifications ── */
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

    /* ── Toolbar buttons ── */
    [data-testid="StyledFullScreenButton"] button,
    [data-testid="stElementToolbar"] button {
        background: #FFFFFF !important;
        border: 1px solid #CFE3DA !important;
        color: #1F2933 !important;
        border-radius: 6px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }

    /* ── Top risk table ── */
    .top-risk-header {
        background: #FFF5F5;
        border: 1px solid #FECACA;
        border-left: 5px solid #E74C3C;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        color: #7F1D1D;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* ── Recommandations ── */
    .reco-section {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marjane-blue);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    }
    .reco-section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--marjane-blue);
        margin-bottom: 1rem;
    }
    .reco-card {
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        border: 1px solid;
    }
    .reco-card-eleve  { background:#FFF5F5; border-color:#FECACA; border-left:5px solid #E74C3C; }
    .reco-card-moyen  { background:#FFFBEB; border-color:#FDE68A; border-left:5px solid #F5A623; }
    .reco-card-faible { background:#F0FDF4; border-color:#BBF7D0; border-left:5px solid #00A86B; }
    .reco-card-title  { font-weight:800; font-size:0.95rem; margin-bottom:0.4rem; }
    .reco-card-eleve  .reco-card-title { color:#7F1D1D; }
    .reco-card-moyen  .reco-card-title { color:#78350F; }
    .reco-card-faible .reco-card-title { color:#14532D; }
    .reco-card-body   { font-size:0.88rem; color:#374151; line-height:1.6; }
    .reco-card-actions{ margin-top:0.6rem; display:flex; flex-wrap:wrap; gap:0.4rem; }
    .reco-tag         { display:inline-block; border-radius:999px; padding:0.2rem 0.65rem; font-size:0.78rem; font-weight:700; }
    .tag-red    { background:#FEE2E2; color:#991B1B; }
    .tag-orange { background:#FEF3C7; color:#92400E; }
    .tag-green  { background:#D1FAE5; color:#065F46; }

    /* ── Divider ── */
    hr {
        border-color: var(--card-border) !important;
        margin: 1.5rem 0 !important;
    }

    @media (max-width: 900px) {
        .brand-row {
            grid-template-columns: 1fr;
        }
        .hero h1 {
            font-size: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """Charge une seule fois le modele, le scaler et les metadonnees."""
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    with open(MODEL_DIR / "model_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)
    return model, scaler, features, model_info

def normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et renomme automatiquement les colonnes selon leur signification."""
    mapping = {
        "Recency": [
            "recency", "récence", "recence", "jours_inactivite",
            "jours_depuis_achat", "derniere_visite", "last_purchase",
            "days_since_last_purchase", "inactivite", "r_score",
        ],
        "Frequency": [
            "frequency", "fréquence", "frequence", "nb_achats",
            "nombre_achats", "nb_visites", "nombre_visites",
            "nb_commandes", "purchases", "orders", "f_score",
        ],
        "Monetary": [
            "monetary", "montant", "ca_total", "chiffre_affaires",
            "total_achats", "total_depense", "revenue", "amount",
            "valeur_client", "m_score",
        ],
        "Customer ID": [
            "customer_id", "customerid", "id_client", "client_id",
            "id", "customer", "client", "num_client",
        ],
        "City": [
            "city", "ville", "magasin", "store", "region",
            "localisation", "location",
        ],
    }
    df = df.copy()
    colonnes_normalisees = {}
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_")
        for nom_standard, synonymes in mapping.items():
            if col_lower in synonymes or col_lower == nom_standard.lower():
                colonnes_normalisees[col] = nom_standard
                break
    df = df.rename(columns=colonnes_normalisees)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables attendues par le modele a partir d'une table RFM."""
    data = df.copy()
    required = ["Recency", "Frequency", "Monetary"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing))
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[required].isna().any().any():
        raise ValueError("Recency, Frequency et Monetary doivent être numériques.")

    # Variables derivees utilisees pendant l'entrainement du modele churn.
    data["Frequency"] = data["Frequency"].replace(0, np.nan)
    data["Montant_par_visite"] = data["Monetary"] / data["Frequency"]
    data["Recency_x_Frequency"] = data["Recency"] * data["Frequency"]
    data["Log_Montant"] = np.log1p(data["Monetary"])
    data["Log_Frequency"] = np.log1p(data["Frequency"])
    data = data.replace([np.inf, -np.inf], np.nan)
    if data.isna().any().any():
        raise ValueError("Certaines lignes contiennent des valeurs invalides (ex: Frequency = 0).")
    return data


def build_rfm_from_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme un fichier de transactions brutes en table client RFM."""
    data = df.copy()
    required = ["Customer ID", "InvoiceDate", "Invoice", "Quantity", "Price"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes pour le mode brut : " + ", ".join(missing))
    data = data.dropna(subset=["Customer ID", "InvoiceDate"])
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    data = data[(data["Quantity"] > 0) & (data["Price"] > 0)]
    if data.empty:
        raise ValueError("Aucune transaction valide après nettoyage.")

    # Montant de chaque ligne de ticket, puis date de reference pour la recence.
    data["TotalAmount"] = data["Quantity"] * data["Price"]
    analysis_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Agregation au niveau client :
    # Recency = jours depuis le dernier achat
    # Frequency = nombre de factures
    # Monetary = montant total depense
    aggs = {
        "InvoiceDate": lambda x: (analysis_date - x.max()).days,
        "Invoice": "nunique",
        "TotalAmount": "sum",
    }
    if "City" in data.columns:
        aggs["City"] = "first"
    rfm = data.groupby("Customer ID").agg(aggs).reset_index()
    cols = ["Customer ID", "Recency", "Frequency", "Monetary"]
    if "City" in data.columns:
        cols.append("City")
    rfm.columns = cols
    rfm = rfm[rfm["Monetary"] > 0]
    return rfm


def detect_and_prepare_input(df: pd.DataFrame):
    """Detecte automatiquement si le fichier est deja RFM ou transactionnel."""
    # Normaliser les colonnes avant toute vérification
    df = normaliser_colonnes(df)

    rfm_cols = {"Recency", "Frequency", "Monetary"}
    tx_cols = {"Customer ID", "InvoiceDate", "Invoice", "Quantity", "Price"}

    if rfm_cols.issubset(df.columns):
        return prepare_features(df), "Fichier RFM préparé"
    if tx_cols.issubset(df.columns):
        rfm = build_rfm_from_transactions(df)
        return prepare_features(rfm), "Fichier transactionnel brut"

    cols_trouvees = list(df.columns)
    raise ValueError(
        f"Format non reconnu. Colonnes trouvées : {cols_trouvees}\n\n"
        "Le fichier doit contenir soit :\n"
        "• Recency / Frequency / Monetary (ou équivalents)\n"
        "• Customer ID / InvoiceDate / Invoice / Quantity / Price"
    )


def assign_risk(p: float) -> str:
    """Convertit une probabilite de churn en niveau de risque."""
    if p < RISK_THRESHOLDS["low"]:
        return "Faible"
    if p < RISK_THRESHOLDS["medium"]:
        return "Moyen"
    return "Élevé"
def get_recommendation(row) -> dict:
    """Génère une recommandation business personnalisée selon le profil client."""
    recency  = row.get("Recency", 0)
    frequency = row.get("Frequency", 0)
    monetary  = row.get("Monetary", 0)
    risque    = row.get("Risque", "Faible")

    if risque == "Élevé":
        if recency > 365:
            return {
                "titre": "🔴 Campagne de reconquête urgente",
                "description": f"Client inactif depuis {int(recency)} jours. Risque très élevé de perte définitive.",
                "actions": ["Bon de réduction 25%", "Appel téléphonique direct", "Offre personnalisée"],
                "priorite": "PRIORITÉ 1",
            }
        else:
            return {
                "titre": "🟠 Campagne de réactivation",
                "description": f"Client à risque élevé — inactif depuis {int(recency)} jours.",
                "actions": ["Email de réactivation", "Offre groupée produits fréquents", "Programme fidélité VIP"],
                "priorite": "PRIORITÉ 1",
            }
    elif risque == "Moyen":
        if frequency >= 5:
            return {
                "titre": "🟡 Renforcer la fidélisation",
                "description": f"Client actif ({int(frequency)} achats) mais montre des signes de désengagement.",
                "actions": ["Newsletter personnalisée", "Points fidélité bonus", "Invitation événement Marjane"],
                "priorite": "PRIORITÉ 2",
            }
        else:
            return {
                "titre": "🟡 Stimuler l'engagement",
                "description": f"Client peu fréquent ({int(frequency)} achats) avec risque modéré.",
                "actions": ["Offre découverte nouveaux rayons", "Coupon revisité", "Push notification"],
                "priorite": "PRIORITÉ 2",
            }
    else:
        return {
            "titre": "🟢 Maintenir la satisfaction",
            "description": f"Client actif et fidèle — CA de {monetary:,.0f} MAD.",
            "actions": ["Programme fidélité premium", "Accès ventes privées", "Enquête satisfaction"],
            "priorite": "PRIORITÉ 3",
        }

def logo_slot(file_name: str, label: str) -> str:
    """Affiche un logo si le fichier existe, sinon un emplacement reserve."""
    path = ASSETS_DIR / file_name
    if path.exists():
        suffix = path.suffix.lower().replace(".", "")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return (
            '<div class="logo-slot">'
            f'<img src="data:image/{mime};base64,{encoded}" alt="{label}">'
            "</div>"
        )
    return (
        '<div class="logo-slot"><div>'
        f"<strong>{label}</strong>"
        "<span>emplacement logo</span>"
        "</div></div>"
    )


def kpi_card(label: str, value: str, color: str = "#009E73"):
    """Composant reutilisable pour afficher les KPI sous forme de carte."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_height(n_rows: int) -> int:
    """Ajuste la hauteur d'un graphique selon le volume de donnees."""
    if n_rows <= 10:
        return 250
    elif n_rows <= 50:
        return 320
    elif n_rows <= 200:
        return 420
    return 500


def altair_cfg():
    """Configuration commune pour garder des graphes lisibles en theme clair."""
    return dict(
        background="#FFFFFF",
        view={"fill": "#FFFFFF", "stroke": None},
        axis={
            "labelColor": "#1F2933",
            "titleColor": "#52616B",
            "gridColor": "#F1F5F9",
            "domainColor": "#CBD5E1",
            "labelFontSize": 11,
            "titleFontSize": 12,
        },
        legend={
            "labelColor": "#1F2933",
            "titleColor": "#1F2933",
            "labelFontSize": 11,
        },
    )


# ── Load model ───────────────────────────────────────────────────────────────

model, scaler, features, model_info = load_artifacts()

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="brand-row">
        {logo_slot("marjane_logo.png", "MARJANE")}
        <div class="hero">
            <div class="hero-badge">Pilotage churn client</div>
            <h1>Dashboard de Prédiction du Churn Client</h1>
            <p>Import d'un fichier magasin · Calcul RFM automatique · Prédiction des clients à risque</p>
        </div>
        {logo_slot("marjane_group_logo.png", "MARJANE GROUP")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #EAF7F1 0%, #FFFFFF 100%);
            border: 1px solid #B8E2D0;
            border-top: 4px solid #009E73;
            border-radius: 10px;
            padding: 1rem 1rem 0.75rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,158,115,0.08);
        ">
            <div style="font-size:0.75rem;font-weight:800;color:#006B4F;
                        text-transform:uppercase;letter-spacing:0.06em;
                        margin-bottom:0.5rem;">
                ⚙️ Paramètres du modèle
            </div>
            <div style="display:flex;flex-direction:column;gap:0.5rem;">
                <div style="background:#FFFFFF;border:1px solid #DDE7E2;
                            border-radius:6px;padding:0.5rem 0.75rem;">
                    <span style="font-size:0.75rem;color:#52616B;display:block;">Modèle utilisé</span>
                    <span style="font-size:0.95rem;font-weight:800;color:#005AA7;">
                        {modele}
                    </span>
                </div>
                <div style="background:#FFFFFF;border:1px solid #DDE7E2;
                            border-radius:6px;padding:0.5rem 0.75rem;">
                    <span style="font-size:0.75rem;color:#52616B;display:block;">Seuil churn</span>
                    <span style="font-size:0.95rem;font-weight:800;color:#E74C3C;">
                        {seuil} jours d'inactivité
                    </span>
                </div>
            </div>
        </div>
        """.format(
            modele=model_info.get("meilleur_modele", "XGBoost"),
            seuil=model_info.get("seuil_churn_jours", 180),
        ),
        unsafe_allow_html=True,
    )
    st.markdown("**Colonnes obligatoires**")
    st.markdown(
        """
        <div class="schema-card">
            <strong>Mode RFM</strong>
            <span>Recency, Frequency, Monetary</span>
        </div>
        <div class="schema-card">
            <strong>Mode brut</strong>
            <span>Customer ID, InvoiceDate,<br>Invoice, Quantity, Price</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Noms alternatifs acceptés**")
    st.markdown(
        """
        <div class="schema-card">
            <strong>Recency</strong>
            <span>jours_depuis_achat, jours_inactivite,
            last_purchase, recence</span>
        </div>
        <div class="schema-card">
            <strong>Frequency</strong>
            <span>nb_achats, nombre_visites,
            nb_commandes, frequence</span>
        </div>
        <div class="schema-card">
            <strong>Monetary</strong>
            <span>montant, ca_total,
            chiffre_affaires, total_depense</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Import zone ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="import-zone">
        <div class="import-zone-header">
            <div class="import-icon">📂</div>
            <div>
                <p class="import-title">Importer un fichier clients</p>
                <p class="import-subtitle">
                    Fichier RFM préparé ou transactionnel brut &nbsp;·&nbsp;
                    Formats acceptés : CSV, Excel (.xlsx) &nbsp;·&nbsp; Max 200 MB
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "",
    type=["csv", "xlsx"],
    help="Le fichier peut être un fichier RFM préparé ou un fichier transactionnel brut.",
    label_visibility="collapsed",
)

# ── Empty state ───────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.info("  Importe un fichier CSV ou Excel pour lancer la prédiction.")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown("**Option 1 — Fichier RFM préparé**")
        st.dataframe(
            pd.DataFrame({
                "Customer ID": [12347, 12348, 12349],
                "Recency": [35, 71, 14],
                "Frequency": [7, 5, 4],
                "Monetary": [5408.50, 2019.40, 4428.69],
                "City": ["Rabat", "Kenitra", "Oujda"],
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_ex2:
        st.markdown("**Option 2 — Fichier transactionnel brut**")
        st.dataframe(
            pd.DataFrame({
                "Invoice": ["INV001", "INV002", "INV003"],
                "Customer ID": [10001, 10001, 10002],
                "InvoiceDate": ["2026-01-10", "2026-02-15", "2025-08-20"],
                "Quantity": [2, 1, 5],
                "Price": [120.0, 350.0, 80.0],
                "City": ["Casablanca", "Casablanca", "Rabat"],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.stop()

# ── Process file ─────────────────────────────────────────────────────────────

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
    st.error(f"❌ Erreur lors du traitement : {exc}")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────

total_clients = len(prepared_df)
clients_churn = int(prepared_df["Churn_Predit"].sum())
taux_churn = clients_churn / total_clients if total_clients else 0
risque_eleve = int((prepared_df["Risque"] == "Élevé").sum())
proba_moyenne = prepared_df["Proba_Churn"].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Clients analysés", f"{total_clients:,}".replace(",", "\u202f"), "#005AA7")
with k2:
    kpi_card("Clients prédits churn", f"{clients_churn:,}".replace(",", "\u202f"), "#E74C3C")
with k3:
    kpi_card("Taux churn prédit", f"{taux_churn:.1%}", "#F5A623")
with k4:
    kpi_card("Risque élevé", f"{risque_eleve:,}".replace(",", "\u202f"), "#E74C3C")

st.divider()

# ── Status + filters ─────────────────────────────────────────────────────────

st.markdown(
    f'<div class="status-box">✅ &nbsp;Type de fichier détecté : {input_mode}</div>',
    unsafe_allow_html=True,
)

fc1, fc2, fc3 = st.columns(3)

with fc1:
    risk_filter = st.multiselect(
        "Filtrer par risque",
        options=["Faible", "Moyen", "Élevé"],
        default=["Faible", "Moyen", "Élevé"],
    )

with fc2:
    city_options = (
        sorted(prepared_df["City"].dropna().unique().tolist())
        if "City" in prepared_df.columns else []
    )
    city_filter = st.multiselect(
        "Filtrer par ville",
        options=city_options,
        default=city_options,
    )

with fc3:
    kpi_card("Probabilité churn moyenne", f"{proba_moyenne:.1%}", "#005AA7")

filtered_df = prepared_df[prepared_df["Risque"].isin(risk_filter)].copy()
if city_options:
    filtered_df = filtered_df[filtered_df["City"].isin(city_filter)]

# ── Charts ────────────────────────────────────────────────────────────────────

ch1, ch2 = st.columns(2)

with ch1:
    st.subheader("Répartition par risque")

    risk_counts = (
        filtered_df["Risque"]
        .value_counts()
        .reindex(["Faible", "Moyen", "Élevé"])
        .fillna(0)
        .reset_index()
    )
    risk_counts.columns = ["Risque", "Nombre"]
    total_r = risk_counts["Nombre"].sum()
    risk_counts["Pct"] = (
        (risk_counts["Nombre"] / total_r * 100).round(1)
        if total_r else 0
    )
    risk_counts["Label"] = risk_counts.apply(
        lambda r: f"{int(r['Nombre'])} ({r['Pct']}%)", axis=1
    )

    # Hauteur fixe raisonnable — pas liée aux données car c'est toujours 3 barres
    H = 300

    bars = (
        alt.Chart(risk_counts)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Risque:N",
                    sort=["Faible", "Moyen", "Élevé"],
                    title=None,
                    axis=alt.Axis(
                        labelFontSize=13,
                        labelFontWeight="bold",
                        labelAngle=0,
                    )),
            y=alt.Y("Nombre:Q",
                    title="Nombre de clients",
                    scale=alt.Scale(
                        domain=[0, max(risk_counts["Nombre"].max() * 1.25, 1)]
                    )),
            color=alt.Color(
                "Risque:N",
                scale=alt.Scale(
                    domain=["Faible", "Moyen", "Élevé"],
                    range=[
                        RISK_COLORS["Faible"],
                        RISK_COLORS["Moyen"],
                        RISK_COLORS["Élevé"],
                    ],
                ),
                legend=None,
            ),
            tooltip=[
                "Risque:N",
                "Nombre:Q",
                alt.Tooltip("Pct:Q", format=".1f", title="% total"),
            ],
        )
        .properties(height=H, width="container")
    )

    labels = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-6,
        fontSize=12,
        fontWeight="bold",
        color="#1F2933",
    ).encode(text="Label:N")

    cfg = altair_cfg()
    st.altair_chart(
        (bars + labels)
        .configure(**{k: v for k, v in cfg.items()
                      if k not in ("view", "axis", "legend")})
        .configure_view(**cfg["view"])
        .configure_axis(**cfg["axis"])
        .configure_legend(**cfg["legend"]),
        use_container_width=True,
    )

with ch2:
    st.subheader("Distribution des probabilités de churn")

    n = len(filtered_df)
    # Hauteur flexible selon volume de données
    H2 = 250 if n <= 20 else 300 if n <= 100 else 350
    # Bins adaptatifs
    n_bins = max(5, min(20, n // 10)) if n > 0 else 10

    hist_data = pd.DataFrame({
        "Probabilité": filtered_df["Proba_Churn"].values,
        "Risque": filtered_df["Risque"].values,
    })

    hist = (
        alt.Chart(hist_data)
        .mark_bar(
            opacity=0.88,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .encode(
            x=alt.X("Probabilité:Q",
                    bin=alt.Bin(maxbins=n_bins),
                    title="Probabilité de churn",
                    axis=alt.Axis(format="%", labelAngle=0)),
            y=alt.Y("count()",
                    title="Nombre de clients",
                    scale=alt.Scale(zero=True)),
            color=alt.Color(
                "Risque:N",
                scale=alt.Scale(
                    domain=["Faible", "Moyen", "Élevé"],
                    range=[
                        RISK_COLORS["Faible"],
                        RISK_COLORS["Moyen"],
                        RISK_COLORS["Élevé"],
                    ],
                ),
                legend=alt.Legend(
                    title="Risque",
                    orient="top-right",
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip("Probabilité:Q", format=".1%", title="Prob. churn"),
                "count()",
                "Risque:N",
            ],
        )
        .properties(height=H2, width="container")
    )

    cfg = altair_cfg()
    st.altair_chart(
        hist
        .configure(**{k: v for k, v in cfg.items()
                      if k not in ("view", "axis", "legend")})
        .configure_view(**cfg["view"])
        .configure_axis(**cfg["axis"])
        .configure_legend(**cfg["legend"]),
        use_container_width=True,
    )

# ── Top 10 clients à risque élevé ────────────────────────────────────────────

st.markdown(
    '<div class="top-risk-header">🔴 &nbsp;Top 10 clients à risque élevé — action prioritaire</div>',
    unsafe_allow_html=True,
)

top_risque = filtered_df[filtered_df["Risque"] == "Élevé"].copy()

if len(top_risque) > 0:
    top_risque = top_risque.sort_values("Proba_Churn", ascending=False).head(10)
    cols_top = [c for c in [
        "Customer ID", "City", "Recency", "Frequency", "Monetary", "Proba_Churn"
    ] if c in top_risque.columns]

    st.dataframe(
        top_risque[cols_top].style
        .format({"Monetary": "{:,.2f}", "Proba_Churn": "{:.1%}"})
        .background_gradient(subset=["Proba_Churn"], cmap="Reds", vmin=0.6, vmax=1.0),
        use_container_width=True,
        height=320,
        hide_index=True,
    )
else:
    st.info("Aucun client à risque élevé dans la sélection actuelle.")

st.divider()
# ── Plan d'action et recommandations ─────────────────────────────

st.subheader(" Plan d'action recommandé")

n_eleve  = int((filtered_df["Risque"] == "Élevé").sum())
n_moyen  = int((filtered_df["Risque"] == "Moyen").sum())
n_faible = int((filtered_df["Risque"] == "Faible").sum())

# Construire le HTML séparément pour éviter les conflits f-string
html_reco = (
    '<div class="reco-section">'
    '<div class="reco-section-title"> Synthèse des recommandations pour les managers Marjane</div>'

    f'<div class="reco-card reco-card-eleve">'
    f'<div class="reco-card-title">🔴 Risque Élevé — {n_eleve} clients — PRIORITÉ 1</div>'
    '<div class="reco-card-body">'
    'Ces clients n\'ont pas acheté depuis plus de 180 jours avec une probabilité de churn &gt; 60%.'
    ' Une action immédiate est nécessaire.<br><br>'
    '<strong>Actions recommandées :</strong><br>'
    '• Bon de réduction personnalisé de 20 à 25% sur leur catégorie préférée<br>'
    '• Campagne SMS/Email de réactivation dans les 48h<br>'
    '• Appel téléphonique pour les clients à forte valeur<br>'
    '• Offre groupée basée sur les règles d\'association identifiées'
    '</div>'
    '<div class="reco-card-actions">'
    '<span class="reco-tag tag-red">⚡ Action sous 48h</span>'
    '<span class="reco-tag tag-red">📱 SMS + Email</span>'
    '<span class="reco-tag tag-red">🎯 Offre personnalisée</span>'
    '</div></div>'

    f'<div class="reco-card reco-card-moyen">'
    f'<div class="reco-card-title">🟡 Risque Moyen — {n_moyen} clients — PRIORITÉ 2</div>'
    '<div class="reco-card-body">'
    'Ces clients montrent des signes de désengagement avec une probabilité entre 30% et 60%.<br><br>'
    '<strong>Actions recommandées :</strong><br>'
    '• Newsletter mensuelle avec offres ciblées<br>'
    '• Points fidélité bonus sur le prochain achat<br>'
    '• Invitation aux événements exclusifs Marjane<br>'
    '• Cross-selling basé sur les produits complémentaires'
    '</div>'
    '<div class="reco-card-actions">'
    '<span class="reco-tag tag-orange">📧 Newsletter ciblée</span>'
    '<span class="reco-tag tag-orange">⭐ Points fidélité</span>'
    '<span class="reco-tag tag-orange">🛍️ Cross-selling</span>'
    '</div></div>'

    f'<div class="reco-card reco-card-faible">'
    f'<div class="reco-card-title">🟢 Risque Faible — {n_faible} clients — PRIORITÉ 3</div>'
    '<div class="reco-card-body">'
    'Ces clients sont actifs et engagés avec une probabilité de churn &lt; 30%.<br><br>'
    '<strong>Actions recommandées :</strong><br>'
    '• Programme de fidélité premium Marjane<br>'
    '• Accès anticipé aux nouvelles collections<br>'
    '• Enquête de satisfaction<br>'
    '• Programme de parrainage'
    '</div>'
    '<div class="reco-card-actions">'
    '<span class="reco-tag tag-green">💎 Programme premium</span>'
    '<span class="reco-tag tag-green">🎁 Ventes privées</span>'
    '<span class="reco-tag tag-green">📊 Satisfaction</span>'
    '</div></div>'

    '</div>'
)

st.markdown(html_reco, unsafe_allow_html=True)

# ── Résultats complets ────────────────────────────────────────────────────────

st.subheader("Résultats de prédiction — tableau complet")

# Colonnes affichées dans le dashboard
preferred = [
    "Customer ID", "City", "Recency", "Frequency",
    "Monetary", "Proba_Churn", "Churn_Predit", "Risque",
]
visible = [c for c in preferred if c in filtered_df.columns]

# Ajouter recommandation dans le tableau dashboard
filtered_df["Recommandation"] = filtered_df.apply(
    lambda row: get_recommendation(row)["titre"], axis=1
)
filtered_df["Action_Priorite"] = filtered_df["Risque"].map({
    "Élevé": "PRIORITÉ 1 — Action sous 48h",
    "Moyen":  "PRIORITÉ 2 — Action sous 2 semaines",
    "Faible": "PRIORITÉ 3 — Suivi mensuel",
})

# Colonnes affichées avec recommandation
visible_avec_reco = visible + ["Recommandation", "Action_Priorite"]

st.dataframe(
    filtered_df[visible_avec_reco].style.format({
        "Monetary": "{:,.2f}",
        "Proba_Churn": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)
# Copie propre pour l'export
export_df = filtered_df[visible_avec_reco].copy()

# Convertir Proba_Churn en pourcentage lisible dans Excel
export_df["Proba_Churn"] = export_df["Proba_Churn"].apply(
    lambda x: f"{x:.1%}"
)

csv_result = export_df.to_csv(index=False).encode("utf-8-sig")

dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    st.download_button(
        " Télécharger les résultats",
        data=csv_result,
        file_name="predictions_churn_marjane.csv",
        mime="text/csv",
    )
