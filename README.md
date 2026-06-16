# PFE - Optimisation de la Gestion Client chez Marjane

## Description du projet

Ce projet est réalisé dans le cadre de mon stage de Projet de Fin d'Études au sein de Marjane.

L'objectif est d'exploiter des données transactionnelles retail afin d'améliorer la compréhension du comportement client et d'aider à la prise de décision commerciale.

Le projet repose sur trois axes Data Science :

- **Segmentation RFM des clients** : identification des profils clients selon la récence, la fréquence et le montant d'achat.
- **Analyse du panier d'achat** : extraction de règles d'association pour identifier les produits souvent achetés ensemble.
- **Prédiction du churn client** : détection des clients présentant un risque d'inactivité ou de départ.

Les résultats sont exploités à travers :

- un dashboard **Power BI** pour le suivi décisionnel ;
- un dashboard **Streamlit** pour la prédiction interactive du churn.

## Titre du PFE

**Optimisation de la gestion client chez Marjane par l'analyse des données transactionnelles**

## Structure du dépôt

```text
PFE-Optimisation-Gestion-Client-Marjane/
├── application_streamlit/
│   ├── App.py
│   └── assets/
│       ├── marjane_logo.png
│       └── marjane_group_logo.png
├── donnees/
│   ├── online_retail_II.xlsx
│   ├── marjane_data_villes.xlsx
│   ├── rfm_export.csv
│   └── churn_predictions.csv
├── donnees_test/
│   ├── test_clients_churn.csv
│   └── test_transactions_brut.csv
├── modeles/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── features.json
│   └── model_info.json
├── notebooks/
│   ├── 01_Segmentation_RFM_Clients_Marjane.ipynb
│   ├── 02_Analyse_Panier_Association_Marjane.ipynb
│   └── 03_Prediction_Churn_Clients_Marjane.ipynb
├── reporting_powerbi/
│   ├── powerbi_clients.csv
│   └── powerbi_kpis.csv
│   └── Dashboard_Intelligence_Client_Marjane.pbix
├── requirements.txt
└── README.md
```

## Notebooks

### 1. Segmentation RFM des Clients Marjane

Fichier :

`notebooks/01_Segmentation_RFM_Clients_Marjane.ipynb`

Ce notebook prépare les données, calcule les métriques RFM et applique un clustering K-Means afin d'identifier les principaux segments clients :

- VIP / Champions
- Clients réguliers
- Clients à risque / perdus
- Gros acheteurs occasionnels

### 2. Analyse du Panier et Règles d'Association

Fichier :

`notebooks/02_Analyse_Panier_Association_Marjane.ipynb`

Ce notebook applique l'analyse du panier d'achat avec FP-Growth afin d'extraire les associations de produits les plus pertinentes.

Les indicateurs utilisés sont :

- support ;
- confiance ;
- lift.

### 3. Prédiction du Churn Client

Fichier :

`notebooks/03_Prediction_Churn_Clients_Marjane.ipynb`

Ce notebook construit une variable cible de churn à partir d'un seuil d'inactivité de 180 jours, entraîne des modèles supervisés et sauvegarde le meilleur modèle pour l'application Streamlit.

Modèles utilisés :

- Random Forest
- XGBoost

## Dashboards

### Dashboard Power BI

Le dossier `reporting_powerbi/` contient les fichiers de données utilisés pour le reporting Power BI :

- `powerbi_clients.csv`
- `powerbi_kpis.csv`
- `Dashboard_Intelligence_Client_Marjane.pbix`

Le dashboard Power BI présente :

- les KPIs globaux ;
- la répartition des clients par ville ;
- les segments RFM ;
- les niveaux de risque de churn.

### Dashboard Streamlit

L'application Streamlit se trouve dans :

`application_streamlit/App.py`

Elle permet de :

- importer un fichier client préparé avec les métriques RFM ;
- importer un fichier transactionnel brut ;
- calculer automatiquement les métriques RFM si nécessaire ;
- prédire la probabilité de churn ;
- afficher le niveau de risque ;
- télécharger les résultats de prédiction.

## Lancer l'application Streamlit

Depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run application_streamlit/App.py
```

## Formats de fichiers acceptés par Streamlit

### Mode RFM préparé

Le fichier doit contenir au minimum :

```text
Recency, Frequency, Monetary
```

### Mode transactionnel brut

Le fichier doit contenir au minimum :

```text
Customer ID, InvoiceDate, Invoice, Quantity, Price
```

Si la colonne `City` est présente, elle est conservée pour les filtres et l'analyse locale.

## Données

Le projet utilise le dataset public **Online Retail II**, adapté au contexte marocain.

La colonne pays a été transformée en villes représentant le réseau Marjane, par exemple :

- Casablanca
- Rabat
- Marrakech
- Tanger
- Agadir
- Fès
- Kenitra
- Oujda

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Mlxtend
- Matplotlib
- Seaborn
- Power BI
- Streamlit

## Auteur

**Ayoub Zouitine**

Étudiant en Data Science & AI Systems
