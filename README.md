# SFE - Optimisation de la Gestion Client chez Marjane

## Description du projet

Ce projet est realise dans le cadre de mon **Stage de Fin d'Etudes** au sein de Marjane.

L'objectif est d'exploiter des donnees transactionnelles retail afin d'ameliorer la comprehension du comportement client et d'aider a la prise de decision commerciale.

Le projet repose sur trois axes Data Science :

- **Segmentation RFM des clients** : identification des profils clients selon la recence, la frequence et le montant d'achat.
- **Analyse du panier d'achat** : extraction de regles d'association pour identifier les produits souvent achetes ensemble.
- **Prediction du churn client** : detection des clients presentant un risque d'inactivite ou de depart.

Les resultats sont exploites a travers :

- un dashboard **Power BI** pour le suivi decisionnel ;
- un dashboard **Streamlit** pour la prediction interactive du churn et les recommandations panier.

## Titre du SFE

**Optimisation de la gestion client chez Marjane par l'analyse des donnees transactionnelles**

## Structure du depot

```text
SFE-Optimisation-Gestion-Client-Marjane/
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
│   ├── 01_test_transactionnel_brut.csv
│   ├── 02_test_transactionnel_colonnes_alternatives.csv
│   └── 03_test_clients_rfm_prepare.csv
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
│   ├── powerbi_kpis.csv
│   └── Dashboard_Intelligence_Client_Marjane.pbix
├── requirements.txt
└── README.md
```

## Notebooks

### 1. Segmentation RFM des Clients Marjane

Fichier :

`notebooks/01_Segmentation_RFM_Clients_Marjane.ipynb`

Ce notebook prepare les donnees, calcule les metriques RFM et applique un clustering K-Means afin d'identifier les principaux segments clients :

- VIP / Champions
- Clients reguliers
- Clients a risque / perdus
- Gros acheteurs occasionnels

### 2. Analyse du Panier et Regles d'Association

Fichier :

`notebooks/02_Analyse_Panier_Association_Marjane.ipynb`

Ce notebook applique l'analyse du panier d'achat avec FP-Growth afin d'extraire les associations de produits les plus pertinentes.

Les indicateurs utilises sont :

- support ;
- confiance ;
- lift.

Une cellule d'export permet aussi de sauvegarder les regles dans :

`donnees/regles_association.csv`

### 3. Prediction du Churn Client

Fichier :

`notebooks/03_Prediction_Churn_Clients_Marjane.ipynb`

Ce notebook construit une variable cible de churn a partir d'un seuil d'inactivite de 180 jours, entraine des modeles supervises et sauvegarde le meilleur modele pour l'application Streamlit.

Modeles utilises :

- Random Forest
- XGBoost

## Dashboards

### Dashboard Power BI

Le dossier `reporting_powerbi/` contient les fichiers de donnees utilises pour le reporting Power BI :

- `powerbi_clients.csv`
- `powerbi_kpis.csv`
- `Dashboard_Intelligence_Client_Marjane.pbix`

Le dashboard Power BI presente :

- les KPIs globaux ;
- la repartition des clients par ville ;
- les segments RFM ;
- les niveaux de risque de churn.

### Dashboard Streamlit

L'application Streamlit se trouve dans :

`application_streamlit/App.py`

Elle permet de :

- importer un fichier client prepare avec les metriques RFM ;
- importer un fichier transactionnel brut ;
- reconnaitre certains noms de colonnes alternatifs ;
- calculer automatiquement les metriques RFM si necessaire ;
- predire la probabilite de churn ;
- afficher le niveau de risque ;
- exploiter l'analyse du panier avec les regles d'association ;
- proposer des recommandations de cross-selling ;
- telecharger les resultats de prediction et les regles d'association.

## Lancer l'application Streamlit

Depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run application_streamlit/App.py
```

## Fichiers de test

Le dossier `donnees_test/` contient trois fichiers pour tester toutes les fonctionnalites du dashboard :

- `01_test_transactionnel_brut.csv` : fichier transactionnel standard, teste le churn et l'analyse du panier.
- `02_test_transactionnel_colonnes_alternatives.csv` : fichier transactionnel avec noms de colonnes alternatifs.
- `03_test_clients_rfm_prepare.csv` : fichier client deja prepare avec les metriques RFM.

## Formats de fichiers acceptes par Streamlit

### Mode RFM prepare

Le fichier doit contenir au minimum :

```text
Recency, Frequency, Monetary
```

### Mode transactionnel brut

Le fichier doit contenir au minimum :

```text
Customer ID, InvoiceDate, Invoice, Quantity, Price
```

Si la colonne `City` est presente, elle est conservee pour les filtres et l'analyse locale.

Pour activer la partie analyse du panier dans Streamlit, le fichier transactionnel doit aussi contenir :

```text
Invoice, Description, Quantity
```

## Donnees

Le projet utilise le dataset public **Online Retail II**, adapte au contexte marocain.

La colonne pays a ete transformee en villes representant le reseau Marjane, par exemple :

- Casablanca
- Rabat
- Marrakech
- Tanger
- Agadir
- Fes
- Kenitra
- Oujda

## Technologies utilisees

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

Etudiant en Data Science & AI Systems
