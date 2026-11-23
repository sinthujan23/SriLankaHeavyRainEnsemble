# 🌧️ Sri Lanka Heavy-Rain Early Warning: Ensemble Learning

**Author:** `<Your Name>` (`<Your ID>`) · Machine Learning – Ensemble Learning assignment
**🎥 Video presentation:** `<paste your YouTube / Google Drive link here>`

Predicts whether a Sri Lankan town will receive **heavy rain (≥ 10 mm) tomorrow** from today's weather. The project compares
**Bagging** (Random Forest, Extra Trees), **Boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost),
**Stacking** (RF + LightGBM + ANN → Logistic Regression) and a **Hybrid Random Forest + Neural Network**, and serves all
of them in an interactive **Streamlit** dashboard.

* **Local dataset:** NASA POWER daily weather for **20 towns in all 9 provinces** of Sri Lanka, 2016-01-01 → 2026-10-31
  (79,140 town-days, 15 variables including rainfall, humidity, pressure, cloud cover and soil wetness).
* **Split:** chronological. Train 2016–2024, test Jan 2025 – Oct 2026 (unseen future data).
* **Result:** the Random Forest (bagging) has the best F1 (0.568) and accuracy (0.820). XGBoost, Gradient Boosting and the
  hybrid share the best ROC-AUC (≈ 0.863). The notebook (Section 10) explains why.

## Project structure

```
├── Ensemble_Learning_SL_Heavy_Rain.ipynb   ← main deliverable: data → cleaning → features → 12 models → comparison → interpretation
├── app.py                                  ← Streamlit dashboard (overview map, live prediction, model comparison, feature importance)
├── requirements.txt
├── scripts/
│   └── fetch_data.py                       ← downloads the dataset from the NASA POWER API
├── data/
│   ├── sri_lanka_weather_daily.csv         ← raw dataset (20 towns × 3,957 days)
│   └── dashboard_features.parquet          ← engineered feature table used by the dashboard
├── models/
│   ├── *.joblib                            ← 12 trained pipelines (preprocessing + model)
│   └── metrics.csv, meta.json, …           ← test-set results shown in the dashboard
├── slides/                                 ← slide images used in the presentation video
└── .streamlit/config.toml                  ← dashboard theme
```

## How to run

```bash
pip install -r requirements.txt

# 1) (optional) re-download the data; it is already in data/
python scripts/fetch_data.py

# 2) (optional) re-run the notebook (≈ 8–10 min on a laptop); this also re-creates models/
jupyter nbconvert --to notebook --execute --inplace Ensemble_Learning_SL_Heavy_Rain.ipynb

# 3) start the dashboard  →  http://localhost:8501
streamlit run app.py
```

The trained models are included, so step 3 works straight after `pip install`.

## Generative-AI disclosure

This project was developed with help from **Claude (Anthropic), used through Claude Code**. It generated the code for
data collection, preprocessing, modelling and the dashboard, the slide images, and it drafted the explanations and a
suggested narration script. The video was recorded by the student. See Section 13 of the notebook.

## Data source

NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science/Applied Science Program,
<https://power.larc.nasa.gov> (MERRA-2 reanalysis, bias-corrected precipitation `PRECTOTCORR`).
