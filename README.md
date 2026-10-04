# Heart Disease Prediction System

Machine Learning assignment: a Support Vector Classifier that predicts heart disease using only **4 input attributes** (Sex, Chest Pain type, Exercise Induced Angina, Oldpeak) from the UCI Cleveland Heart Disease dataset. Built by modifying the Titanic Survival Prediction example. No Label Encoding is used (all inputs are numeric).

**Live app:** _paste your Streamlit link here_

## Files
- `Heart_Disease_Prediction.ipynb` - full notebook (load, train, test, apply, deploy)
- `app.py` - Streamlit web app
- `svc_trained_model.pkl` - trained model
- `requirements.txt` - packages for the web app (used by Streamlit Cloud)
- `requirements-notebook.txt` - extra packages to run the notebook (Jupyter, seaborn...)
- `processed.cleveland.data` - original dataset
- `sample-data.csv` - the 4 selected attributes + output

## Run locally
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Learning project only - not a medical tool.
