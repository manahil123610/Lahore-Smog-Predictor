# Lahore Smog Predictor

Predicts tomorrow's PM2.5 in Lahore using a Random Forest trained on daily
Open-Meteo air quality and weather data (Aug 2022 to Nov 2026).

## How it works
1. `fetch_data.py` downloads hourly PM2.5, PM10, temperature, humidity, wind and pressure and averages them per day.
2. `train.py` builds features (today's values, 3 lag days of PM2.5, month), splits train/test by date, and compares Persistence, Linear Regression and Random Forest.
3. `app.py` is a Streamlit web app that loads the saved model and predicts tomorrow's PM2.5.

## Run locally
```
pip install -r requirements.txt
python fetch_data.py
python train.py
streamlit run app.py
```

## Results
<img width="823" height="407" alt="Screenshot 2026-11-23 164726" src="https://github.com/user-attachments/assets/53d04168-2b55-477e-8bb1-510336fd3351" />
<img width="800" height="464" alt="Screenshot 2026-11-23 164617" src="https://github.com/user-attachments/assets/f40b7935-b3ba-416d-b44d-b6f205c80490" />



## Limitations
- Open-Meteo pollution values are modelled (CAMS), not ground sensor readings.
- Single city, daily averages only.

Data: Open-Meteo (open-meteo.com). Learning project, not an official air quality alert.
