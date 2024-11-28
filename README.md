# Forecasting of photovoltaic power using deep learning

## Objective
This project is part of my master's degree research on applying deep learning techniques for forecasting photovoltaic (PV) power. The complete details of the research are available in the ACM Digital Library, accessible through this [link](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/ACMLC2024/3/437cea8a-6a10-11ef-ada9-16bb50361d1f/OUT/acmlc2024-3.html).

I had the privilege of presenting my research at the **2024 6th Asia Conference on Machine Learning and Computing (ACMLC)**, held in Bangkok, Thailand, from July 26–28, 2024. This project demonstrates key aspects of the research, including data preparation, model development, and evaluation, highlighting the potential of deep learning in renewable energy forecasting.

## Features
- **Data sources**:  The data is divided into two parts: weather data serving as the independent variables and PV power as the dependent variable
- **Forecast models**: Multiple forecast models are designed and constructed to identify correlations between PV power and weather data and to predict PV power
- **Assessment of model accuracy**: The accuracy of the models depends on the disparity between the forecasted value and the actual value

## Project Structure
```plaintext
├── Dataset/
│   ├── fore_data.csv   # The forecast data (NWP data) were collected between March 17 and 24, 2023. The dataset comprises a total of 104 samples.
│   └── hist_data.csv   # The historical data comprises weather data and PV power data collected from 2016 to 2019. 
│    
├── Models/
│   ├── forecast_models_with_NWP.ipynb       # Forecast models which predict PV power using NWP data 
│   ├── forecast_models_with_hist_data.ipynb # Forecast models which predict PV power using sequences of historical data
│   └── helper_functions.py                  # Helper functions for calculating accuracy, plotting graphs and etc.
|
└── README.md           # Overall summary of the project 
```

## Notes
The results displayed in the forecast_models Jupyter notebooks may not exactly match those presented in the research paper. This discrepancy arises because the notebooks were re-executed before being uploaded to GitHub to ensure that they remain functional and executable. 