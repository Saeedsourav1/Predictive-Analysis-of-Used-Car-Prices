# Used Car Price Prediction

**Predict the price of used Toyota cars using machine learning**  
A complete end-to-end machine learning project that performs Exploratory Data Analysis (EDA), builds regression models (Linear Regression + XGBoost), and tunes hyperparameters to achieve strong predictive performance on real Toyota used-car data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-yellow)
![pandas](https://img.shields.io/badge/pandas-2.0+-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7+-red)
![seaborn](https://img.shields.io/badge/seaborn-0.12+-purple)

## 📊 Project Overview

This project uses a dataset of ~6,700 Toyota used cars (from the UK market) to predict selling price based on features like:

- Model (Yaris, Auris, GT86, etc.)
- Year
- Mileage
- Fuel type
- Transmission
- MPG
- Engine size

I have compared two approaches:
- **Baseline**: Linear Regression (with one-hot encoding)
- **Advanced**: XGBoost Regressor (with hyperparameter tuning via GridSearchCV)

**Best model performance (on test set):**
- RMSE ≈ 1,100–1,300 (depending on tuning)
- R² ≈ 0.96–0.97

## ✨ Key Features

- Clean, modular code using scikit-learn **Pipelines**
- Proper handling of categorical variables (OneHotEncoder)
- Comprehensive EDA with visualizations
- Hyperparameter tuning with GridSearchCV
- Model comparison (Linear vs XGBoost)
- Sample prediction table

## 📁 Project Structure

```
Predictive-Analysis-of-Used-Car-Prices/
├── toyota_price_prediction.ipynb     # Main Jupyter notebook (clean version)
├── toyota.csv                        # Original dataset (~6.7k rows)
├── README.md                         # This file
└── images                            # Screenshots of plots & results
    
```


## 🛠️ How to Run

1. Make sure `toyota.csv` is in the same folder
2. Run all cells top to bottom
3. The notebook will:
   - Load & explore data
   - Show visualizations
   - Train Linear Regression
   - Train & tune XGBoost
   - Show predictions & metrics

## 📈 Results Highlights

| Model              | RMSE (test set) | R² Score   |
|--------------------|-----------------|------------|
| Linear Regression  | ~3,500–4,000    | ~0.85      |
| XGBoost (tuned)    | ~1,200          | ~0.965     |

→ XGBoost significantly outperforms the linear baseline thanks to its ability to capture non-linear patterns and interactions.

Sample predictions:

| Actual Price | LinearReg Pred | XGBoost Pred |
|--------------|----------------|--------------|
| 37,440       | 32,800         | 35,200       |
| 4,159        | 4,900          | 4,350        |
| 10,600       | 11,200         | 10,850       |
| ...          | ...            | ...          |


## 🔧 Technologies Used

- **Language**: Python
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (Pipeline, OneHotEncoder, GridSearchCV), XGBoost
- **Environment**: Jupyter Notebook

## 📄 Dataset

- Source: Commonly available Toyota used car dataset (UK market)
- File: `toyota.csv`
- Rows: ~6,738
- Columns: model, year, price, transmission, mileage, fuelType, mpg, engineSize
