# KPMG Forecasting Model

**Team members**: Taotao Jiang, Albert Li, Peishan Li, Christina Lv, Jinghan (Katherine) Ma, Kushal Wijesundara, Michelle A. Zee

## Project Description

This project was developed by graduate students in Columbia University's Quantitative Methods in the Social Sciences program in collaboration with KPMG US. The purpose was to develop models that forecast the earnings growth in the overall S&P 500 Index as well as the S&P 500 Consumer Discretionary, Financials, Information Technology, Telecommunications, and Healthcare sectors. Models were constructed using traditional economic indicators as well as predicted recession and monetary policy features.

Additional information about the project can be found on the project slides [here](https://drive.google.com/file/d/1e8A9KN6Vd7Jnt7rs3f0el_kzR8H85jv8/view?usp=sharing).

Team member contact information can also be found on the slides.


## Folder organization

| -- [script](/script)

| -- [data](/data)              

| | -- [raw](/data/raw)

| | -- [processed](/data/processed)

| | -- [model_outputs](/data/model_outputs)

| -- fed_funds_rate <- leads to separate repo

## Code Description

#### 01. Data Cleaning

#### 02. Recession Prediction Model

#### 03. Fed Funds Rate Prediction

#### 04. Short Term Overall Model

#### 05. Long Term Overall Model

#### 06. Overall Model Results Ensemble

#### 07. Consumer Discretionary Sector Model
##### Input: 
* ../data/raw/consumer_discretionary_target.csv
* employment, unemployment...... data from Quandl
##### Output:
* ../data/model_outputs/consumer_discretionary_forecast.csv
##### Process:
1. Load data from csv and Quandl
2. Shift data for 1, 3, 6, 12, 18 month time horizons
3. Find highest correlation features
4. Find order/ seasonal order for SARIMAX Model
5. Choose features/ run models for each time horizon
6. Stitch model predictions to single 18-month forecast

#### 08. Financials Sector Model

#### 09. Information Technology Sector Model

#### 10. Telecommunications Sector Model

#### 11. Healthcare Sector Model

