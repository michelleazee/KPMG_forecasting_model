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

#### 02. Recession Prediction

#### 03. Fed Funds Rate Prediction

#### 04. Short Term Overall Model

#### 05. Long Term Overall Model

#### 06. Overall Model Results Ensemble

#### 07. Consumer Discretionary Sector Model
##### Input: 
* ../data/raw/con_disc_pe_ratio.csv
* ../data/raw/con_disc_price.csv
* employment, unemployment...... data from Quandl
##### Output:
* ../data/model_outputs/consumer_discretionary_forecast.csv
##### Process:
1. Create target variable from price, P/E ratio, and Treasury Rate
2. Load data from csv and Quandl
3. Shift data for 1, 3, 6, 12, 18 month time horizons
4. Find highest correlation features
5. Find order/ seasonal order for SARIMAX Model
6. Choose features/ run models for each time horizon
7. Stitch model predictions to single 18-month forecast

#### 08. Financials Sector Model

#### 09. Information Technology Sector Model

The IT industry includes three major industry groups:

1. Software and services
2. Technology hardware and equipment
3. Semiconductor and semiconductor equipment

In terms of the model:

##### Input data (data source is Quandl unless otherwise stated): 
* Target variable calculated from Price, P/E ratio, and Treasury Rate
* Private fixed investment: Nonresidential: Intellectual property products: Software
* Value of Manufacturers' Shipments for Information Technology Industries
* Value of Manufacturers' New Orders for Information Technology Industries
* Value of Manufacturers' Total Inventories for Information Technology Industries
* Compensation of employees: Domestic private industries: Information
* Unemployment Rate: Information Industry, Private Wage and Salary Workers
* **Prediction results from overall market models (For 1, 3, 6, 12, 18 months)**

##### Process:
1. Create target variable from Price, P/E ratio, and Treasury Rate using Quandl
2. Load independent variables from Quandl and transform them into the wanted format (ex. Transform private fixed investment on intellectual property products from quarterly to monthly)
4. Load csv file of market level prediction results
5. Perform feature correlation analysis and select the independent variables of the model
6. Shift data for 1, 3, 6, 12, 18 month time horizons
7. Find order/ seasonal order for SARIMAX model of each horizon
8. Run models for each time horizon
9. Stitch model predictions to single 18-month forecast

#### 10. Telecommunications Sector Model

#### 11. Healthcare Sector Model

