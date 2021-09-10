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

### 01. Data Cleaning

#### 02. Recession Prediction
The recession prediction model helps capture turning points in the business cycles and send near-term recession siganls; therefore, the results of the recession model is incorporated as an engineered feature into the overall market short-term model to improve overall model accuracy. Specifically, this recession model takes the difference between 10-year and 3-month Treasury rates (also defined as the “yield spread”), as well as other economic indicators to calculate recession probability in 12-month ahead horizon.

#### Input data: The independent variables including adjusted yield spread and several economic indicators are used to predict 12-month ahead NBER recession dates. 

<ins>Independent Variables</ins>  

* Yield Spread/Inversion of the Yield Curve =  10-Year Treasury Constant Maturity Rate - 3-month treasury bill - secondary market rate (Quandl)
* Adjusted Yield Spread - equation reference: [Robobank Reference](https://research.rabobank.com/publicationservice/download/publication/token/8Wsu7NXGcHfF4ZkWAkjg)
* Unemployment Rate (Quandl) 
* Fed Funds Rate (Quandl) 
* CPI (Quandl)

<ins>Dependent Variables (DV) </ins> : NBER recession dates (Quandl): 
* NBERt,t+12, equals one if there is an NBER recession starting at any time in the 12 months that follow the observed independent variables, and zero otherwise. 

#### Output data:

#### Process:

1. Load both independent and dependent variable data from quandl (note: quandl API key needed) 
2. Fill recession dates: replace NA values by zeros & visualize recession data 
3. Transform raw data: 
   * Unemployment Rate: log differential
   * CPI: log differential
   * Fed Funds Rate: differential
   * Adjusted Term Spread using equation from Robobank
4. Manipulate recession dates (DV) to prepare for 12-month prediction horizon
5. Select three machine learning models: Guassian, SVM, Random Forest Classifier
   * Set training set ["1962-01-31":"2006-01-31"] and testing set ["2006-01-31" and onwards]
   * Calculate model accuracy: AUC score, True & False Positive rates 
   * Choice the best performing model (SVM) based on model performance metrics
6. Predict recession probability using *model.predict_proba*
7. Visualize model performance
   


#### 03. Fed Funds Rate Prediction

#### 04. Short Term Overall Model

For the short term, we're using a **Facebook Prophet Model**. After trying a few methods to improve the model, the final model is an **additive model with both endogenous trend decomposition and exogenous regressors**. 

**We've adopted a Rolling Window Cross Validation approach**, where we'll be predicting from the cutoff point to cutoff+horizon. The time before cutoff point, length of which is the intial/lookback period, is the training set. The training set and prediction results rolls along the entire dataframe.

##### Input data:
* data_short_term.csv (In which target variable is from the client and other data are previous processed data) **We used Recursive Feature Elimination(RFE) to identify the most important features from all the data we have which could be found in the data folder of this repository**
* COVID_Boolean.csv (We identify that Covid starts from March, 2020, therefore starting from March of 2020 the variable value equals 1, whereas previous values equal to 0.)
* preprocessed_recession_probability.csv (Preprocessed based on recession prediction results from **02. Recession Prediction**)
* preprocessed_Fed_minutes.csv (Preprocessed based on Fed minutes prediction results from **03. Fed Funds Rate Prediction**)

##### Output data:
* Prediction results of 1-12 month horizon from the Facebook Prophet Model (Path: KPMG_forecasting_model/script/Final Short-term Model/**X**_month_prediction.csv)

##### Process (short-term_model_with_new_variables.ipynb)
1. Load target variable and economic indicator independent variables (whose data processing procedure and data source could be found in the data and script folder)
2. Load data for recession probability prediction, fed funds rate prediction and the Covid Boolean
3. Use the **generate_model_result** function, which is the concentrate of our intermediate results and thoughts of model improvement; **The parameters are tuned such that seasonality is set as 'multiplicative', initial as 6 years, horizon as 1-12 months, and cutoff half of the horizons respectively)**
4. Apply the function with different horizons (1-12 month) and write csv files for each
5. As these files include duplicate rows, use the **updatedf** function to eliminate duplicate rows
6. Apply the function and the output would be the prediction results wanted
7. Plot the predictions and residules

#### 05. Long Term Overall Model
#### Input: 
* ../data/raw/con_disc_pe_ratio.csv
* ../data/raw/con_disc_price.csv
* ../data/final/mth_data_michelle.csv
##### Output:
* ../script/Long-term model/Results_for_sector/

#### 06. Overall Model Results Ensemble
#### Input: 
* ../script/Final Short-term Model/Results/
* ../script/Long-term model/Results_for_sector/
##### Output:
* ../script/Overall model Ensemble/Results/

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

##### Input: 




##### Process:
1. Create target variable from price, P/E ratio, and Treasury Rate
2. Load data from Quandl
3. Shift independent variable data for 1, 3, 6, 12, 18 month time horizons
4. Find highest correlation features
5. Find order/ seasonal order for SARIMAX Model
6. Choose features/ run models for each time horizon
7. Stitch model predictions to single 18-month forecast

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

##### Input data (data source is Quandl unless otherwise stated): 
* PE_daily_normalized_tele.csv (download from ciq)
* daily_price_tele.csv (download from ciq)
* one_month_prediction.csv (prediction from short-term model)
* twelve_month_prediction.csv (prediction from short-term model)
* eighteen_month_prediction.csv (prediction from short-term model)
* exogenous variable data from quandl 

##### Process:
1. Create target variable from Price, P/E ratio, and Treasury Rate using Quandl
2. Load independent variables from Quandl and transform them into the wanted format
3. Load csv file of market level prediction results
4. Perform feature correlation analysis and select the independent variables of the model
5. Shift data for 1, 3, 6, 12, 18 month time horizons
6. Find order/ seasonal order for SARIMAX model of each horizon
7. Run models for each time horizon

#### 11. Healthcare Sector Model

The Healthcare Sector includes 4 sub-industries:

1. Services & Facilities
2. Medical Devices & Equipment
3. Medical Insurance & Managed Care
4. Pharmaceuticals

##### Input data: 
* % aging population - World Bank
* US national health expenditure as % of GDP - Bureau of Labor Statistics
* % of medical policyholders - US Census Bureau
* number of medical SaaS startups - Crunchbase
* research & development expenditure - Congressional Budget Office
* all variables organized into one file: /data/raw/hc_data.csv
##### Output:
* Short-term: /script/Sector_Models/healthcare_univariate_sarima.ipynb
* Long-term: /script/Sector_Models/healthcare_regression.ipynb
* Correlation Matrix: /script/Sector_Models/correlation_matrix_healthcare.html
##### Process:
1. Create target variable from Price, P/E ratio, and Treasury Rate using Quandl
2. Load csv file of independent variables and target variable
3. Conduct correlation analysis and draw correlation matrix map
4. Run univariate sarima for short-term (1-9 months) forecasting
5. Run ensemble model incorporating sarima and random forest regression for long-term forecasting: 1) shifting data for 12, 15, 18 month time horizons and running models for each time horizon; 2) adjusting weights assigned to sarima and random forest regression to reach the best scenario
