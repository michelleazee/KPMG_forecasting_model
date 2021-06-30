# ---------------------------------- #
# KPMG Forecasting Project
# Data preprocessing code
#
# Author: Michelle A. Zee
# Email: maz2136@columbia.edu
#
# Date: June 8, 2021
# ---------------------------------- #

library(tidyverse)
library(lubridate)
library(stringr)

rm(list=ls())

setwd("G:/My Drive/KPMG/KPMG_forecasting_model/data/")

# ---- functions ----

add_quarter <- function(data, date_col, value_col) {
  data <- data %>% dplyr::rename(date = date_col, value = value_col)
  data$year <- year(data$date)
  data$quarter <- quarter(data$date)
  
  data %>%
    mutate(value = as.numeric(value)) %>%
    group_by(year, quarter) %>%
    dplyr::summarise(value = mean(value)) %>%
    ungroup()
}

qoq_change <- function(data, period_lag) {
  data = as.numeric(data)
  data_prev = c(rep(NA, period_lag), data[2:length(data) - period_lag])
  qoq_change = (data - data_prev)/ data
  
  qoq_change
}


# ---- original dv ----

dv_qtr <- read_csv("raw/target_variable.csv")
colnames(dv) <- dv[1,]
dv_qtr <- dv[-c(1:2),]

dv_qtr <- dv %>%
  mutate(year = as.numeric(Year),
         quarter = as.numeric(gsub("Q", "", Quarter)),
         dv = as.numeric(gsub("%", "", `Annualized EBITDA Per Share Growth minus 10-Year`))/100) %>%
  select(year, quarter, dv)


dv_mth <- read_csv("raw/2021_6_24_new_target_variable_sp500.csv")
colnames(dv) <- dv[1,]
dv_mth <- dv[-1,]

treasury_mth <- read_csv("raw/treasury_10yr.csv") %>%
  mutate(date = ymd(DATE),
         treasury = as.numeric(GS10)/100) %>% 
  select(-DATE, -GS10)

dv_mth <- dv_mth %>% 
  select(date = Date, PE_ratio = `S&P 500 PE Ratio`, price = `S&P 500 Price`,
         eps = `Calculated EPS (LTM)`, eps_growth = `EPS (LTM) Growth`) %>%
  mutate(date = dmy(date),
         esp_growth_annual = gsub("%", "", eps_growth),
         esp_growth_annual = (as.numeric(esp_growth_annual)/100 + 1)^12 - 1) %>%
  left_join(treasury, by = "date") %>%
  mutate(target_variable = esp_growth_annual - treasury) %>%
  filter(date < ymd("2022/1/1"))

# ---- business confidence ----
bus_conf <- read_csv("raw/business_conf_idx.csv") %>%
  mutate(year = as.numeric(gsub("-.*", "", TIME)),
         month = as.numeric(str_remove(TIME, paste0(year, "-"))),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA))))) %>%
  filter(year > 2000) %>%
  group_by(year, quarter) %>%
  dplyr::summarise(value = mean(Value)) %>%
  ungroup() %>%
  mutate(value_prev = c(NA, value[2:length(value) - 1]),
         bus_conf = (value - value_prev)/ value) %>%
  select(year, quarter, bus_conf)

# ---- consumer credit ----

con_cred <- read_csv("raw/total_consumer_credit_in_millions_dollars.csv")
con_cred <- con_cred[-c(1:5),] %>%
  dplyr::rename(TIME = `Series Description`,
         value = `Total consumer credit owned and securitized, not seasonally adjusted level`) %>%
  mutate(year = as.numeric(gsub("-.*", "", TIME)),
       month = as.numeric(str_remove(TIME, paste0(year, "-"))),
       quarter = ifelse(month %in% c(1:3), 1,
                        ifelse(month %in% c(4:6), 2,
                               ifelse(month %in% c(7:9), 3,
                                      ifelse(month %in% c(10:12), 4, NA)))),
       value = as.numeric(value)) %>%
  filter(year > 2000) %>%
  group_by(year, quarter) %>%
  dplyr::summarise(value = mean(value)) %>%
  ungroup() %>%
  mutate(value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         con_cred = (value - value_prev)/ value) %>%
  select(year, quarter, con_cred)

# ---- consumer price index ----

cpi <- read_csv("raw/CPI.csv")

cpi <- cpi %>%
  mutate(time_parsed = gsub("/", " ", DATE),
         year = word(time_parsed, 1),
         year = as.numeric(year),
         month = word(time_parsed, 2),
         month = as.numeric(month),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA)))),
         cpi = cpi/100) %>%
  select(year, quarter, cpi)

# ---- personal consumption expenditure ----

con_exp <- read_csv("raw/personal_consumption_exp.csv")

con_exp <- con_exp %>%
  mutate(time_parsed = gsub("/", " ", DATE),
         year = word(time_parsed, -1),
         year = as.numeric(year),
         month = word(time_parsed, 1),
         month = as.numeric(month),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA)))),
         con_exp = pce/100) %>%
  select(year, quarter, con_exp)

# ---- producer price index ----

ppi <- read_csv("raw/PPI.csv")

ppi <- ppi %>%
  mutate(time_parsed = gsub("/", " ", DATE),
         year = word(time_parsed, 1),
         year = as.numeric(year),
         month = word(time_parsed, 2),
         month = as.numeric(month),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA)))),
         ppi = ppi/100) %>%
  select(year, quarter, ppi)

# ---- volatility index ----

vix <- read_csv("raw/volatility_idx.csv")

vix <- vix %>%
  mutate(time_parsed = gsub("/", " ", DATE),
         year = word(time_parsed, -1),
         year = as.numeric(year),
         month = word(time_parsed, 1),
         month = as.numeric(month),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA)))),
         value = OPEN) %>%
  filter(year > 2000) %>%
  group_by(year, quarter) %>%
  dplyr::summarise(value = mean(value)) %>%
  ungroup() %>%
  mutate(value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         vix = (value - value_prev)/ value) %>%
  select(year, quarter, vix)

# ---- inv items ----

sp500bs <- read_csv("raw/S P 500 SPX Financials Balance Sheet.csv")
    
colnames(sp500bs) <- sp500bs[8,]
sp500bs <- sp500bs[-c(1:7),]

inv <- sp500bs[8,] %>%
  pivot_longer(cols = c(2:81)) %>%
    mutate(time_parse = gsub("3 months\nCQ", "", name),
           year = as.numeric(word(time_parse, -1)),
           quarter = as.numeric(word(time_parse, 1)),
           
           value = as.numeric(value),
           value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
           inv = (value - value_prev)/ value) %>%
  select(year, quarter, inv)

# ---- ppe ----

ppe <- sp500bs[11,] %>%
  pivot_longer(cols = c(2:81)) %>%
  mutate(time_parse = gsub("3 months\nCQ", "", name),
         year = as.numeric(word(time_parse, -1)),
         quarter = as.numeric(word(time_parse, 1)),
         
         value = as.numeric(value),
         value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         ppe = (value - value_prev)/ value) %>%
  select(year, quarter, ppe)


# ---- cad to usd ----

cad_usd <- read_csv("raw/fred_cad2usd.csv")
cad_usd <- add_quarter(cad_usd, "DATE", "EXCAUS") %>%
  mutate(cad_usd = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- cny to usd ----

cny_usd <- read_csv("raw/fred_cny2usd.csv")
cny_usd <- add_quarter(cny_usd, "DATE", "EXCHUS") %>%
  mutate(cny_usd = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- inr to usd ----

inr_usd <- read_csv("raw/fred_inr2usd.csv")
inr_usd <- add_quarter(inr_usd, "DATE", "EXINUS") %>%
  mutate(inr_usd = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- jpy to usd ----

jpy_usd <- read_csv("raw/fred_jpy2usd.csv")
jpy_usd <- add_quarter(jpy_usd, "DATE", "EXJPUS") %>%
  mutate(jpy_usd = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- usd to aud ----

usd_aud <- read_csv("raw/fred_usd2aud.csv")
usd_aud <- add_quarter(usd_aud, "DATE", "EXUSAL") %>%
  mutate(usd_aud = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- usd to euro ----

usd_euro <- read_csv("raw/fred_usd2euro.csv")
usd_euro <- add_quarter(usd_euro, "DATE", "EXUSEU") %>%
  mutate(usd_euro = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- usd to gbp ----

usd_gbp <- read_csv("raw/fred_usd2gbp.csv")
usd_gbp <- add_quarter(usd_gbp, "DATE", "EXUSUK") %>%
  mutate(usd_gbp = qoq_change(value, period_lag = 1)) %>%
  select(-value)


# ---- wages ----

wages <- read_csv("raw/wages.csv")
wages <- wages %>%
  mutate(wages = qoq_change(Wages, period_lag = 1)) %>%
  select(year = Year, quarter = Quarter, wages)

# ---- employee ----

employee <- read_csv("raw/employee.csv")

colnames(employee) <- employee[13,]
employee <- employee[-c(1:13),] %>% select(-Annual)

employee <- employee %>%
  pivot_longer(cols = c(Jan:Dec), names_to = "month") %>%
  mutate(quarter = ifelse(month %in% c("Jan", "Feb", "Mar"), 1,
                          ifelse(month %in% c("Apr", "May", "Jun"), 2,
                                 ifelse(month %in% c("Jul", "Aug", "Sep"), 3,
                                        ifelse(month %in% c("Oct", "Nov", "Dec"), 4, NA)))),
         value = as.numeric(value),
         year = as.numeric(Year)) %>%
  group_by(year, quarter) %>%
  dplyr::summarise(value = mean(value)) %>%
  ungroup() %>%
  mutate(employee = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- unemployment rate ----

unemploy <- read_csv("raw/unemployment_rate.csv")
colnames(unemploy) <- unemploy[2,]
unemploy <- unemploy[-c(1:2),]
unemploy <- unemploy[-c(881:882),] %>%
  mutate(date = mdy(Period))

unemploy <- add_quarter(unemploy, "date", "Value") %>%
  mutate(unemploy = qoq_change(value, period_lag = 1)) %>%
  select(-value)

# ---- s&p 500 index value ----

sp500_value <- read_csv("raw/sp500_index_value.csv") %>%
  mutate(date = mdy(Dates)) %>%
  dplyr::rename(value = `S&P 500 (^SPX) - Index Value`)

sp500_value <- add_quarter(sp500_value, "date", "value") %>%
  mutate(sp500_value = qoq_change(value, period_lag = 1)) %>%
  select(-value)
   
# ---- part 1 data ----

part1 <- read_csv("final/Part1_data.csv") %>%
  mutate(date = yq(Date),
         year = year(date),
         quarter = quarter(date),
         gdp = qoq_change(GDP, period_lag = 1),
         real_gdp = qoq_change(real_GDP, 1),
         gdp_p_deflat = qoq_change(`Gross Domestic Product: Implicit Price Deflator`, 1),
         gnp = qoq_change(`Gross National Product`, 1),
         gov_con_exp = qoq_change(`Government Consumption Expenditures and Gross Investment, Billions of Dollars, Quarterly, Seasonally Adjusted Annual Rate`, 1),
         gov_tot_exp = qoq_change(`Government total expenditures`, 1),
         sh_gdp_con_exp = qoq_change(`Shares of gross domestic product: Personal consumption expenditures, Percent, Quarterly, Not Seasonally Adjusted`, 1),
         sh_gdp_pr_dom_inv = qoq_change(`Shares of gross domestic product Gross private domestic investment`, 1),
         sh_gdp_gov_exp = qoq_change(`Shares of gross domestic product Government consumption expenditures and gross investment`, 1),
         sh_gdp_export = qoq_change(`Shares of gross domestic product: Net exports of goods and services, Percent, Quarterly, Not Seasonally Adjusted`, 1)
         ) %>%
  select(year, quarter, gdp, real_gdp, gdp_p_deflat, gnp, gov_con_exp, gov_tot_exp,
         sh_gdp_con_exp, sh_gdp_pr_dom_inv, sh_gdp_gov_exp, sh_gdp_export)

# ---- export price index ----

export_p_idx <- read_csv("raw/fred_exportpriceindex.csv") %>%
  mutate(value = as.numeric(IQ))

export_p_idx <- add_quarter(export_p_idx, "DATE", "value") %>%
  mutate(export_p_idx = qoq_change(value, 1)) %>%
  select(-value)

# ---- import price index ----

import_p_idx <- read_csv("raw/fred_importpriceindex.csv") %>%
  mutate(value = as.numeric(IR))

import_p_idx <- add_quarter(import_p_idx, "DATE", "value") %>%
  mutate(import_p_idx = qoq_change(value, 1)) %>%
  select(-value)

# ---- new order capital goods nondefense ----

order_cap_good_nondef <- read_csv("raw/fred_neworder_capgood_nondefense.csv") %>%
  mutate(value = as.numeric(NEWORDER))

order_cap_good_nondef <- add_quarter(order_cap_good_nondef, "DATE", "value") %>%
  mutate(order_cap_good_nondef = qoq_change(value, 1)) %>%
  select(-value)

# ---- new order durable ----

order_dur <- read_csv("raw/fred_neworder_durable.csv") %>%
  mutate(value = as.numeric(DGORDER))

order_dur <- add_quarter(order_dur, "DATE", "value") %>%
  mutate(order_dur = qoq_change(value, 1)) %>%
  select(-value)


# ---- new order durable, excluding transportation ----

order_dur_excl_trans <- read_csv("raw/fred_neworder_durable_excl_transportation.csv") %>%
  mutate(value = as.numeric(ADXTNO))

order_dur_excl_trans <- add_quarter(order_dur_excl_trans, "DATE", "value") %>%
  mutate(order_dur_excl_trans = qoq_change(value, 1)) %>%
  select(-value)

# ---- shipment durable ----

ship_dur <- read_csv("raw/fred_shipments_durable.csv") %>%
  mutate(value = as.numeric(AMDMVS))

ship_dur <- add_quarter(ship_dur, "DATE", "value") %>%
  mutate(ship_dur = qoq_change(value, 1)) %>%
  select(-value)

# ---- shipments consumer durable ----

ship_con_dur <- read_csv("raw/fred_shipments_consumer_durable.csv") %>%
  mutate(value = as.numeric(ACDGVS))

ship_con_dur <- add_quarter(ship_con_dur, "DATE", "value") %>%
  mutate(ship_con_dur = qoq_change(value, 1)) %>%
  select(-value)

# ---- shipments consumer durable, excluding defense ----

ship_con_dur_excl_def <- read_csv("raw/fred_shipments_consumer_durable_exclud_defense.csv") %>%
  mutate(value = as.numeric(ADXDVS))

ship_con_dur_excl_def <- add_quarter(ship_con_dur_excl_def, "DATE", "value") %>%
  mutate(ship_con_dur_excl_def = qoq_change(value, 1)) %>%
  select(-value)

# ---- shipments consumer durable, excluding defense ----

ship_con_dur_excl_trans <- read_csv("raw/fred_shipments_consumer_durable_exclud_transportation.csv") %>%
  mutate(value = as.numeric(ADXTVS))

ship_con_dur_excl_trans <- add_quarter(ship_con_dur_excl_trans, "DATE", "value") %>%
  mutate(ship_con_dur_excl_trans = qoq_change(value, 1)) %>%
  select(-value)


# ---- join ----

df <- dv %>%
  left_join(bus_conf, by = c("year", "quarter")) %>%
  left_join(con_cred, by = c("year", "quarter")) %>%
  left_join(cpi, by = c("year", "quarter")) %>%
  left_join(con_exp, by = c("year", "quarter")) %>%
  left_join(ppi, by = c("year", "quarter")) %>%
  left_join(vix, by = c("year", "quarter")) %>%
  left_join(inv, by = c("year", "quarter")) %>%
  left_join(ppe, by = c("year", "quarter")) %>%
  
  left_join(cad_usd, by = c("year", "quarter")) %>%
  left_join(cny_usd, by = c("year", "quarter")) %>%
  left_join(inr_usd, by = c("year", "quarter")) %>%
  left_join(jpy_usd, by = c("year", "quarter")) %>%
  left_join(usd_aud, by = c("year", "quarter")) %>%
  left_join(usd_euro, by = c("year", "quarter")) %>%
  left_join(usd_gbp, by = c("year", "quarter")) %>%
  left_join(wages, by = c("year", "quarter")) %>%
  left_join(employee, by = c("year", "quarter")) %>%
  left_join(unemploy, by = c("year", "quarter")) %>%
  left_join(sp500_value, by = c("year", "quarter")) %>%
  left_join(part1, by = c("year", "quarter")) %>%
  
  left_join(export_p_idx, by = c("year", "quarter")) %>%
  left_join(import_p_idx, by = c("year", "quarter")) %>%
  left_join(order_cap_good_nondef, by = c("year", "quarter")) %>%
  left_join(order_dur, by = c("year", "quarter")) %>%
  left_join(order_dur_excl_trans, by = c("year", "quarter")) %>%
  left_join(ship_dur, by = c("year", "quarter")) %>%
  left_join(ship_con_dur, by = c("year", "quarter")) %>%
  left_join(ship_con_dur_excl_def, by = c("year", "quarter")) %>%
  left_join(ship_con_dur_excl_trans, by = c("year", "quarter"))


write_csv(df, "final/data_michelle.csv")

