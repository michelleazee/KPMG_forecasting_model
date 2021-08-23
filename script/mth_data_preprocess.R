# ---------------------------------- #
# KPMG Forecasting Project
# Monthly data preprocessing code
#
# Author: Michelle A. Zee
# Email: maz2136@columbia.edu
#
# Date: June 29, 2021
# ---------------------------------- #

library(tidyverse)
library(lubridate)
library(stringr)

rm(list=ls())

setwd("G:/My Drive/KPMG/KPMG_forecasting_model/data/")

# ---- functions ----

pct_change <- function(data, period_lag) {
  data = as.numeric(data)
  data_prev = c(rep(NA, period_lag), data[2:length(data) - period_lag])
  pct_change = (data - data_prev)/ data
  
  pct_change
}

quarter_month <- as_tibble(cbind(c(1,1,1,2,2,2,3,3,3,4,4,4),
                                 c(1:12))) %>% 
  rename(quarter = V1, month = V2)

# ---- original dv ----

dv_qtr <- read_csv("raw/target_variable.csv")
colnames(dv_qtr) <- dv_qtr[1,]
dv_qtr <- dv_qtr[-c(1:2),]

dv_qtr <- dv_qtr %>%
  mutate(year = as.numeric(Year),
         quarter = as.numeric(gsub("Q", "", Quarter)),
         ebitda_dv = as.numeric(gsub("%", "", `Annualized EBITDA Per Share Growth minus 10-Year`))/100) %>%
  left_join(quarter_month, by = "quarter") %>%
  mutate(date = paste0(year, "-", month, "-01"),
         date = ymd(date)) %>%
  select(date, ebitda_dv)


dv_mth <- read_csv("raw/2021_6_24_new_target_variable_sp500.csv")
colnames(dv_mth) <- dv_mth[1,]
dv_mth <- dv_mth[-1,]

treasury <- read_csv("raw/mth_treasury_10yr.csv") %>%
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
  mutate(stock_idx_dv = esp_growth_annual - treasury) %>%
  filter(date < ymd("2022/1/1")) %>%
  select(date, stock_idx_dv)

# ---- business confidence ----

bus_conf <- read_csv("raw/mth_usa_bci.csv") %>%
  mutate(date = ym(TIME)) %>%
  mutate(bus_conf = pct_change(Value, 1)) %>%
  select(date, bus_conf)

# ---- consumer price index ----
cpi <- read_csv("raw/mth_cpi.csv") %>%
  mutate(data = gsub("\\s+", " ", data),
         type = word(data, 1)) %>%
  filter(type == "CUSR0000SA0") %>%
  mutate(year = word(data, 2),
         month = gsub("M", "", word(data, 3)),
         date = paste0(year, "-", month, "-", "01"),
         date = ymd(date),
         value = word(data, 4),
         cpi = pct_change(value, 1)) %>%
  select(date, cpi)

# ---- volatility index ----
vix <- read_csv("raw/vix_1990_to_present.csv") %>%
  mutate(date = mdy(DATE),
         year = year(date),
         month = month(date)) %>%
  group_by(year, month) %>%
  summarize(close = mean(CLOSE)) %>%
  ungroup() %>%
  mutate(date = paste0(year, "-", month),
         date = ym(date),
         vix = pct_change(close, 1)) %>%
  select(date, vix)

# ---- inventory ----

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
  select(year, quarter, inv) %>%
  left_join(quarter_month, by = "quarter") %>%
  mutate(date = paste0(year, "-", month, "-01"),
         date = ymd(date)) %>%
  select(date, inv)

# fed funds rate ----

fed_funds <- read_csv("raw/mth_fed_funds.csv")  %>%
  mutate(fed_funds = pct_change(FEDFUNDS, 1),
         fed_funds_cat = ifelse(fed_funds < 0, 1,
                                ifelse(fed_funds == 0, 2,
                                       ifelse(fed_funds > 0, 3, NA)))) %>%
  select(date = DATE, fed_funds, fed_funds_cat)



# join

df <- dv_mth %>%
  left_join(dv_qtr, by = "date") %>%
  left_join(bus_conf, by = "date") %>%
  left_join(cpi, by = "date") %>%
  left_join(vix, by = "date") %>%
  left_join(inv, by = "date") %>%
  left_join(fed_funds, by = "date")

write_csv(df, "final/mth_data_michelle.csv")



df_new <- df %>% 
  filter(!is.na(ebitda_dv))
cor(df_new$stock_idx_dv, df_new$ebitda_dv)

df_new <- df %>% 
  filter(!is.na(ebitda_dv)) %>%
  mutate(stock_idx_dv = ifelse(stock_idx_dv > 1, 1, stock_idx_dv))
cor(df_new$stock_idx_dv, df_new$ebitda_dv)

df_new <- df %>% 
#  filter(!is.na(ebitda_dv)) %>%
  mutate(stock_idx_dv = ifelse(stock_idx_dv > 2, 2, stock_idx_dv))

cor(df_new$stock_idx_dv, df_new$ebitda_dv)

ggplot(df_new, aes(stock_idx_dv, ebitda_dv, color = date)) +
  geom_point(position = "jitter", alpha = 0.5) + 
  geom_smooth(method = "lm") +
  labs(title = "Original Target vs Stock Index Target",
       x = "Stock Index Target",
       y = "Original EBITDA Target") +
  theme_minimal()

ggplot() +
  geom_line(df_new, aes(date, stock_idx_dv, color = "Stock Index DV")) +
  geom_line(df_new, aes(date, ebitda_dv, color = "Balance Sheet DV")) +
  labs(title = "Original Target vs Stock Index Target",
       x = "Years",
       y = "Target Variable") +
  scale_color_manual(name = "DV Calculation", values = c("blue", "red", "yellow")) +
  theme_minimal()

range(df$date)
