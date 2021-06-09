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

# ---- business confidence ----
bus_conf <- read_csv("business_conf_idx.csv") %>%
  mutate(year = as.numeric(gsub("-.*", "", TIME)),
         month = as.numeric(str_remove(TIME, paste0(year, "-"))),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA))))) %>%
  filter(year > 2000) %>%
  group_by(year, quarter) %>%
  summarise(value = mean(Value)) %>%
  ungroup() %>%
  mutate(value_prev = c(NA, value[2:length(value) - 1]),
         qoq_change = (value - value_prev)/ value) %>%
  select(year, quarter, qoq_change)

# ---- consumer credit ----

con_cred <- read_csv("total_consumer_credit_in_millions_dollars.csv")
con_cred <- con_cred[-c(1:5),] %>%
  rename(TIME = `Series Description`,
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
  summarise(value = mean(value)) %>%
  ungroup() %>%
  mutate(value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         qoq_change = (value - value_prev)/ value) %>%
  select(year, quarter, qoq_change)

# ---- consumer price index ----

cpi <- read_csv("CPI.csv")

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
         qoq_change = cpi/100) %>%
  select(year, quarter, qoq_change)

# ---- personal consumption expenditure ----

cons_exp <- read_csv("personal_consumption_exp.csv")

cons_exp <- cons_exp %>%
  mutate(time_parsed = gsub("/", " ", DATE),
         year = word(time_parsed, -1),
         year = as.numeric(year),
         month = word(time_parsed, 1),
         month = as.numeric(month),
         quarter = ifelse(month %in% c(1:3), 1,
                          ifelse(month %in% c(4:6), 2,
                                 ifelse(month %in% c(7:9), 3,
                                        ifelse(month %in% c(10:12), 4, NA)))),
         qoq_change = pce/100) %>%
  select(year, quarter, qoq_change)

# ---- producer price index ----

ppi <- read_csv("PPI.csv")

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
         qoq_change = ppi/100) %>%
  select(year, quarter, qoq_change)

# ---- volatility index ----

vix <- read_csv("volatility_idx.csv")

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
  summarise(value = mean(value)) %>%
  ungroup() %>%
  mutate(value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         qoq_change = (value - value_prev)/ value) %>%
  select(year, quarter, qoq_change)

# ---- s&p 500 balance sheet items

sp500bs <- read_csv("S P 500 SPX Financials Balance Sheet.csv")
    
colnames(sp500bs) <- sp500bs[8,]
sp500bs <- sp500bs[-c(1:7),]

inv <- sp500bs[8,] %>%
  pivot_longer(cols = c(2:81)) %>%
    mutate(time_parse = gsub("3 months\nCQ", "", name),
           year = as.numeric(word(time_parse, -1)),
           quarter = as.numeric(word(time_parse, 1)),
           
           value = as.numeric(value),
           value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
           qoq_change = (value - value_prev)/ value) %>%
  select(year, quarter, qoq_change)
  
ppe <- sp500bs[11,] %>%
  pivot_longer(cols = c(2:81)) %>%
  mutate(time_parse = gsub("3 months\nCQ", "", name),
         year = as.numeric(word(time_parse, -1)),
         quarter = as.numeric(word(time_parse, 1)),
         
         value = as.numeric(value),
         value_prev = as.numeric(c(NA, value[2:length(value) - 1])),
         qoq_change = (value - value_prev)/ value) %>%
  select(year, quarter, qoq_change)

# ---- dv ----

dv <- read_csv("target_variable.csv")
colnames(dv) <- dv[1,]
dv <- dv[-c(1:2),]

dv <- dv %>%
  mutate(year = as.numeric(Year),
         quarter = as.numeric(gsub("Q", "", Quarter)),
         dv = as.numeric(gsub("%", "", `Annualized EBITDA Per Share Growth minus 10-Year`))/100) %>%
  select(year, quarter, dv)

# ---- join ----

df <- dv %>%
  left_join(bus_conf %>% rename(bus_conf = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(con_cred %>% rename(con_cred = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(cpi %>% rename(cpi = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(cons_exp %>% rename(con_exp = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(ppi %>% rename(ppi = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(vix %>% rename(vix = "qoq_change"),
            by = c("year", "quarter")) %>%
  left_join(inv %>% rename(inv = "qoq_change"), 
            by = c("year", "quarter")) %>%
  left_join(ppe %>% rename(ppe = "qoq_change"),
            by = c("year", "quarter"))

write_csv(df, "out/data.csv")

summary(lm(dv ~ bus_conf + con_cred + cpi + con_exp + ppi + vix + inv + ppe, df))
