##7PAM2000 Applied Data Science 1
###Assignment 3: Clustering and Fitting
------------------------------------------------------------------------------
Mohit Agarwal (Student ID-22031257)

Exploring the impact of Economical Growth on Climate, using indicators
from World-Bank.

https://data.worldbank.org/indicator

The chosen indicators are:
1. GDP (current US$)
    -> https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
2. Forest area (% of land area) 
    -> https://data.worldbank.org/indicator/AG.LND.FRST.ZS
3. Access to electricity (% of population)
    -> https://data.worldbank.org/indicator/EG.ELC.ACCS.ZS
4. CO2 emissions (metric tons per capita)
    -> https://data.worldbank.org/indicator/EN.ATM.CO2E.PC
5. Manufacturing, value added (% of GDP)
    -> https://data.worldbank.org/indicator/NV.IND.MANF.ZS
6. Total greenhouse gas emissions (kt of CO2 equivalent)
    -> https://data.worldbank.org/indicator/EN.ATM.GHGT.KT.CE

The income-group, regional and country wise cluster and fitting is
constructed using kmeans method for cluster and ploynomial of order 2
is taken as function for fitting and forecasting for next 15 years.

Provided error python script file was used to calculate the confidence range.

The program will create three type of graphs, namely:-
1. The silhouette score graph for clustering.
2. The cluster plots.
3. The fitting plots with forecast for next 15 years.

Note: 
No plots/grahs are shown but all are saved directly in the working directory.
