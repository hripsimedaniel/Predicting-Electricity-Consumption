### **Project Title: Predicting Electricity Consumption with Regression Models**

**Project Description:** Prediction of electricity consumption with Regression models. The following Regression models are applied in this project: RandomForestRegressor, DecisionTreeRegressor, ExtraTreeRegressor, and LinearRegression. During the preprocessing stage, the dimensionality reduction of features is conducted with Truncated SVD. The prediction is based on historical data - the Residential Energy Consumption Survey.

**Results:** The best result is achieved with LinearRegression - Mean Absolute Error: 4012.697, Mean Squared Error: 30283301.804, Root Mean Squared Error: 5503.026  
The model's performance can be improved if feature selection analysis is conducted more thoroughly, and NaN/missing values are better handled (these values were coded as '-2' in this dataset ). 

**Dependencies:** Python

**Algorithm:** RandomForestRegressor, DecisionTreeRegressor, ExtraTreeRegressor, and LinearRegression

**Dataset:** The Residential Energy Consumption Survey is a national sample survey that collects energy-related data for housing units occupied as a primary residence and the households that live in them. Data were collected from 12,083 households selected at random using a complex multistage, area-probability sample design. The sample represents 113.6 million U.S. households, the Census Bureau’s statistical estimate for all occupied housing units in 2009 derived from their American Community Survey (ACS). You can access the dataset [here](https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata).
