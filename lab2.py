import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



housing_raw_data = pd.read_csv('D:/hv.se/AI/handson-ml2/datasets/housing/housing.csv')

housing_raw_data["income_cat"] = pd.cut(housing_raw_data["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(housing_raw_data, test_size=0.2, stratify=housing_raw_data["income_cat"], random_state=42)
housing = strat_train_set.copy()
# housing_with_index = housing.reset_index()
housing_with_index = strat_train_set.copy()
incomplete_rows = housing_with_index[housing_with_index.isnull().any(axis=1)].head()

def cleanData(housing_with_index):

    housing_with_index.drop("median_house_value", axis=1, inplace=True)
    housing_with_index.drop("ocean_proximity", axis=1, inplace=True)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_with_index)
    X = imputer.transform(housing_with_index)
    housing_tr = pd.DataFrame(X, columns=housing_with_index.columns)
    return housing_tr


housing_tr = cleanData(housing_with_index)
# housing_with_index.iloc[86]
# housing_tr.iloc[86]

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
 def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
    self.add_bedrooms_per_room = add_bedrooms_per_room
 def fit(self, X, y=None):
    return self # nothing else to do
 def transform(self, X, y=None):
    print("Transform")
    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
    population_per_household = X[:, population_ix] / X[:, households_ix]
    if self.add_bedrooms_per_room:
        print("add bed_room_per_room")
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                    bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
# housing_num_tr = num_pipeline.fit_transform(housing_with_index)
# original_columns = housing_with_index.columns.tolist()
# new_columns = ['rooms_per_household', 'population_per_household','bed_room_per_room']
# all_columns = original_columns + new_columns

# # Convert transformed data to DataFrame
# convertedDataFrame = pd.DataFrame(housing_num_tr, columns=all_columns)
# convertedDataFrame

num_attribs = list(housing_with_index.columns)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)

cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = cat_encoder.get_feature_names_out(cat_attribs)

original_columns = num_attribs+list(cat_one_hot_attribs)
new_columns = ['rooms_per_household', 'population_per_household','bed_room_per_room']
all_columns = original_columns + new_columns


convertedDataFrame = pd.DataFrame(housing_prepared, columns=all_columns)
# print(convertedDataFrame.head())


housing_labels = strat_train_set["median_house_value"].copy()

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)

def calculateRootMeanSquareError(housing_labels, housing_predictions, predictionModel):
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(predictionModel , " Error : ",lin_rmse)

calculateRootMeanSquareError(housing_labels, housing_predictions, "Linear Regression")

#Tree Reg
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions_tree = tree_reg.predict(housing_prepared)
calculateRootMeanSquareError(housing_labels, housing_predictions_tree, "Decision Tree")

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
          
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions_random = tree_reg.predict(housing_prepared)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)










