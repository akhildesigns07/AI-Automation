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

housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()

numerical_features = list(housing.select_dtypes(include=['float64', 'int64']).columns)
categorical_features = ['ocean_proximity','income_cat'] 

numerical_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('std_scaler', StandardScaler()),
 ])

housing_num_attributes_only = housing.drop(categorical_features, axis =1)
housing_prepared_num = numerical_pipeline.fit_transform(housing_num_attributes_only)
numerical_features_dataFrame = pd.DataFrame(housing_prepared_num, columns=numerical_features)

housing_cat = housing[["ocean_proximity"]]
encode = OneHotEncoder()
housing_cat_encoded = encode.fit_transform(housing_cat)

full_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

housing_prepared = full_pipeline.fit_transform(housing)
columns = (numerical_features + list(full_pipeline.named_transformers_['cat'].get_feature_names_out()))
full_pipeline_dataframe = pd.DataFrame(housing_prepared, columns=columns)
full_pipeline_dataframe

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.tree import DecisionTreeRegressor

dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
housing_predictions_dec = dec_reg.predict(housing_prepared)
dec_mse = mean_squared_error(housing_labels, housing_predictions_dec)
dec_rmse = np.sqrt(dec_mse)
dec_rmse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(dec_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
dec_rmse_scores = np.sqrt(-scores)
dec_rmse_scores
scores_tree = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score


# svm_reg = SVR(kernel="linear")
# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# #SVR Evaluation
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)


# scores = cross_val_score(svm_reg, housing_prepared, housing_labels,
# scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)

# from sklearn.model_selection import GridSearchCV

# param_grid = [
#     {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},  # Linear kernel with varying regularization parameter C
#     {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']},  # RBF kernel with different gamma values
#     {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3], 'gamma': ['scale', 'auto']},  # Polynomial kernel
# ]

# # Perform GridSearchCV on SVR model
# grid_search = GridSearchCV(SVR(), param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)

# grid_search.fit(housing_prepared, housing_labels)

# # View the best parameters found by GridSearchCV
# print("Best parameters found:", grid_search.best_params_)





