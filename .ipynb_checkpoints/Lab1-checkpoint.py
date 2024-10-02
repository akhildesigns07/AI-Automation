import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

#I. Fetch the data to your working directory

housing = pd.read_csv('D:/hv.se/AI/handson-ml2/datasets/housing/housing.csv')

#II. Load the data using ‘pandas’ and take a quick look at it using info() and describe()
housing.info()

housing.describe()

housing.hist(bins=50, figsize=(20,15))
plt.show()

# np.random.seed(42)
# shuffled_indices = np.random.permutation(len(housing))
# test_set_size = int(len(housing) * 0.2)
# test_indices = shuffled_indices[:test_set_size]
# train_indices = shuffled_indices[test_set_size:]
# train_set, test_set = housing.iloc[train_indices], housing.iloc[test_indices]

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# housing["income_cat"].hist()

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

rand_train_set, rand_test_set = train_test_split(housing, test_size=0.2, random_state=42)

strat_dist_train = strat_train_set["income_cat"].value_counts() / len(strat_train_set)
strat_dist_test = strat_test_set["income_cat"].value_counts() / len(strat_test_set)

rand_dist_train = rand_train_set["income_cat"].value_counts() / len(rand_train_set)
rand_dist_test = rand_test_set["income_cat"].value_counts() / len(rand_test_set)

compare_props = pd.DataFrame({
    "Overall": housing["income_cat"].value_counts() / len(housing),
    "Stratified": strat_dist_test,
    "Random": rand_dist_test,
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    set_.drop("ocean_proximity", axis=1, inplace=True)

housing_data = strat_train_set.copy()

housing_data.plot(kind= "scatter", x= "longitude", y= "latitude")

housing_data.plot(kind= "scatter", x= "longitude", y= "latitude", alpha = 0.1)

housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing_data["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()

corr_matrix = housing_data.corr()


attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing_data[attributes], figsize=(12, 8))

housing_data.plot(kind = "scatter" , x = "median_income", y = "median_house_value", alpha = 0.1)


housing_data["rooms_per_household"] = housing_data["total_rooms"]/housing_data["households"]
housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
housing_data["population_per_household"]=housing_data["population"]/housing_data["households"]

corr_matrix_att = housing_data.corr()
median_house_value_corr = corr_matrix_att["median_house_value"].sort_values(ascending = False)


