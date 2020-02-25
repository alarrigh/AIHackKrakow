import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
import warnings
import os, argparse

from azureml.core import Run

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point")
parser.add_argument("--n-estimators", type=int, dest="n_estimators", default=500)
parser.add_argument("--max-depth", type=int, dest="max_depth", default=4)
parser.add_argument("--min-samples-split", type=int, dest="min_samples_split", default=2)
parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.01)

args = parser.parse_args()

data_path = os.path.join(args.data_folder, "data/ames.csv")

df_housing = pd.read_csv(data_path)

df_housing.drop("Order", axis = 1, inplace = True)
df_housing.drop("PID", axis = 1, inplace = True)

fill_none = ["Pool.QC", "Misc.Feature", "Alley", "Fence", "Fireplace.Qu", "Garage.Type", "Garage.Finish", "Garage.Qual", "Garage.Cond",
            "Bsmt.Exposure", "Bsmt.Cond", "Bsmt.Qual", "Mas.Vnr.Type"]
for var in fill_none:
    df_housing[var] = df_housing[var].fillna("None")
    
fill_zero = ["Garage.Yr.Blt", "BsmtFin.Type.2", "BsmtFin.Type.1", "Bsmt.Half.Bath", "Bsmt.Full.Bath", "Total.Bsmt.SF",
             "Bsmt.Unf.SF", "BsmtFin.SF.1", "BsmtFin.SF.2", "Garage.Area", "Garage.Cars", "Mas.Vnr.Area"]
for var in fill_zero:
    df_housing[var] = df_housing[var].fillna(0)

df_housing["Lot.Frontage"] = df_housing.groupby("Neighborhood")["Lot.Frontage"].transform(lambda x: x.fillna(x.median()))

df_housing['Electrical'] = df_housing['Electrical'].fillna(df_housing['Electrical'].mode()[0])

df_housing = df_housing.dropna()

response_var = ["SalePrice"]

numeric_vars = ["Lot.Frontage", "Lot.Area", "Mas.Vnr.Area", "BsmtFin.SF.1", "BsmtFin.SF.2", "Bsmt.Unf.SF", "Total.Bsmt.SF",
                "X1st.Flr.SF", "X2nd.Flr.SF", "Low.Qual.Fin.SF", "Gr.Liv.Area", "Garage.Area", "Wood.Deck.SF",
                "Open.Porch.SF", "Enclosed.Porch", "X3Ssn.Porch", "Screen.Porch", "Pool.Area", "Misc.Val"]

categorical_vars = [v for v in df_housing.columns if v not in numeric_vars + response_var]

def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
categorical_vars_E = []
for q in categorical_vars:  
    encode(df_housing, q)
    categorical_vars_E.append(q+'_E')
    
X_train, X_test, y_train, y_test = train_test_split(df_housing[numeric_vars + categorical_vars_E], df_housing[response_var],
                                                    test_size=0.4, random_state=0)

scoring = {'R2': make_scorer(r2_score), 'MAE': make_scorer(mean_absolute_error)}

n_estimators = args.n_estimators
max_depth = args.max_depth
min_samples_split = args.min_samples_split
learning_rate = args.learning_rate

clf = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, learning_rate=learning_rate, loss="ls")
scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=10, n_jobs=-1)

run = Run.get_context()

run.log("train_MAE", scores["train_MAE"].mean())
run.log("train_R2", scores["train_R2"].mean())
run.log("val_MAE", scores["test_MAE"].mean())
run.log("val_R2E", scores["test_R2"].mean())

clf_fitted = clf.fit(X_train, y_train)
y_test_predicted = clf_fitted.predict(X_test)

run.log("test_MAE", mean_absolute_error(y_test, y_test_predicted))
run.log("test_R2", r2_score(y_test, y_test_predicted))

model_file_name = "gbr_" + str(n_estimators) + "_" + str(max_depth) + "_" + str(min_samples_split) + "_" + str(learning_rate) + ".pkl"
joblib.dump(value=clf_fitted, filename="./outputs/" + model_file_name)
