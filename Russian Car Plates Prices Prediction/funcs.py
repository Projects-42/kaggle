from supplemental_russian import REGION_CODES, GOVERNMENT_CODES
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVR
os.chdir(r"C:\Users\User\OneDrive\Документы\Kaggle\kaggle\Russian Car Plates Prices Prediction")
def smape_torch(model, X, y):
    pred = model(torch.tensor(X, dtype=torch.float32))
    val = y.resize_(X.shape[0], 1).detach().clone()
    return (torch.sum(torch.abs(pred - val) / (torch.abs(val) + torch.abs(pred)) * 2) * 100 / len(pred)).item()
def smape(estimator, X, y):
    return 100 / len(y) * sum(abs(y - estimator.predict(X)) / (abs(y) + abs(estimator.predict(X))) * 2)

def convert_dates(data):
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year.astype(str)
    data["month"] = data["date"].dt.month.map(
        {
            1: "january",
            2: "february",
            3: "march",
            4: "april",
            5: "may",
            6: "june",
            7: "july",
            8: "august",
            9: "september",
            10: "october",
            11: "november",
            12: "december"
        }
    )
    data["day_week"] = data["date"].dt.dayofweek.map(
        {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday"
        }
    )
    return data

def reverse_codes(REGION_CODES):
    dict = {}
    for v, k in REGION_CODES.items():
        for key in k:
            dict[key] = v
    return dict
regions = reverse_codes(REGION_CODES)

def extract_government_info(row):
    for ident, info in GOVERNMENT_CODES.items():
        if (row["letters"] == ident[0]) and (ident[1][0] <= int(row["numbers"]) <= ident[1][1]) and (row["region"] == ident[2]):
            return info[0], info[1], info[2], info[3]
        
    return "not governmental", 0, 0, 0

def extract_plate_info(data):
    data[["dept_name", "forb_buy", "adv_road", "significance"]] = data[["letters", "numbers", "region"]].apply(extract_government_info, axis=1, result_type="expand")
    data["region"] = data["region"].map(regions)
    return data

def wrangle(path, data=None):
    if data == None:
        data = pd.read_csv(path)
    convert_dates(data)
    extract_plate(data)
    extract_plate_info(data)
    return data

def extract_plate(data):
    data["letters"] = data["plate"].str[0] + data["plate"].str[4: 6]
    data["numbers"] = data["plate"].str[1: 4]
    data["region"] = data["plate"].str[6:]
    return data

import pickle
def dump(estimator, name):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(estimator, f)

def load(name):
    with open(f"{name}.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()