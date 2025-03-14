"""
columns + encoding method:
side (2 factors) -> one hot
type (2 factors) -> one hot
breast density (4 levels) -> ordinal
view (2 factors) -> one hot
pathology (3 levels) -> ordinal
single/ multiple (mass) (2 factors + blanks) -> one hot ***
mass density (>5 factors) -> ordinal**
mass shape (>5 factors) -> ordinal**
mass margin (>5 factors) -> ordinal**
single/ multiple (mass enhancement) (2 factors + blanks) -> one hot ***
enhancement patterns (>10 factors)  -> ordinal**
mass enhancement shape (4 factors + blanks)  -> ordinal**
mass enhancement margin (4 factors + blanks) -> ordinal**
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load the dataset
df = pd.read_csv("..\\data\\clinical_data.csv")

df.columns = df.columns.str.lower()

# Mapping for ordinal encoding
ordinal_mappings = {
    "breast density": ["A", "B", "C", "D"],
    "pathology": ["Benign", "High Risk", "Malignant"],

    # EDIT HERE FOR MAPPINGS AFTER FACTOR LEVELS ARE CONFIRMED
    "mass density" : [],
    "mass shape": [],
    "mass margin": [],
    "enhancement patterns": [],
    "mass enhancement shape": [],
    "mass enhancement margin": [],
    ###
}

# Apply ordinal encoding
for col, categories in ordinal_mappings.items():
    if col in df.columns:
        encoder = OrdinalEncoder(categories=[categories])
        df[col] = encoder.fit_transform(df[[col]])

# Apply one-hot encoding for the specified columns
one_hot_columns = [
    "side", "type", "view", 
    "single/multiple (mass)","single/multiple (mass enhancement)"
]

# Initializing OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

for col in one_hot_columns:
    if col in df.columns:
        transformed = ohe.fit_transform(df[[col]])
        ohe_df = pd.DataFrame(transformed, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
        df = df.drop(columns=[col]).join(ohe_df)

# Save the encoded dataset
# df.to_csv("clinical_data_encoded_ver2.csv", index=False)

# Display first few rows
print(df.head())