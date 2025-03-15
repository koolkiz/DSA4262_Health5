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
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load the dataset
df = pd.read_csv("..\\data\\clinical_data.csv")

df.fillna("Unknown")

# Mapping for ordinal encoding
ordinal_mappings = {
    "Breast density": ["A", "B", "C", "D"],

    "Pathology Classification/ Follow up": ["Normal", "Benign", "Malignant"],

    # FACTOR LEVELS
    "Mass density" : ["Equal", "Equal with overlying macrocalcification", 
                      "High - Equal", "High", 
                      "High with overlying microcalcification", "IMLN"],

    "Mass shape": ["Oval", "Round - Oval", "Irregular - Oval", 
                    "Irregular - Round", "Irregular"],

    "Mass margin": ["Circumscribed", "Circumscribed - Obscured", "Partially obscured",  
                    "Obscured", "Indistinct", "Indistinct - Circumscribed",  
                    "Lobulated - Partially Obscured", "Microlobulated - Circumscribed",  
                    "Microlobulated", "Speculated - Circumscribed",  
                    "Speculated", "Speculated - Ulcerating"],

    "Enhancement pattern": ["Non Enhancement", "Other", "Homogenous",  
                        "Focus enhancement", "Focal enhancement",  
                        "Stippled", "Enhancing mass", "Heterogenous",  
                        "Irregular rim", "Rim Enhancement", "IMLN"],

    "Mass enhancement shape": ["Oval", "Round - Oval", "Round", "Irregular"],

    "Mass enhancement margin": ["Circumscribed", "Lobulated circumscribed", 
                                "Irregular", "Speculated"],
}

# Apply ordinal encoding
for col, categories in ordinal_mappings.items():
    if col in df.columns:
        encoder = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=np.nan)
        df[col] = encoder.fit_transform(df[[col]])

# Apply one-hot encoding for the specified columns
one_hot_columns = [
    "Side", "Type", "View", "Single/Multiple (Mass)",
    "Single/Multiple (Mass enhancement)"
]

# Initializing OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

for col in one_hot_columns:
    if col in df.columns:
        transformed = ohe.fit_transform(df[[col]])
        ohe_df = pd.DataFrame(transformed, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
        df = df.drop(columns=[col]).join(ohe_df)

# Save the encoded dataset
df.to_csv("../data/clinical_data_encoded_ver2.csv", index=False)

# Display first few rows
# print(df.head())