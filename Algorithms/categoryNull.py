import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']], columns=['color', 'size', 'price', 'classlabel'])
print(df)

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1,
}
inv_size_mapping = {v: k for k, v in size_mapping.items()}

df['size'] = df['size'].map(size_mapping)

print(df)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
inv_class_mapping = {v: k for k, v in class_mapping.items()}

# skLearn

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
class_le.inverse_transform(y)

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# skLearn way
ct = ColumnTransformer([("color", OneHotEncoder(), [0])], remainder='passthrough')
new = ct.fit_transform(X)

# pandas way
new2 = pd.get_dummies(df[['price', 'color', 'size']])

print(new2)

