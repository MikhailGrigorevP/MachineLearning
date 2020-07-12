import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

print(" # Dataframe:\n", df)
print(" # Num of missed values:\n", df.isnull().sum())
print(" # Delete rows with NaN:\n", df.dropna(axis=0))
print(" # Delete cols with NaN:\n", df.dropna(axis=1))
print(" # Delete rows with all NaN cols:\n", df.dropna(how='all'))
print(" # Delete rows with less 4 :\n", df.dropna(thresh=4))
print(" # Delete rows with special NaN col:\n", df.dropna(subset=['C']))


print("Use mean imputation:")
# strategy: median, most_frequent
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean = imp_mean.fit(df.values)
imputed_date = imp_mean.transform(df.values)
print(imputed_date)





