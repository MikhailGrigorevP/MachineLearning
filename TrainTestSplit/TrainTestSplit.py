import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None,
                      names=['Метка класса', 'Алкоголь', 'Яблочная кислта',
                             'Зола', 'Щелочность золы', 'Магний',
                             'Всего фенолов', 'Флавоноиды',
                             'Нефлавоноидные фенолы', 'Проантоцианидины',
                             'Интенсивность цвета', 'Оттенок',
                             'OD280/0D315 разбавленных вин', 'Пролин'])
print('Метка класса', np.unique(df_wine['Метка класса']))
print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
lr.fit(X_train_std, y_train)
print('Правильность при обучении:', lr.score(X_train_std, y_train))
print('Правильность при испытании:', lr.score(X_test_std, y_test))
