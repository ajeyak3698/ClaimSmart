import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv('data/claims.csv')

df['claim_type'] = LabelEncoder().fit_transform(df['claim_type'])
X = df[['claim_amount', 'claim_type']]
y = df['claim_amount'].apply(lambda x: 0 if x < 1000 else (1 if x < 10000 else 2))  # 0: Low, 1: Medium, 2: High complexity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)

joblib.dump(clf, 'models/claim_complexity_model.pkl')
print(classification_report(y_test, clf.predict(X_test)))
