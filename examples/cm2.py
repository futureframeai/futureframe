import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import futureframe as ff

warnings.filterwarnings("ignore")


dataset_name = "tests/data/churn.csv"
target_variable = "Churn"
df = pd.read_csv(dataset_name)
df.info()

X, y = df.drop(columns=[target_variable]), df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##############
# Future Frame
##############
model = ff.models.CM2Classifier()
model.finetune(X_train, y_train)

y_pred = model.predict(X_test)
##############

auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:0.2f}")
