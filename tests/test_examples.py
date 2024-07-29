# Import standard libraries
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import Future Frame
import futureframe as ff


def test_kickstart_example():
    # Import data
    dataset_name = "tests/data/churn.csv"
    target_variable = "Churn"
    df = pd.read_csv(dataset_name)

    # Split data
    X, y = df.drop(columns=[target_variable]), df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Fine-tune a pre-trained classifier with Future Frame
    model = ff.models.cm2.CM2Classifier()
    model.finetune(X_train, y_train, max_steps=10)

    # Make predictions with Future Frame
    y_pred = model.predict(X_test)

    # Evaluate your model
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC: {auc:0.2f}")
