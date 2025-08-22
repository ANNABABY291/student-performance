import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your data (adjust filename)
df = pd.read_csv("students.csv")

X = df.drop("performance", axis=1)  # Replace 'performance' with your target column
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")

