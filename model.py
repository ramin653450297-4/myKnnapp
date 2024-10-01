import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv('hypertension_data.csv')

# Define the features and target variable
X = data[['Age', 'Weight', 'Height', 'Systolic_BP', 'Diastolic_BP','Heart_Rate']]  # Features
y = data['Hypertension']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
joblib.dump(model, 'decision_tree_model.pkl')

print("Model trained and saved as 'decision_tree_model.pkl'")
