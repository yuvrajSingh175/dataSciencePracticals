import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the CSV file
cell_df = pd.read_csv('cell_samples.csv')

# Select the last 5 rows
cell_df.tail()

# Get the shape, size, and count of the DataFrame
cell_df.shape
cell_df.size
cell_df.count()

# Count the occurrences of each class in the 'Class' column
cell_df['Class'].value_counts()

# Select 200 benign and 200 malignant samples
benign_df = cell_df[cell_df['Class'] == 2].head(200)
malignant_df = cell_df[cell_df['Class'] == 4].head(200)

# Scatter plot of Clump vs. UnifSize with different colors for benign and malignant samples
axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes)
plt.show()

# Clean the data
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

# Select relevant features
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x = np.asarray(feature_df)
y = np.asarray(cell_df['Class'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create an SVM classifier
clf = SVC()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=2)
recall = recall_score(y_test, y_pred, pos_label=2)
f1 = f1_score(y_test, y_pred, pos_label=2)

# Print the metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# New patient features
new_patient_features = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Make a prediction on the new patient data
new_patient_prediction = clf.predict([new_patient_features])

# Print the prediction
print('Predicted cancer class:', new_patient_prediction[0])
