#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load data from CSV
df = pd.read_csv('data.csv')

# Preprocessing
X = df[['Latitude', 'Longitude']].values
y = df['Label'].values

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels to numeric representation
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create KDTree using the training data
kdtree = KDTree(X_train)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define the scaler and fit it with the training data
scaler = StandardScaler(with_mean=False, with_std=False)
scaler.fit(df[['Latitude', 'Longitude']])

# Route for predicting nearest institutions
@app.route('/predict', methods=['POST'])
def predict_nearest_institutions():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    label = data['label']

    location = scaler.transform([[latitude, longitude]])
    kdtree = KDTree(scaler.transform(df[['Latitude', 'Longitude']]))
    _, indices = kdtree.query(location, k=5)
    nearest_institutions = df.loc[indices[0], :]
    nearest_institutions = nearest_institutions[nearest_institutions['Label'] == label]

    predictions = []
    for _, row in nearest_institutions.iterrows():
        institution_data = row[['ID', 'KETERANGAN', 'TLP', 'Latitude', 'Longitude', 'Label']].to_dict()
        predictions.append(institution_data)

    if nearest_institutions.empty:
        all_institutions = df[df['Label'] == label]
        if all_institutions.empty:
            return jsonify({'message': 'No institutions found for the specified type.'})
        else:
            for _, row in all_institutions.iterrows():
                institution_data = row[['KETERANGAN', 'TLP', 'Latitude', 'Longitude', 'Label']].to_dict()
                predictions.append(institution_data)

    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run()


# In[ ]:




