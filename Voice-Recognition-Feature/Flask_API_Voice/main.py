from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

app = Flask(_name_)

# Load the models
institution_model = load_model('institution_model.h5')
sound_model = load_model('sound_model.h5')

# Load the datasets
df_institution = pd.read_csv('data.csv')
df_sound = pd.read_csv('Train set Data Capstone Bangkit.csv')

# Preprocess the data
scaler = StandardScaler()
scaler.fit(df_institution[['Latitude', 'Longitude']])
kdtree = KDTree(scaler.transform(df_institution[['Latitude', 'Longitude']]))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_sound['Kalimat'].astype(str).tolist())
word_index = tokenizer.word_index
max_sequence_length = max([len(seq) for seq in tokenizer.texts_to_sequences(df_sound['Kalimat'].astype(str).tolist())])

labels_dict = {'Polisi': 0, 'RS': 1, 'Damkar': 2, 'PMI': 3, 'Basarnas': 4}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    latitude = data['Latitude']
    longitude = data['Longitude']
    new_sentences = [data['bahaya anda']]

    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    sound_prediction = sound_model.predict(new_padded_sequences)
    sound_label = list(labels_dict.keys())[np.argmax(sound_prediction)]

    location = scaler.transform([[latitude, longitude]])
    _, indices = kdtree.query(location, k=5)
    nearest_institutions = df_institution.loc[indices[0], :]
    nearest_institutions = nearest_institutions[nearest_institutions['Label'] == sound_label]

    response_data = []
    for _, row in nearest_institutions.iterrows():
        institution_data = {
            'ID': row['ID'],
            'KETERANGAN': row['KETERANGAN'],
            'TLP': row['TLP'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Label': row['Label']
        }
        response_data.append(institution_data)
        break

    if nearest_institutions.empty:
        all_institutions = df_institution[df_institution['Label'] == sound_label]
        if all_institutions.empty:
            response_data = "No institutions found for the specified type."
        else:
            all_institutions_data = []
            for _, row in all_institutions.iterrows():
                institution_data = {
                    'KETERANGAN': row['KETERANGAN'],
                    'TLP': row['TLP'],
                    'Latitude': row['Latitude'],
                    'Longitude': row['Longitude'],
                    'Label': row['Label']
                }
                all_institutions_data.append(institution_data)
                break
            response_data = all_institutions_data
            

    return jsonify(response_data)

if _name_ == '_main_':
    app.run()