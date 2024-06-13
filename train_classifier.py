import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inicjalizacja Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Ścieżka do katalogu z danymi
DATA_DIR = './data'

data = []
labels = []

# Przetwarzanie obrazów w katalogu
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Sprawdzenie maksymalnej długości list w danych
max_length = max(len(d) for d in data)

# Wyrównanie wszystkich list do tej samej długości
data_padded = np.array([d + [0] * (max_length - len(d)) for d in data])

# Konwersja wyrównanych danych do tablicy NumPy
data_array = np.asarray(data_padded)
labels_array = np.asarray(labels)

# Podział danych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Trening modelu
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predykcja na danych testowych
y_predict = model.predict(x_test)

# Ocena modelu
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Zapisanie modelu do pliku
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Zapisanie danych do pliku pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)
