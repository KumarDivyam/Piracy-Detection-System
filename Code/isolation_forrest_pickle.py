import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib

n_estimators = 100

clean_videos = [
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Avataar.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Avengers.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/John Wick.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Bhramastra.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/CostaRica.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Everest.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Flash.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Interstellar.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/John Wick.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Jurrasic World.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Nick Fury.mp4",
    "D:/UPES/SEM-6/Minor 2/code/Training Data/Clean Videos/Oppenhiemer.mp4"
    
]

color_histograms = []
edge_histograms = []

for video_path in clean_videos:
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            color = ('b', 'g', 'r')
            color_hist = []
            for i, col in enumerate(color):
                histr = cv2.calcHist([frame], [i], None, [512], [0, 256])
                color_hist.extend(histr)
            color_histograms.append(color_hist)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            edge_histograms.append(edge_hist)
        else:
            break
    cap.release()

color_histograms = np.array(color_histograms)
edge_histograms = np.array(edge_histograms)

color_histograms = color_histograms.reshape(color_histograms.shape[0], -1)
edge_histograms = edge_histograms.reshape(edge_histograms.shape[0], -1)

X = np.concatenate((color_histograms, edge_histograms), axis=1)

model_file = 'isolation_forest.pkl'

if os.path.exists(model_file):
    # Load the saved model from a file
    model = joblib.load(model_file)
else:
    # Train the IsolationForest model
    model = IsolationForest(n_estimators=n_estimators, contamination='auto')
    model.fit(X)

    # Save the trained model to a file
    joblib.dump(model, model_file)

test_video_path = "D:/UPES/SEM-6/Minor 2/code/Training Data/Pirated/Dementers.mp4"

cap = cv2.VideoCapture(test_video_path)

anomalies = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        color = ('b', 'g', 'r')
        color_hist = []
        for i, col in enumerate(color):
            histr = cv2.calcHist([frame], [i], None, [512], [0, 256])
            color_hist.extend(histr)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        test_X = np.concatenate(
            (np.array(color_hist).flatten(), np.array(edge_hist).flatten()), axis=0)
        anomaly_score = model.score_samples(test_X.reshape(1, -1))[0]
        anomalies.append(anomaly_score)
    else:
        break

cap.release()

plt.plot(anomalies)
plt.title('Anomaly scores over time')
plt.xlabel('Frame number')
plt.ylabel('Anomaly score')
plt.show()