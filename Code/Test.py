
import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib

n_estimators = 100
model_file = 'isolation_forest.pkl'

model = joblib.load(model_file)
test_video_path = "Training Data\Pirated\Thor Entry.mp4"

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
