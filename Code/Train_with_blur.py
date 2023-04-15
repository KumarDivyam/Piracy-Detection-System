import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

n_estimators = 100

# Open the text file containing the clean video file paths
with open('Training.txt', 'r') as f:
    # Read the file paths one per line and append to the clean_videos list
    clean_videos = [line.strip() for line in f]

color_histograms = []
edge_histograms = []
blurs = []

# Initialize tqdm with the total number of videos
pbar = tqdm(total=len(clean_videos))

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
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            blurs.append(blur)
        else:
            break
    cap.release()
    pbar.update(1)  # Update the progress bar after processing each video

color_histograms = np.array(color_histograms)
edge_histograms = np.array(edge_histograms)
blurs = np.array(blurs)

color_histograms = color_histograms.reshape(color_histograms.shape[0], -1)
edge_histograms = edge_histograms.reshape(edge_histograms.shape[0], -1)
blurs = blurs.reshape(blurs.shape[0], -1)

X = np.concatenate((color_histograms, edge_histograms, blurs), axis=1)

model_file = 'Model_training_Blur.pkl'

model = IsolationForest(n_estimators=n_estimators, contamination='auto')
model.fit(X)

# Save the trained model to a file
joblib.dump(model, model_file)
