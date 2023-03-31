import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Parameters
n_estimators = 100
model_file = 'Model_training.pkl'
test_videos_dir = 'Training Data/Pirated'
output_dir = 'Anomaly_Graphs/'

# Load model
model = joblib.load(model_file)

# Loop through video files in the directory
for filename in tqdm(os.listdir(test_videos_dir)):
    if filename.endswith('.mp4'):
        video_path = os.path.join(test_videos_dir, filename)
        cap = cv2.VideoCapture(video_path)
        anomalies = []

        # Process each frame in the video
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Compute color and edge histograms
                color = ('b', 'g', 'r')
                color_hist = []
                for i, col in enumerate(color):
                    histr = cv2.calcHist([frame], [i], None, [512], [0, 256])
                    color_hist.extend(histr)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])

                # Compute anomaly score for the frame
                test_X = np.concatenate(
                    (np.array(color_hist).flatten(), np.array(edge_hist).flatten()), axis=0)
                anomaly_score = model.score_samples(test_X.reshape(1, -1))[0]
                anomalies.append(anomaly_score)
            else:
                break

        cap.release()

        # Save anomaly graph for the video
        output_path = os.path.join(output_dir, filename[:-4] + '_anomaly.png')
        plt.plot(anomalies)
        plt.title('Anomaly scores over time for ' + filename)
        plt.xlabel('Frame number')
        plt.ylabel('Anomaly score')
        plt.savefig(output_path)
        plt.clf()  # Clear figure for the next video
