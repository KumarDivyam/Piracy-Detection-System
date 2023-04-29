import cv2
import numpy as np #csv
import os
import pandas as pd#csv
from sklearn.ensemble import IsolationForest
import joblib
from tqdm import tqdm

n_estimators = 100
model_file = 'Model_training_Blur.pkl'
piracy_threshold = -0.35 

model = joblib.load(model_file)
video_folder = 'Training Data\Test'
result_file = 'Final.csv'

videos = os.listdir(video_folder)
results = []

for video_name in tqdm(videos):
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    anomalies = []

    with tqdm(total=total_frames) as pbar:
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
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                laplacian_var_arr = np.array([laplacian_var])
                test_X = np.concatenate(
                    (np.array(color_hist).flatten(), np.array(edge_hist).flatten(), laplacian_var_arr), axis=0)
                anomaly_score = model.score_samples(test_X.reshape(1, -1))[0]
                anomalies.append(anomaly_score)
                pbar.update(1)
            else:
                break

    cap.release()

    avg_anomaly = np.mean(anomalies)
    results.append({'Video Name': video_name, 'Anomaly Score': avg_anomaly, 'Piracy': avg_anomaly > piracy_threshold})

result_df = pd.DataFrame(results)
result_df.to_csv(result_file, index=False)
