import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

n_estimators = 100
model_file = 'Model_training_Blur.pkl'

model = joblib.load(model_file)

input_path = "Training Data\Pirated"
output_path = "Anomaly_Graphs_Blur/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

for video_file in os.listdir(input_path):
    video_path = os.path.join(input_path, video_file)
    cap = cv2.VideoCapture(video_path)

    anomalies = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc='Processing Frames of {}'.format(video_file))

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
    pbar.close()

    # save the graph
    plt.plot(anomalies)
    plt.title('Anomaly scores over time for {}'.format(video_file))
    plt.xlabel('Frame number')
    plt.ylabel('Anomaly score')
    output_file = os.path.join(output_path, os.path.splitext(video_file)[0] + '.png')
    plt.savefig(output_file)
    plt.clf()  # clear the plot

print("All videos processed and graphs saved to {}".format(output_path))
