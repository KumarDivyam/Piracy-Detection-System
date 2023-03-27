import cv2
import numpy as np
import requests
import io

def extract_histograms(video_url):
    # Download video from url
    video_data = requests.get(video_url).content

    # Read video using OpenCV
    cap = cv2.VideoCapture(io.BytesIO(video_data))

    color_hist = np.zeros((256, 1, 3), dtype=np.float32)
    edge_hist = np.zeros((16, 1), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        color_hist_temp = cv2.calcHist([frame], [0, 1, 2], None, [256, 1, 3], [0, 256, 0, 256, 0, 256])
        color_hist += color_hist_temp / (gray.shape[0] * gray.shape[1])

        edges = cv2.Canny(gray, 100, 200)
        edge_hist_temp = cv2.calcHist([edges], [0], None, [16], [0, 256])
    edge_hist += edge_hist_temp / (gray.shape[0] * gray.shape[1])

cap.release()

color_hist /= np.sum(color_hist)
edge_hist /= np.sum(edge_hist)

return color_hist, edge_hist