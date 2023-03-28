import cv2
import numpy as np
from matplotlib import pyplot as plt
import urllib.request
import json

# Read the video from the local file
cap = cv2.VideoCapture("video.mp4")

# Initialize the color and edge histograms
color_hist = np.zeros((256, 3))
edge_hist = np.zeros(256)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Update the color histogram
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([frame], [i], None, [256], [0, 256])
        color_hist[:, i] += histr.ravel()

    # Update the edge histogram
    edges = cv2.Canny(frame, 50, 150)
    edge_hist += np.histogram(edges.ravel(), 256, [0, 256])[0]

# Normalize the histograms
color_hist /= color_hist.sum()
edge_hist /= edge_hist.sum()
    
# Plot the color histogram
for i, col in enumerate(color):
    plt.plot(color_hist[:, i], color=col)
    plt.xlim([0, 256])
plt.title('Color Histogram')
plt.show()

# Plot the edge histogram
plt.plot(edge_hist)
plt.xlim([0, 256])
plt.title('Edge Histogram')
plt.show()

# Release the video capture object
cap.release()

histograms = {'color_hist': color_hist.tolist(), 'edge_hist': edge_hist.tolist()}
with open('histograms.json', 'w') as f:
    json.dump(histograms, f)