# Read the original video
original_cap = cv2.VideoCapture('original_video.mp4')

# Extract the color histogram of the original video
original_color_hist = np.zeros((256, 1, 3), dtype=np.float32)
while True:
    ret, frame = original_cap.read()
    if not ret:
        break
    original_color_hist += cv2.calcHist([frame], [0, 1, 2], None, [256, 1, 3], [0, 256, 0, 256, 0, 256])
original_color_hist /= np.sum(original_color_hist)
original_cap.release()

# Read the suspected pirated copy
pirated_cap = cv2.VideoCapture('pirated_video.mp4')

# Extract the color histogram of the suspected pirated copy
pirated_color_hist = np.zeros((256, 1, 3), dtype=np.float32)
while True:
    ret, frame = pirated_cap.read()
    if not ret:
        break
    pirated_color_hist += cv2.calcHist([frame], [0, 1, 2], None, [256, 1, 3], [0, 256, 0, 256, 0, 256])
pirated_color_hist /= np.sum(pirated_color_hist)
pirated_cap.release()

# Compute the correlation coefficient between the two histograms
corr = cv2.compareHist(original_color_hist, pirated_color_hist, cv2.HISTCMP_CORREL)

if corr > 0.9:
    print('The suspected pirated copy is highly similar to the original video.')
else:
    print('The suspected pirated copy is not similar to the original video.')
##################################################################################################################################
##################################################################################################################################
import cv2
import numpy as np

def extract_histograms(video_file):
    cap = cv2.VideoCapture(video_file)

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

def compare_histograms(hist1, hist2):
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return intersection

def detect_piracy(video_file, pirated_color_hist, pirated_edge_hist):
    video_color_hist, video_edge_hist = extract_histograms(video_file)

    color_similarity = compare_histograms(pirated_color_hist, video_color_hist)
    edge_similarity = compare_histograms(pirated_edge_hist, video_edge_hist)

    if color_similarity > 0.9 and edge_similarity > 0.9:
        print
##########################################################################################
import cv2
import numpy as np

def extract_histograms(video_file):
    cap = cv2.VideoCapture(video_file)

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

def compare_histograms(hist1, hist2):
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return intersection

def detect_piracy(video_file, pirated_color_hist, pirated_edge_hist):
    video_color_hist, video_edge_hist = extract_histograms(video_file)

    color_similarity = compare_histograms(pirated_color_hist, video_color_hist)
    edge_similarity = compare_histograms(pirated_edge_hist, video_edge_hist)

    if color_similarity > 0.9 and edge_similarity > 0.9:
        print
###############################################################################################################
#Integrated code:
import cv2
import numpy as np

def extract_histograms(video_file):
    cap = cv2.VideoCapture(video_file)

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

def compare_histograms(hist1, hist2):
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return intersection

def detect_piracy(video_file, pirated_color_hist, pirated_edge_hist):
    video_color_hist, video_edge_hist = extract_histograms(video_file)

    color_similarity = compare_histograms(pirated_color_hist, video_color_hist)
    edge_similarity = compare_histograms(pirated_edge_hist, video_edge_hist)

    if color_similarity > 0.9 and edge_similarity > 0.9:
        print('The video may be a pirated copy')
    else:
        print('The video is not a pirated copy')

# Load the histograms of a pirated video from the database
pirated_color_hist = np.load('pirated_color_hist.npy')
pirated_edge_hist = np.load('pirated_edge_hist.npy')

# Detect piracy in a video to be checked
detect_piracy('video_to_check.mp4', pirated_color_hist, pirated_edge_hist)
##################################################################################
#Extracting color and edge histograms from OTT and video sharing pltf.
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
