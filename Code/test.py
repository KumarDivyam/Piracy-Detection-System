import io
import numpy as np
import cv2
import requests

def extract_histograms(video_url):
    # Read video data from URL using requests library
    video_data = requests.get(video_url).content

    # Write video data to local file
    with open("video.mp4", "wb") as f:
        f.write(video_data)

    # OpenCV requires a file name, so we pass the local file name to the VideoCapture function
    cap = cv2.VideoCapture("video.mp4")

    # Initialize color histogram and edge histogram
    color_hist = np.zeros((180, 256))
    edge_hist = np.zeros((180, 2))

    # Loop through each frame of the video and extract histograms
    while True:
        ret, frame = cap.read()

        # If we've reached the end of the video, break out of the loop
        if not ret:
            break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        median=np.median(gray)
        lower=int(max(0, 0.7 * median))
        upper=int(min(255, 1.3 * median))

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        color_hist += hist

        # Calculate edge histogram
        edges = cv2.Canny(frame, lower, upper)
        hist_red = cv2.calcHist([edges], [0], None, [180], [0, 180])
        hist_blue = cv2.calcHist([edges], [0], None, [180], [0, 180])

        edge_hist[:, 0] += hist_red.flatten()
        edge_hist[:, 1] += hist_blue.flatten()

    # Release VideoCapture object
    cap.release()

    # Normalize histograms
    color_hist = cv2.normalize(color_hist, None).flatten()
    edge_hist = cv2.normalize(edge_hist, None).flatten()

    # Return histograms
    return color_hist, edge_hist


def main():
    # URL of the video to extract histograms from
    video_url = "https://www.appsloveworld.com/wp-content/uploads/2018/10/Sample-Videos-Mp425.mp4"

    # Extract color and edge histograms
    color_hist, edge_hist = extract_histograms(video_url)

    # Print the histograms
    print("Color Histogram:")
    print(color_hist)
    print("Edge Histogram:")
    print(edge_hist)
    
if __name__ == "__main__":
    main()
