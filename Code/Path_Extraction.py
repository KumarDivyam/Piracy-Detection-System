import os

dir_path = r"Training Data\Clean Videos"
file_paths = []

# Iterate over all files in the directory and its subdirectories
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # Get the file path and add it to the list
        file_path = os.path.join(root, file)
        file_paths.append(file_path)

# Write the file paths to a text file
with open("file_paths.txt", "w") as f:
    for path in file_paths:
        f.write(path + "\n")
