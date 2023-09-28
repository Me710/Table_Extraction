import json
import shutil
import os

# Load the JSON data from the file
json_file = '../Output Json/train.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# Create a sub-folder named 'test' if it doesn't exist
test_folder = 'train'
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Extract the filenames from the JSON data and move the files to the 'test' folder
for image in data['images']:
    filename = image['file_name']
    source_path = filename
    dest_path = os.path.join(test_folder, os.path.basename(filename))

    try:
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} to {dest_path}")
    except FileNotFoundError:
        print(f"File not found: {filename}")

print("File extraction and moving completed.")
