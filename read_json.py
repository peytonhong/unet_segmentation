import json
import os
import numpy as np

image_path = os.path.join("sample_dataset", "train", "00000AIVFWIP220OK.bmp")
json_path = image_path.replace('.bmp', '.json')

with open(json_path, 'r') as jsonfile:
    data = json.load(jsonfile)

# print(data)
# print(data['shapes'])
shapes = data['shapes']
for shape in shapes:
    # print(shape)
    points = shape['points']
    points = np.array(points).round().astype(int)
    print(points)