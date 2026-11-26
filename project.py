
print('Starting program')

print('Importing libraries')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from lxml import etree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC

SIZE = (64, 64)
DISTRIBUTION = False # Display graph of speed limit distributions and exit

# display classification report and confusion matrix
def display_results(y_test, predicted):
    print()
    print(metrics.classification_report(y_test, predicted, zero_division=0.0))
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    plt.show()

print('Loading dataset')

X = []
y = []
image_ids = []
X_limit = []
limit_indices = []
limit_image_ids = []
y_limit = []
classes = {}

for num in range(877):
    # Concatenate paths
    img_path = './road-sign-detection/images/road' + str(num) + '.png'
    xml_path = './road-sign-detection/annotations/road' + str(num) + '.xml'
    
    # Load files
    img = Image.open(img_path).convert("RGB")
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # Iterate through image road signs
    for obj in root.findall('./object'):
        # Add label
        name = obj.findtext('name')
        y.append(name)
        image_ids.append(num)

        # crop the image
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))

        # Convert image to ndarray and add to dataset
        sign = img.crop((xmin, ymin, xmax, ymax))
        sign = ImageOps.contain(sign, SIZE, method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", SIZE, (0, 0, 0))
        x_pos = (SIZE[0] - sign.width) // 2
        y_pos = (SIZE[1] - sign.height) // 2
        canvas.paste(sign, (x_pos, y_pos))
        X.append(np.asarray(canvas, dtype=np.float32) / 255.0)

        # Speed limit case
        if name == 'speedlimit':
            speed = obj.findtext('speed')
            y_limit.append(int(speed))
            limit_indices.append(len(X) - 1)
            limit_image_ids.append(num)
            
            # Increment distribution
            if DISTRIBUTION:
                if not speed in classes:
                    classes[speed] = 0
                classes[speed] += 1

# Display the distribution graph
if DISTRIBUTION:
    plt.bar(classes.keys(), classes.values())
    plt.xlabel("speed limit")
    plt.ylabel("distribution")
    plt.show()
    exit()

print('Preprocessing X and y')
X = np.array(X)
X = X.reshape(len(X), -1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y)

print('Spliting X and y')
# Split road signs by keeping original images in the same part
train_ids, test_ids = train_test_split(np.unique(image_ids), test_size=0.2)
train_mask = np.isin(image_ids, train_ids)
test_mask = np.isin(image_ids, test_ids)
X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print('Training X and y Model')
model = SVC().fit(X_train, y_train)

print('Testing X and y Model')
y_pred = model.predict(X_test)

print('Preprocessing X_limit and y_limit')
X_limit = X[limit_indices]
limit_indices = np.array(limit_indices)
y_limit = np.array(y_limit, dtype=np.int64)
limit_image_ids = np.array(limit_image_ids)

print('Spliting X_limit and y_limit')
# Split road signs by keeping original images in the same part
unique_limit_ids = np.unique(limit_image_ids)
train_limit_ids, test_limit_ids = train_test_split(unique_limit_ids, test_size=0.2)
train_limit_mask = np.isin(limit_image_ids, train_limit_ids)
test_limit_mask = np.isin(limit_image_ids, test_limit_ids)
X_limit_train = X_limit[train_limit_mask]
X_limit_test = X_limit[test_limit_mask]
y_limit_train = y_limit[train_limit_mask]
y_limit_test = y_limit[test_limit_mask]

print('Training X_limit and y_limit Model')
model_limit = LinearSVC(penalty="l1").fit(X_limit_train, y_limit_train)

print('Testing X_limit and y_limit Model')
y_limit_pred = model_limit.predict(X_limit_test)

print('Displaying X and y Model results')
display_results(y_test, y_pred)

print('Displaying X_limit and y_limit results')
display_results(y_limit_test, y_limit_pred)

print('Closing program')

