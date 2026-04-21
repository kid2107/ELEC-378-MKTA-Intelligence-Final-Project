import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from skimage.feature import hog   # NEW

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")

RANDOM_STATE = 42


def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} entries, {df['TARGET'].nunique()} classes")
    return df


# grayscale + smaller size
def load_image(filename, size=64):
    img = Image.open(filename).convert("L").resize((size, size))
    return np.array(img)


#  HOG feature extraction
def extract_features(image):
    features = hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features


# uses HOG instead of raw pixels
def imgToFeatureMatrix(imgPath, imgFiles, labelList):
    feature_list = []
    label_arr = []

    for i in range(len(imgFiles)):
        filename = os.path.join(imgPath, imgFiles[i])
        img = load_image(filename)

        features = extract_features(img)

        feature_list.append(features)
        label_arr.append(labelList[i])

        if i % 500 == 0:
            print(f"Processed {i} images")

    return np.array(label_arr), np.array(feature_list)


def train_kernel_svm(X_train, X_val, y_train, y_val, kernel_type):

    le = LabelEncoder()
    y_train = le.fit_transform(np.ravel(y_train))
    y_val = le.transform(np.ravel(y_val))   

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"Training SVM with kernel='{kernel_type}' ...")

    # slightly better hyperparameters
    svm = SVC(kernel=kernel_type, C=10, gamma='scale')

    svm.fit(X_train, y_train)

    val_pred = svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation accuracy ({kernel_type}): {val_acc:.4f}")

    return svm, scaler, le


def makeSubmission(ids, prediction, file='submission.csv'):
    submission = np.column_stack((ids, prediction))
    submission = np.vstack((['ID', 'TARGET'], submission))
    np.savetxt(file, np.asarray(submission), delimiter=',', fmt='%s')


def main():
    print("Loading dataset...")
    df = load_metadata()

    # Train/validation split
    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df["file_name"].values,
        df["TARGET"].values,
        test_size=0.2,
        stratify=df["TARGET"].values,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(X_train_files)}, Validation: {len(X_val_files)}")

    # Prepare test files
    testFiles = []
    testIds = []
    for j in range(1, 1001):
        zeros = '0' * (6 - len(str(j)))
        fname = 'test_' + zeros + str(j) + '.jpg'
        testFiles.append(fname)
        testIds.append('test_' + zeros + str(j))

    # use all data
    print("Extracting training features...")
    trainLabel, trainDataMatrix = imgToFeatureMatrix(TRAIN_IMG_DIR, X_train_files, y_train)

    print("Extracting validation features...")
    valLabel, valDataMatrix = imgToFeatureMatrix(TRAIN_IMG_DIR, X_val_files, y_val)

    print("Extracting test features...")
    # dummy labels for compatibility
    dummy_labels = np.zeros(len(testFiles))
    _, testDataMatrix = imgToFeatureMatrix(TEST_IMG_DIR, testFiles, dummy_labels)

    # Train SVM
    svm, scaler, le = train_kernel_svm(
        trainDataMatrix,
        valDataMatrix,
        trainLabel,
        valLabel,
        "rbf"
    )

    # Predict test set
    yhat = svm.predict(testDataMatrix)
    ypred = le.inverse_transform(yhat)

    makeSubmission(testIds, ypred)

    print("Submission file created!")


if __name__ == "__main__":
    main()