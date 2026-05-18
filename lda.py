#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter
import time

# --------------------------------------------------
# Configuration
# --------------------------------------------------
TRAIN_TXT = "Archive/train.txt"
TEST_TXT = "Archive/test.txt"
#DATA_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
DATA_SIZES = [20000]

scores_knn = []
filenameTXT = "resultadoLDA.txt"
filenamePNG = "learningcurveLDA.png"
name = "LDA"


# --------------------------------------------------
# Load TXT files
# --------------------------------------------------
print("Loading feature files...")

X_train, y_train = load_svmlight_file(TRAIN_TXT)
X_test, y_test = load_svmlight_file(TEST_TXT)

# --------------------------------------------------
# Split labels and features
# X vai ter o numero de records por linha e uma coluna para cada featura
# Y vai ser a resposta para cada entrada, qual a classe
# --------------------------------------------------

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------------------------------------
# k-NN classifier
# --------------------------------------------------

X_train, y_train = shuffle(X_train, y_train, random_state=42)

X_train = X_train.toarray()
X_test = X_test.toarray()

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lda_classifier = LinearDiscriminantAnalysis(
    solver="lsqr",
    shrinkage="auto"
)

# usando esse pq o learning_curve do scikit learn n levava em conta a base de teste, cross validation na base de treinamento
with open(filenameTXT, "a") as f:
    for N in DATA_SIZES:
        X_train_sub = X_train[:N]
        y_train_sub = y_train[:N]

        lda_classifier.fit(X_train_sub, y_train_sub)

        start = time.time()
        y_pred = lda_classifier.predict(X_test)
        train_time = time.time() - start
        print(f"Training time: {train_time:.4f} seconds \n")

        acc = accuracy_score(y_test, y_pred)
        scores_knn.append(acc)

        # print(f"Accuracy: {acc:.4f}")

        # print("\nClassification Report: ", name)
        # print(classification_report(y_test, y_pred))

        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))

        # print("train:", Counter(y_train_sub))
        # print("test:", Counter(y_test))
        # print("pred:", Counter(y_pred))

        # print(f"\nTraining with {N} samples...")
        # f.write(f"\nTraining with {N} samples...")
        # f.write(f"Accuracy: {acc:.4f}\n")

        # f.write(f"Classification Report {name} :\n")
        # f.write(classification_report(y_test, y_pred))
        # f.write("\n")

        # f.write("Confusion Matrix:\n")
        # f.write(str(confusion_matrix(y_test, y_pred)))
        # f.write("\n")

        # f.write("="*50 + "\n")  # separador entre iterações


    #f.write(f"Média training time: {train_time/20:.4f} seconds")
    
# --------------------------------------------------
# Evaluation
# --------------------------------------------------

# Plotar o gráfico
# plt.figure()
# plt.plot(DATA_SIZES, scores_knn, marker='o', label="LDA")

# plt.xlabel("Training set size")
# plt.ylabel("Accuracy")
# plt.title("Learning Curve - LDA")
# plt.legend()
# plt.grid()

# plt.savefig(filenamePNG) 