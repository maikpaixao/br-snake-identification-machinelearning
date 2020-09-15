import numpy as np
import pandas as pd
from preprocess import Preprocess
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

path = 'benchmark/dataset/'
preprocess = Preprocess()

def main():
        image_dataset = preprocess.loadfile(container_path = path)
        x_train, x_test, y_train, y_test = preprocess.splitData(image_dataset)

        print(x_train[1])

        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print('Accuracy: ', accuracy)

if __name__ == "__main__":
        main()