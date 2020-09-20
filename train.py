import numpy as np
import pandas as pd
from preprocess import Preprocess
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

path = 'benchmark/dataset/'
preprocess = Preprocess()

def main():
        image_dataset = preprocess.loadfile(container_path = path)
        x_train, x_test, y_train, y_test = preprocess.splitData(image_dataset)

        classifiers = [LinearSVC(), LogisticRegression(), GaussianNB(), RandomForestClassifier()]

        for clf in classifiers:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                print('Accuracy: ', accuracy)

if __name__ == "__main__":
        main()