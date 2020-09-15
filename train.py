import numpy as np
import pandas as pd
from preprocess import Preprocess

path = 'benchmark/dataset/'
preprocess = Preprocess()

def main():
        image_dataset = preprocess.loadfile(container_path = path)
        x_train, x_test, y_train, y_test = preprocess.splitData(image_dataset)

if __name__ == "__main__":
        main()