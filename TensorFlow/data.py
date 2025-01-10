import os
import numpy as np
import cv2
from glob import glob
import json
from sklearn.model_selection import train_test_split

def get_data(dataset_path, train=True):
    if train == True:
        data_type = "train"
    else:
        data_type = "test"

    with open(os.path.join(dataset_path, data_type, "questions.json"), "r") as file:
        data = json.load(file)

    questions, answers, image_paths = [], [], []
    for q, a, p in data:
        questions.append(q)
        answers.append(a)
        image_paths.append(os.path.join(dataset_path, data_type, "images", f"{p}.png"))

    return questions, answers, image_paths

def get_answers_labels(dataset_path):
    with open(os.path.join(dataset_path, "answers.txt"), "r") as file:
        data = file.read().strip().split("\n")
        return data

def main():
    dataset_path = "data"
    trainQ, trainA, trainI = get_data(dataset_path, train=True)
    testQ, testA, testI = get_data(dataset_path, train=False)
    unique_answers = get_answers_labels(dataset_path)

    """ Split the data into training and validation """
    trainQ, valQ, trainA, valA, trainI, valI = train_test_split(
        trainQ, trainA, trainI, test_size=0.2, random_state=42
    )
    
    print(f"Train -> Questions: {len(trainQ)} - Answers: {len(trainA)} - Images: {len(trainI)}")
    print(f"Valid -> Questions: {len(valQ)} - Answers: {len(valA)} - Images: {len(valI)}")
    print(f"Test -> Questions: {len(testQ)} - Answers: {len(testA)} - Images: {len(testI)}")




if __name__ == "__main__":
    main()
