
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data import get_data, get_answers_labels

def main():
    """ Seeding """
    tf.random.set_seed(42)

    """ Hyperparameters """
    image_shape = (64, 64)
    batch_size = 32
    num_epochs = 20

    model_path = os.path.join("files", "model.h5")

    """ Dataset Processing """
    dataset_path = "data"
    trainQ, trainA, trainI = get_data(dataset_path, train=True)
    testQ, testA, testI = get_data(dataset_path, train=False)
    unique_answers = get_answers_labels(dataset_path)
    num_answers = len(unique_answers)

    """ Split the data into training and validation """
    trainQ, valQ, trainA, valA, trainI, valI = train_test_split(
        trainQ, trainA, trainI, test_size=0.2, random_state=42
    )

    print(f"Train -> Questions: {len(trainQ)} - Answers: {len(trainA)} - Images: {len(trainI)}")
    print(f"Valid -> Questions: {len(valQ)} - Answers: {len(valA)} - Images: {len(valI)}")
    print(f"Test -> Questions: {len(testQ)} - Answers: {len(testA)} - Images: {len(testI)}")

    """ Tokenizer: BOW """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(trainQ + valQ)
    vocab_size = len(tokenizer.word_index) + 1

    testQ = tokenizer.texts_to_matrix(testQ)

    """ Loading model """
    model = tf.keras.models.load_model("files/model.h5")

    true_values, pred_values = [], []
    for question, answer, image_path in tqdm(zip(testQ, testA, testI), total=len(testQ)):
        """ Question """
        question = np.expand_dims(question, axis=0)

        """ Answer """
        answer = unique_answers.index(answer)
        true_values.append(answer)

        """ Image """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_shape)
        image = image/255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        """ Prediction """
        pred = model.predict([image, question], verbose=0)[0]
        pred = np.argmax(pred, axis=-1)
        pred_values.append(pred)

    """ Classification Report """
    report = classification_report(true_values, pred_values, target_names=unique_answers)
    print(report)

    """ Confusion Matrix """
    cm = confusion_matrix(true_values, pred_values)


    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_answers, yticklabels=unique_answers)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')


    plt.savefig('files/confusion_matrix_heatmap.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
