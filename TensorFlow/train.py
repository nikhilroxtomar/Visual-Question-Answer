import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from data import get_data, get_answers_labels
from model import build_model

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TFDAtaset:
    def __init__(self, tokenizer, labels, image_h, image_w):
        self.tokenizer = tokenizer
        self.labels = labels
        self.image_h = image_h
        self.image_w = image_w

    def parse(self, question, answer, image_path):
        question = question.decode()
        answer = answer.decode()
        image_path = image_path.decode()

        """ Question """
        question = self.tokenizer.texts_to_matrix([question])
        question = np.array(question[0], dtype=np.float32)

        """ Answer """
        index = self.labels.index(answer)
        answer = [0] * len(self.labels)
        answer[index] = 1
        answer = np.array(answer, dtype=np.float32)

        """ Image """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_w, self.image_h))
        image = image/255.0
        image = image.astype(np.float32)

        return question, answer, image


    def tf_parse(self, question, answer, image_path):
        q, a, i = tf.numpy_function(
            self.parse,
            [question, answer, image_path],
            [tf.float32, tf.float32, tf.float32]
        )
        q.set_shape([len(self.tokenizer.word_index) + 1,])
        a.set_shape([len(self.labels),])
        i.set_shape([self.image_h, self.image_w, 3])

        return (i, q), a

    def tf_dataset(self, questions, answers, image_paths, batch_size=16):
        ds = tf.data.Dataset.from_tensor_slices((questions, answers, image_paths))
        ds = ds.map(self.tf_parse).batch(batch_size).prefetch(10)
        return ds

def main():
    """ Seeding """
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    image_shape = (64, 64, 3)
    batch_size = 32
    num_epochs = 20

    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

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

    """ Dataset pipeline """
    ds = TFDAtaset(tokenizer, unique_answers, image_h=image_shape[0], image_w=image_shape[1])
    train_ds = ds.tf_dataset(trainQ, trainA, trainI, batch_size=batch_size)
    valid_ds = ds.tf_dataset(valQ, valA, valI, batch_size=batch_size)

    """ Model """
    model = build_model(image_shape=image_shape, vocab_size=vocab_size, num_answers=num_answers)
    model.compile(Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
