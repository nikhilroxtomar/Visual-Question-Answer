import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data, get_answers_labels
from model import VQAModel
from train import BOWTokenizer

def main():
    """ Seeding """
    torch.manual_seed(42)

    """ Hyperparameters """
    image_shape = (64, 64)
    model_path = os.path.join("files", "model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    tokenizer = BOWTokenizer()
    tokenizer.fit_on_texts(trainQ + valQ)

    testQ = tokenizer.texts_to_matrix(testQ)

    """ Loading Model """
    model = VQAModel(image_shape=image_shape, vocab_size=len(tokenizer.word_index) + 1, num_answers=num_answers)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    true_values, pred_values = [], []
    for question, answer, image_path in tqdm(zip(testQ, testA, testI), total=len(testQ)):
        """ Question """
        question = torch.tensor(question, dtype=torch.float32).unsqueeze(0).to(device)

        """ Answer """
        answer_idx = unique_answers.index(answer)
        true_values.append(answer_idx)

        """ Image """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_shape)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        """ Prediction """
        with torch.no_grad():
            pred = model(image, question)
            pred = torch.argmax(pred, dim=-1).item()
            pred_values.append(pred)

    """ Classification Report """
    report = classification_report(true_values, pred_values, target_names=unique_answers, zero_division=0)
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
