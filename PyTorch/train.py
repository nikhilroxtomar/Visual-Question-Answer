import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data import get_data, get_answers_labels
from model import VQAModel

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class BOWTokenizer:
    """ A simple Bag of Words tokenizer. """
    def __init__(self):
        self.word_index = {}

    def fit_on_texts(self, texts):
        unique_words = set(word for text in texts for word in text.split())
        self.word_index = {word: idx + 1 for idx, word in enumerate(unique_words)}

    def texts_to_matrix(self, texts):
        vocab_size = len(self.word_index) + 1
        result = np.zeros((len(texts), vocab_size), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.split():
                if word in self.word_index:
                    result[i, self.word_index[word]] += 1
        return result

class VQADataset(Dataset):
    def __init__(self, questions, answers, image_paths, tokenizer, labels, image_h, image_w):
        self.questions = questions
        self.answers = answers
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.labels = labels
        self.image_h = image_h
        self.image_w = image_w

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        image_path = self.image_paths[idx]

        # Process question
        question = self.tokenizer.texts_to_matrix([question])[0]

        # Process answer
        answer_idx = self.labels.index(answer)
        answer_one_hot = np.zeros(len(self.labels), dtype=np.float32)
        answer_one_hot[answer_idx] = 1.0

        # Process image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_w, self.image_h))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format

        return torch.tensor(image, dtype=torch.float32), torch.tensor(question, dtype=torch.float32), torch.tensor(answer_one_hot, dtype=torch.float32)

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs, save_path):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == answers.argmax(dim=1)).sum().item()

        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, questions, answers in valid_loader:
                images, questions, answers = images.to(device), questions.to(device), answers.to(device)

                outputs = model(images, questions)
                loss = criterion(outputs, answers)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == answers.argmax(dim=1)).sum().item()

        val_acc = val_correct / len(valid_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss/len(valid_loader):.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Model saved!")

def main():
    """ Seeding """
    torch.manual_seed(42)

    """ Hyperparameters """
    image_shape = (64, 64)
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-4
    save_path = "files/model.pth"

    """ Directory for storing files """
    create_dir("files")

    """ Dataset Preparation """
    dataset_path = "data"
    trainQ, trainA, trainI = get_data(dataset_path, train=True)
    testQ, testA, testI = get_data(dataset_path, train=False)
    unique_answers = get_answers_labels(dataset_path)

    trainQ, valQ, trainA, valA, trainI, valI = train_test_split(
        trainQ, trainA, trainI, test_size=0.2, random_state=42
    )

    print(f"Train -> Questions: {len(trainQ)} - Answers: {len(trainA)} - Images: {len(trainI)}")
    print(f"Valid -> Questions: {len(valQ)} - Answers: {len(valA)} - Images: {len(valI)}")
    print(f"Test -> Questions: {len(testQ)} - Answers: {len(testA)} - Images: {len(testI)}")

    """ Tokenizer """
    tokenizer = BOWTokenizer()
    tokenizer.fit_on_texts(trainQ + valQ)
    vocab_size = len(tokenizer.word_index) + 1

    """ Datasets and DataLoaders """
    train_dataset = VQADataset(trainQ, trainA, trainI, tokenizer, unique_answers, image_h=image_shape[0], image_w=image_shape[1])
    val_dataset = VQADataset(valQ, valA, valI, tokenizer, unique_answers, image_h=image_shape[0], image_w=image_shape[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    """ Model, Optimizer, Loss """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel(image_shape=image_shape, vocab_size=vocab_size, num_answers=len(unique_answers))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    """ Training """
    train_model(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs, save_path)

if __name__ == "__main__":
    main()
