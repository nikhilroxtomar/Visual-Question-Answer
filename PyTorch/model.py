import torch
import torch.nn as nn

class VQAModel(nn.Module):
    def __init__(self, image_shape, vocab_size, num_answers):
        super(VQAModel, self).__init__()

        # CNN: Image sub-network
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * (image_shape[0] // 4) * (image_shape[1] // 4), 32),
            nn.Tanh()
        )

        # MLP: Question sub-network
        self.question_mlp = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )

        # Combined network
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, num_answers),
            nn.Softmax(dim=-1)
        )

    def forward(self, image, question):
        image_features = self.cnn(image)
        question_features = self.question_mlp(question)
        combined = image_features * question_features
        output = self.fc(combined)
        return output


if __name__ == "__main__":
    image = torch.randn((8, 3, 64, 64))
    quesn = torch.randn((8, 27))

    model = VQAModel(image_shape=(64, 64), vocab_size=27, num_answers=13)
    answer = model(image, quesn)
    print(answer.shape)
