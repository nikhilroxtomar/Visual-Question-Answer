from tensorflow.keras import layers as L
from tensorflow.keras import Model

def build_model(image_shape=(64, 64, 3), vocab_size=27, num_answers=13):

    """ CNN: image sub-network """
    image_input = L.Input(image_shape)

    x1 = L.Conv2D(8, 3, padding='same')(image_input)
    x1 = L.MaxPooling2D()(x1)
    x1 = L.Activation("relu")(x1)
    x1 = L.Conv2D(16, 3, padding='same')(x1)
    x1 = L.MaxPooling2D()(x1)
    x1 = L.Activation("relu")(x1)
    x1 = L.Flatten()(x1)

    x1 = L.Dense(32, activation='tanh')(x1)

    """ MLP: question sub-network """
    question_input = L.Input(shape=(vocab_size,))
    x2 = L.Dense(32, activation='tanh')(question_input)
    x2 = L.Dense(32, activation='tanh')(x2)

    """ Combine: answer sub-network """
    out = L.Multiply()([x1, x2])
    out = L.Dense(32, activation='tanh')(out)
    out = L.Dense(num_answers, activation='softmax')(out)

    model = Model(inputs=[image_input, question_input], outputs=out)

    return model
