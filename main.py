import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras as keras




# Load the data from the data folder
def load_data(file, ratio=0.2, maxInputs=3000):

    # Load all the files in the data folder
    all_files = glob.glob(os.path.join('data', '*.npy'))
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    
    # Load the data and labels
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: maxInputs, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, extra = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
    data = None
    labels = None
    
    # Mix up the data
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    # Split the data into training and testing
    split = int(x.shape[0]/100*(ratio*100))

    x_test = x[0:split, :]
    y_test = y[0:split]

    x_train = x[split:x.shape[0], :]
    y_train = y[split:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names

# Plot a random image from the dataset
def plot(x_train, y_train, class_names):
    picture = randint(0, len(x_train))
    plt.imshow(x_train[picture].reshape(28,28)) 
    plt.show()
    print(f"The image is a(n) {class_names[int(y_train[picture].item())]}")

def preprocess(x_train, y_train, x_test, y_test, num_classes, image_size):
    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # One hot encode the labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

# Create the model
def model(x_train, y_train, image_size, validation_split=0.1, batch_size = 256, verbose=2, epochs=5):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(27, activation='softmax')) 

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    history = model.fit(x = x_train, y = y_train, validation_split=validation_split, batch_size=batch_size, verbose=verbose, epochs=epochs)
    print(model.summary())
    return model

# Test the model
def test_acc(model, x_test, class_names):
    idx = randint(0, len(x_test))
    img = x_test[idx]
    plt.imshow(img.squeeze()) 
    plt.show()
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    ind = (-pred).argsort()[:5]
    guesses = [class_names[x] for x in ind]
    print(guesses)

# Put everything together
def main():
    x_train, y_train, x_test, y_test, class_names = load_data('data')
    plot(x_train, y_train, class_names)
    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test, len(class_names), 28)
    CNN = model(x_train, y_train, 28, validation_split=0.1, batch_size = 256, verbose=2, epochs=5)
    score = CNN.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_acc(CNN, x_test, class_names)

# Call main function
if __name__ == '__main__':
    main()



