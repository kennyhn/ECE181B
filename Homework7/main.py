#!/usr/bin/env python3
import tensorflow           as tf
import matplotlib.pyplot    as plt

from tensorflow.keras       import datasets, layers, models
import copy



def main():
    train_images, train_labels, test_images, test_labels = prepareData()
    #validateData(train_images, train_labels)
    
    print("Do you want to load existing model? (y/n)")
    if input().lower() == 'y':
        # Load the model when we want to use it again.
        model = tf.keras.models.load_model('./model')
    else:
        model               = createModel()
    model.summary()
    history, model      = trainModel(model, train_images, train_labels, test_images, test_labels)
    print("Do you want to save model? (y/n)")
    if input().lower() == 'y':
        saveModel(model)
    plotModel(model, history, test_images, test_labels)
    wrongPredictions(model, test_images, test_labels)
    return 0

def prepareData():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values from 0-255 to be between 0 and 1
    return train_images / 255.0, train_labels, test_images / 255.0, test_labels

def validateData(train_images, train_labels):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    classes_to_be_plotted = copy.copy(class_names)
    i = 0
    j = 0
    plt.figure(figsize=(10,10))
    while (len(classes_to_be_plotted)) != 0:
        i += 1
        if classes_to_be_plotted.count(class_names[train_labels[i][0]]) != 0:
            classes_to_be_plotted.remove(class_names[train_labels[i][0]])
            plt.rc('font', size = 20)

            plt.subplot(2,5,j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)

            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(class_names[train_labels[i][0]])

            j += 1
    plt.show()

def createModel():
    model = models.Sequential()
    # Input (convolutional) layer 1 which takes in a 3-channel 32x32 image
    model.add(layers.Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = (32, 32, 3)))
    # Pooling layer 1
    model.add(layers.MaxPooling2D(pool_size = (2,2), strides = 2))

    # Convolutional layer 2
    model.add(layers.Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu'))
    # Pooling layer 2
    model.add(layers.MaxPooling2D(pool_size = (2,2), strides = 2))

    # Need to flatten the output before sending it into the dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 1024, activation = 'relu'))
    model.add(layers.Dropout(rate = 0.4))

    model.add(layers.Dense(units = 10, activation = 'softmax'))

    return model

def trainModel(model, training_images, training_labels, test_images, test_labels):
    optim       = tf.keras.optimizers.Adam(learning_rate = 0.0005, beta_1 = 0.99, beta_2 = 0.99999)
    model.compile(optimizer=optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(training_images, training_labels, epochs = 20, validation_split = 1/5, batch_size = 32)

    return history, model

def plotModel(model, history, test_images, test_labels):
    plt.figure(1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1])
    plt.legend(loc='lower right')
    
    plt.figure(2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([0.5, 4])
    plt.legend(loc='lower right')
    

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"test loss: {test_loss}, test accuracy: {test_acc}")
    plt.show()

def wrongPredictions(model,test_images, test_labels):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    classes_to_be_plotted = copy.copy(class_names)
    i                     = 0
    j                     = 0
    plt.figure(figsize=(10,10))
    y_test                = model.predict(test_images)
    while (len(classes_to_be_plotted)) != 0:
        y_test_pred = y_test[i].argmax(-1)

        i += 1
        if classes_to_be_plotted.count(class_names[test_labels[i][0]]) != 0 and y_test_pred != test_labels[i][0]:
            classes_to_be_plotted.remove(class_names[test_labels[i][0]])
            plt.rc('font', size = 20)

            plt.subplot(2,5,j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[i], cmap=plt.cm.binary)

            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(f'Correct: {class_names[test_labels[i][0]]}\n Prediction: {class_names[y_test_pred]}')

            j += 1
    plt.show()

def saveModel(model):
    # Now we are ready to save the network.
    model.save('./model')

if __name__ == '__main__':
    main()
