from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# load dbata
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# create a grid of 3x3 images
print(X_train.shape[0])
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9,shuffle=False):
    print(X_batch.min(), X_batch.mean(), X_batch.max())
# create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True,figsize=(4,4))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X_batch[i*3+j], cmap=plt.get_cmap("gray"))
# show the plot
    plt.show()
    break

datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True, zca_whitening=True)
datagen.flow(X_train, y_train, batch_size=9, shuffle=False,save_to_dir='images', save_prefix='aug', save_format='png')