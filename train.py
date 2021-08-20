import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAvgPool2D, BatchNormalization, Concatenate, ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')

DATA_PATH = './dataset'

epochs = 80
batch_size = 128
image_size = (50, 50)
datagen = ImageDataGenerator(rescale=1./255, 
                             validation_split=0.2,
                             #rotation_range=10,
                             width_shift_range = 0.05,   
                             height_shift_range = 0.05,
                             zoom_range = 0.1)

train_gen = datagen.flow_from_directory(DATA_PATH, 
                                        target_size = image_size,
                                        class_mode = "categorical",
                                        batch_size = batch_size,
                                        subset = 'training',
                                        color_mode = 'rgb',
                                        shuffle = True)

val_gen = datagen.flow_from_directory(DATA_PATH, 
                                        target_size = image_size,
                                        class_mode = "categorical",
                                        batch_size = batch_size,
                                        subset = 'validation',
                                        color_mode = 'rgb',
                                        shuffle = True)

#datagen.fit(train_gen)

num_classes = len(train_gen.class_indices)

class DenseNet:
    def __init__(self, input_shape, num_classes, layers, growth_rate=32):
        repetitions = {
            121: (6,12,24,12),
            169: (6,12,32,32),
            201: (6,12,48,32)
        }
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = layers
        self.r = repetitions[layers]
        self.f = growth_rate

    def dense_block(self, tensor, r, f):
        for i in range(r):
            x = BatchNormalization(epsilon=1.001e-5)(tensor)
            x = ReLU()(x)
            x = Conv2D(f*4, (1, 1), strides=1, padding='same')(x)
            x = BatchNormalization(epsilon=1.001e-5)(x)
            x = ReLU()(x)
            x = Conv2D(f, (3, 3), strides=1, padding='same')(x)
            tensor = Concatenate()([tensor,x])
        return tensor

    def transition_layer(self, x):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(K.int_shape(x)[-1]//2, (1,1), strides=1, padding='same')(x)
        x = AveragePooling2D(2, strides=2)(x)
        return x

    def build(self):
        input = Input(self.input_shape)
        x = Conv2D(64, (7, 7), strides=2, padding='same')(input)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        
        d = self.dense_block(x, self.r[0], self.f)
        x = self.transition_layer(d)
        d = self.dense_block(x, self.r[1], self.f)
        x = self.transition_layer(d)
        d = self.dense_block(x, self.r[2], self.f)
        x = self.transition_layer(d)
        d = self.dense_block(x, self.r[3], self.f)
        

        x = GlobalAvgPool2D()(d)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

input_shape = (50, 50, 3)
model = DenseNet(input_shape, num_classes, layers=201).build()


reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, mode='auto', cooldown=3, min_lr=0.00001)

opt = tf.keras.optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_accuracy'])
model.summary()

history = model.fit(train_gen,
                    epochs=epochs,
                    validation_data=val_gen,
                    steps_per_epoch=train_gen.samples//batch_size,
                    verbose=1,
                    callbacks=[reduce_lr]
                    )

model.save("TCH.h5")

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
p_epochs = range(1, len(acc) + 1)

def plot_fig(epoch, train_acc, train_loss, val_acc, val_loss):
    plt.figure(figsize=(11, 7))
    plt.subplot(1,2,1)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.plot(p_epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(p_epochs, val_acc, 'g', label='Validation accurarcy')
    plt.ylim(0, 1)
    plt.legend(loc=5)

    plt.subplot(1,2,2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.plot(p_epochs, train_loss, 'r', label='Training loss')
    plt.plot(p_epochs, val_loss, 'y', label='Validation loss')
    #plt.ylim(ymax=0.8)
    plt.legend(loc=5)
    plt.savefig('train.png')
    plt.show()


#Train and validation accuracy
score = model.evaluate_generator(val_gen, steps=40, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print("Testing loss:", score[0])
print("Testing accuracy:", score[1])
print(score)

plot_fig(p_epochs, acc, loss, val_acc, val_loss)