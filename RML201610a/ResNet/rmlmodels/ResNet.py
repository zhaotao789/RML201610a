import os
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,ReLU,Dropout,Activation,concatenate,Softmax,Conv2D,MaxPool2D,Add,BatchNormalization
from keras.layers import Bidirectional,Flatten
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# pip install tensorflow-gpu==2.5.0 -i https://pypi.doubanio.com/simple
def ResNet(weights=None,
             input_shape=[2,128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr=0.6
    input = Input(input_shape+[1],name='input')

    x=Conv2D(256,(1,3), name="conv1",kernel_initializer='glorot_uniform', padding='same')(input)
    x = Activation('relu')(x)
    # x = Dropout(dr)(x)
    x=Conv2D(256,(2,3), name="conv2", kernel_initializer='glorot_uniform',padding='same')(x)
    # x = Dropout(dr)(x)
    x1 = Add()([input, x])
    x1 = Activation('relu')(x1)
    x=Conv2D(80,(1,3), activation="relu", name="conv3", kernel_initializer='glorot_uniform',padding='same')(x1)
    x=Conv2D(80,(1,3), activation="relu", name="conv4", kernel_initializer='glorot_uniform',padding='same')(x)
    x = Dropout(dr)(x)
    x=Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dr)(x)
    output = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs = input,outputs = output)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    train_X = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\训练集\X_train.npy')
    train_Y = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\训练集\Y_train.npy')
    test_X = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\测试集\X_test.npy')
    # 将第二、三维转换为16x16
    train_X = train_X.reshape(train_X.shape[0], 2, 128)
    test_X = test_X.reshape(test_X.shape[0], 2, 128)

    # y = np.argmax(train_Y, axis=1)

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
    nb_epoch = 10000  # number of epochs to train on
    batch_size = 64  # training batch size
    model =  ResNet(None,input_shape=[2,128],classes=11)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    model.fit(X_train,
              Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=[
                  # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                  #                                 mode='auto'),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patince=5,
                                                    min_lr=0.0000001),
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
                  # keras.callbacks.TensorBoard(histogram_freq=1,write_graph=True,write_images=True)
              ]
              )
    test_Y=model.predict(X_test, batch_size=batch_size)
    one_hot_preds = np.zeros((len(test_Y), 11))
    one_hot_preds[np.arange(len(test_Y)), test_Y] = 1
    np.save("Y_test.npy", one_hot_preds)

    # print('models layers:', model.layers)
    # print('models config:', model.get_config())
    # print('models summary:', model.summary())