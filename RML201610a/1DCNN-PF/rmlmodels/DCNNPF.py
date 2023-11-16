import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

def DLmodel(weights=None,
             input_shape=[128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    
    input1 =Input(input_shape,name='input1')
    Reshape1=Reshape(input_shape+[1])(input1)
    input2=Input(input_shape,name='input2')
    Reshape2=Reshape(input_shape+[1])(input2)

    x2=Conv1D(64,3, activation="relu" )(Reshape1)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    
    x3=Conv1D(64,3, activation="relu" )(Reshape2)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)

    x=concatenate([x2,x3])
    x=Conv1D(64,3, activation="relu" )(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)    
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Flatten()(x)
    x = Dense(128,activation='selu',name='fc1')(x)
    x=Dropout(dr)(x)
    x = Dense(128,activation='selu',name='fc2')(x)
    x=Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = [input1,input2],outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    model = DLmodel(None,classes=11)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:')
    model.summary()