#
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, AveragePooling1D, GlobalAveragePooling1D, Activation, Add
from keras.models import Model
from keras.layers.core import Dropout, Lambda, Dense
from keras.layers import Input, merge, Masking, BatchNormalization, Reshape, Flatten, Concatenate, Dot, Multiply,Highway, Lambda, LeakyReLU
from keras.optimizers import Adam

def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(layer_n//8, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out=Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x  

def ResCNN(input_shape=(56, 1)):
    layer_n = 64
    kernel_size = 3
    depth = 1

    input_layer = Input(input_shape)    
    input_layer_1 = AveragePooling1D(4)(input_layer)
    input_layer_2 = AveragePooling1D(14)(input_layer)
    
    print(input_layer.shape, input_layer_1.shape, input_layer_2.shape)

    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000

    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x


    x = cbr(x, layer_n*2, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])    
    x = cbr(x, layer_n*3, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1)
    out_2 = x
    
    x = Concatenate()([x, input_layer_2])    
    x = cbr(x, layer_n*4, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1)
    
    #regressor
    #x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    #out = Activation("sigmoid")(x)
    #out = Lambda(lambda x: 12*x)(out)
    
    #classifier
    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    #out = Activation("softmax")(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu') (x)
    x = Dense(32, activation='relu') (x)
    out = Dense(2, activation='softmax') (x)

    model = Model(input_layer, out)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=0.001), 
                  metrics=["accuracy"])

    
    return model


