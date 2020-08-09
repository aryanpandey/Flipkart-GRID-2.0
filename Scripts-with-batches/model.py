from tensorflow.keras.layers import Dense, Conv1D, LayerNormalization, Dropout, LeakyReLU, Input, Flatten, Multiply, Conv2D, Reshape, PReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import sigmoid
import tensorflow.compat.v1 as tf
import numpy as np
import keras.backend as K


class models():

    '''
    Function Name: __init__

    Parameters:  A(the number of checks the audio is broken into), L(the lengths of each chunk), N(the dimension of encoding vector), B, H, Sc, (hyperparameters 
    explained below), vr(number of TCNs), bl(number of ConvBlocks in each TCN)

    Returns: None

    Function Description:
    This function build the whole model using the building blocks for which functions are defined later. The model starts with a encoder model. Followed by a 
    channel changer and normalisation layer. This is then attached to vr number of TCNs with bl blocks in each. The BxN output of each block is the input for the
    next block and the ScxN output of each block is accumulated and added up. This accumulated ScxN then goes through a ChannelChanger and becomes AxN which gets 
    sigmoid activated and this output is the mask. This mask is then multiplied element wise to the output of the encoder and this value is passed on to the decoder.

    gbl_model is the whole setup described above and encoder decoder model is just the encoder paired with decoder of the gbl_model which is also trained
    separately.
    
    The loss function of the gbl_model has two parts
    1) The signal to noise ratio of the predicted output and the expected output which is a supervision loss and is being maximised
    2) A similarity measure between the predicted output and the rest of the original output which is a unsupervised loss and is being minimised

    The loss function of the encoder_decoder model is the signal to noise ratio of the input and the output, which are expected to be the same,
    which needs to be maximised. This training is included to ensure that the encoder decoder are indeed inverse of each other
    '''

    def __init__(self, A, L, N, B, H, Sc, vr, bl):
        self.L = L
        self.N = N
        self.B = B
        self.H = H
        self.A = A
        self.Sc = Sc

        self.whole_audio = Input(shape = (A*L,))
        
        self.encoder_model = self.encoder()
        gbl_model = Model(inputs = self.encoder_model.input, outputs = [self.encoder_model.output,LayerNormalization(axis=2)(self.encoder_model.output)])

        encoded_values = gbl_model.output[0]
        scale_conv1 = self.ChannelChanger(A, B)
        gbl_model = Model(inputs = gbl_model.input, outputs = scale_conv1(gbl_model.output[1]))
        
        new_block = self.ConvBlock(0,0,0)[0]
        gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output))
        Sc_layer = gbl_model.output[0]

        for blocks in range(bl-1):
            new_block = self.ConvBlock(blocks+1, 0, blocks+1)[0]
            gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
            Sc_layer = Add()([Sc_layer, gbl_model.output[0]])

        for verticals in range(vr-2):
            for blocks in range(bl):
                new_block = self.ConvBlock(blocks, verticals+1, blocks)[0]
                gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
                Sc_layer = Add()([Sc_layer, gbl_model.output[0]])
        
        for blocks in range(bl-1):
            new_block = self.ConvBlock(blocks+1, vr-1, blocks)[0]
            gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
            Sc_layer = Add()([Sc_layer, gbl_model.output[0]])

        new_block = self.ConvBlock(bl-1, vr-1, bl-1)[1]
        gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
        Sc_layer = Add()([Sc_layer, gbl_model.output])

        Sc_layer = PReLU()(Sc_layer)

        scale_conv2 = self.ChannelChanger(self.Sc, self.A)
        Sc_layer = scale_conv2(Sc_layer)
        mask = sigmoid(Sc_layer)

        mult = Multiply()([mask,encoded_values])
    
        self.decoder_model = self.decoder()
        

        final_output = self.decoder_model(mult)

        self.gbl_model = Model(inputs = [gbl_model.input, self.whole_audio], outputs = final_output)
        
        def gbl_model_loss(ytrue, ypred, whole_audio = self.whole_audio):
            
            non_primary = whole_audio - ypred
            sim = tf.math.divide(K.batch_dot(non_primary, ypred, axes = (1,1)),tf.math.multiply(tf.norm(non_primary, axis = [-2,-1]),tf.norm(ypred, axis = [-2,-1]))) 
            sim = K.mean(sim)
            
            
            ypred = ypred - K.mean(ypred)
            ytrue = ytrue - K.mean(ytrue)
            s_target = tf.math.multiply(tf.math.divide(K.batch_dot(ytrue, ypred, axes = (1,1)),K.batch_dot(ytrue,ytrue, axes = (1,1))),ytrue)
            e_noise = ypred - s_target
            SNR = 10*(K.log(tf.math.divide(K.batch_dot(s_target, s_target, axes = (1,1)),K.batch_dot(e_noise, e_noise, axes = (1,1))))/K.log(10.0))
            SNR = K.mean(SNR)
            return sim - 10*SNR
            
        
        self.gbl_model.compile(optimizer = Adam(lr=0.001), loss = gbl_model_loss)
        
        self.encoder_decoder_model = Model(inputs = self.encoder_model.input, outputs = self.decoder_model(self.encoder_model.output))
        def encoder_decoder_model_loss(ytrue, ypred):
            ypred = ypred - K.mean(ypred)
            ytrue = ytrue - K.mean(ytrue)            
            s_target = tf.math.multiply(tf.math.divide(K.batch_dot(ytrue, ypred, axes = (1,1)),K.batch_dot(ytrue,ytrue, axes = (1,1))),ytrue) # (2,220000)
            e_noise = ypred - s_target
            SNR = 10*(K.log(tf.math.divide(K.batch_dot(s_target, s_target, axes = (1,1)),K.batch_dot(e_noise, e_noise, axes = (1,1))))/K.log(10.0))
            SNR = K.mean(SNR)
            return -SNR
        self.encoder_decoder_model.compile(Adam(lr = 0.001), loss = encoder_decoder_model_loss)

    '''
    Function Name: Encoder

    Parameters: NONE

    Returns: A convolutional Encoder model

    Function Description:
    The following functions is an encoder function that encodes a sequence of length L, which in this case is a segment of the input audio,
    into a vector of dimension N. The audio is broken into A chunks of length L and each segment is encoded. So the Input dimension of the encoder is
    AxL and the output dimension is AxN.
    The encoder uses a gated convolution network. There are two flattened convolution outputs from the same input layer. One is relu activated and the 
    other is sigmoid activated and the final output is a product of the two
    '''

    def encoder(self):
        
        input_layer = Input(shape = (self.A, self.L, 1))
        
        layer1 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.A, self.N, 1))(layer2)

        layer4 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'sigmoid')(input_layer)
        layer5 = Flatten()(layer4)
        layer6 = Reshape(target_shape = (self.A, self.N, 1))(layer5)

        layer7 = Multiply()([layer3, layer6])
        model = Model(inputs = input_layer, outputs = layer7)
        return model

    '''
    Function Name: ConvBlock

    Parameters: x(dialation factor), vertical(index of the TCN created to keep track), block(index of the block in the TCN to keep track)

    Returns: model with and without skip connection

    Function Description:
    This function makes the building block of a TCN. It takes an input of dimension BxN where B is a hyperparameter. First it goes through a Convolution operation
    which gives an output of dimension HxN, where H is a hyperparameter. This then goes through a Depthwise convolution operation that operates on each row individually
    with a filter size determined by x and outputs and array of dimension HxN. This output then branches into two convolution layers, one giving an output of
    Dimension BxN and the other ScxN(called skip connection). There also exists a residual connection between the Input to this model and the BxN dimensional output
    '''

    def ConvBlock(self, x, vertical, block):
    
        input_layer = Input(shape = (self.B, self.N, 1))
        layer1 = Conv2D(self.H, (self.B, 1), input_shape = (self.B, self.N, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.H, self.N, 1))(layer2)
        layer4 = PReLU()(layer3)
        layer5 = LayerNormalization(axis=2)(layer4)
        layer6 = Conv2D(1, (1,2**(x)), activation = 'linear', padding = 'same')(layer5)
        layer7 = PReLU()(layer6)
        layer8 = LayerNormalization(axis=2)(layer7)
        layer9 = Conv2D(self.Sc, (self.H,1), activation = 'relu')(layer8)
        layer10 = Flatten()(layer9)
        Skip_Connection = Reshape(target_shape = (self.Sc, self.N, 1))(layer10)

        layer11 = Conv2D(self.B, (self.H,1), activation = 'relu')(layer8)
        layer12 = Flatten()(layer11)
        output = Add()([Reshape(target_shape = (self.B, self.N, 1))(layer12), input_layer]) 

        model = Model(inputs = input_layer, outputs = [Skip_Connection, output], name = 'Vertical'+str(vertical)+'block'+str(block))
        model1 = Model(inputs = input_layer, outputs = Skip_Connection)
        return [model, model1]
    
    '''
    Function Name: decoder

    Parameters: None

    Returns: A convolutional Decoder model

    Function Description:
    This model is the inverse of the encoder model given above. It is supposed to return the original sequence of length L when given
    and input of dimension N as encoded by the encoder model
    '''

    def decoder(self):
        input_layer = Input(shape = (self.A, self.N, 1))
        layer1 = Conv2D(self.L, (1, self.N), activation = 'linear', input_shape = (self.A, self.N, 1))(input_layer)
        layer2 = Flatten()(layer1)
        

        model = Model(inputs = input_layer, outputs = layer2, name = 'decoder')
        return model

    '''
    Function Name: ChannelChanger

    Parameters: input_channel(initial number of channels), output_channes(expected number of chanels)

    Returns: A tensorflow model 

    Function Description:
    This model purely exists to manipulate the dimensions of arrays so that they are compatible with the models following it
    '''

    def ChannelChanger(self, input_channels, output_channels):
        input_layer = Input(shape = (input_channels, self.N, 1))
        layer1 = Conv2D(output_channels, (input_channels, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (output_channels, self.N, 1))(layer2)

        model = Model(inputs = input_layer, outputs = layer3)
        return model
    
    '''
    Function Name: train

    Parameters:  input_value(the noisy audio reshaped to AxL), whole_audio(the noisy audio), output(expected output), epochs_gbl_model, 
                epochs_encoder_decoder, batch_size, valid(validation data), valid_target(validation data), valid_reshaped(validation data)

    Returns: None

    Function Description:
    This function trains the encoder decoder model and the overall model separately with the specified epochs and batch sizes
    '''

    def train(self, input_value, whole_audio, output, epochs_gbl_model, epochs_encoder_decoder, batch_size, valid, valid_target, valid_reshaped):
        
        print('...training encoder decoder...')
        self.encoder_decoder_model.fit(x = input_value, y = whole_audio,validation_data = (valid_reshaped,valid_target), batch_size=batch_size, epochs = epochs_encoder_decoder) 
    
        print('...training gbl model...')
        self.gbl_model.fit(x = [input_value, whole_audio], y = output, epochs = epochs_gbl_model, validation_data = ([valid_reshaped,valid], valid_target), batch_size = batch_size)   

    '''
    Function Name: export_model

    Parameters:  path(to save the model)

    Returns: None 

    Function Description:
    To save the mmodel with weights to the desired path after training
    ''' 

    def export_model(self, path):
        print('.....Saving Model......')
        self.gbl_model.save(path)