from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform

def Resnet(size):
    def identity_block(X, f, filters, initializer=random_uniform):
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X) # Default axis
        X = Activation('relu')(X)
        
        ### START CODE HERE
        ## Second component of main path (≈3 lines)
        ## Set the padding = 'same'
        X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = "same", kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation("relu")(X)

        ## Third component of main path (≈2 lines)
        ## Set the padding = 'valid'
        X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = "valid", kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        
        ## Final step: Add shortcut value to main path, and pass it through a RELU activation 
        X = Add()([X_shortcut,X])
        X = Activation("relu")(X)
        ### END CODE HERE

        return X

    def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        
        # First component of main path glorot_uniform(seed=0)
        X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        ### START CODE HERE
        
        ## Second component of main path 
        X = Conv2D(filters = F2, kernel_size = f,strides = 1,padding = "same",kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X) 
        X = Activation("relu")(X)

        ## Third component of main path 
        X = Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = "valid" , kernel_initializer = initializer(seed = 0))(X)
        X = BatchNormalization(axis = 3)(X) 
        
        ##### SHORTCUT PATH ##### 
        X_shortcut = Conv2D(filters = F3,kernel_size=1, strides = (s,s), padding = "valid", kernel_initializer = initializer(seed = 0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
        
        ### END CODE HERE

        # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X


    def ResNet50(input_shape = (size, size, 3), classes = 2, training=False):
        """
        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
        X = identity_block(X, 3, [64, 64, 256])
        X = identity_block(X, 3, [64, 64, 256])

        ### START CODE HERE
        
        # Use the instructions above in order to implement all of the Stages below
        # Make sure you don't miss adding any required parameter
        
        ## Stage 3 
        # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = convolutional_block(X,f = 3, filters = [128,128,512], s = 2)
        
        # the 3 `identity_block` with correct values of `f` and `filters` for this stage
        X = identity_block(X,3,[128,128,512])
        X = identity_block(X,3,[128,128,512])
        X = identity_block(X,3,[128,128,512])

        # Stage 4 
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = convolutional_block(X, f = 3, filters = [256,256,1024], s=2)
        
        # the 5 `identity_block` with correct values of `f` and `filters` for this stage
        X = identity_block(X,3,[256,256,1024])
        X = identity_block(X,3,[256,256,1024])
        X = identity_block(X,3,[256,256,1024])
        X = identity_block(X,3,[256,256,1024])
        X = identity_block(X,3,[256,256,1024])

        # Stage 5
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = convolutional_block(X,f=3, filters=[512,512,2048], s=2)
        
        # the 2 `identity_block` with correct values of `f` and `filters` for this stage
        X = identity_block(X,3,[512,512,2048])
        X = identity_block(X,3,[512,512,2048])

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
        X = GlobalAveragePooling2D()(X)
        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
        
        
        # Create model
        model = Model(inputs = X_input, outputs = X)

        return model

    model = ResNet50(input_shape=(size,size,3))
    return model