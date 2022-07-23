#%%
import numpy as np
import cv2
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from numpy.random import randn
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import TimeDistributed
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import Embedding
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model

# define the discriminator model
def define_discriminator(image_shape, vector_shape, n_classes=7):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # vector input
    in_target = Input(shape=(vector_shape[0], vector_shape[1]))
    # label input
    in_label = Input(shape=(1,))
    # Vector layer
    n_nodes = (image_shape[0] * image_shape[1] * image_shape[2])
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((image_shape[0], image_shape[1], image_shape[2]))(li)
    # LSTM Model
    p = LSTM(200, activation='relu')(in_target)
    p = Dense(n_nodes)(p)
    # Reshape and mergo to image dimension
    p = Reshape((image_shape[0], image_shape[1], image_shape[2]))(p)
    merged = Concatenate()([p, in_src_image, li])
    # concatenate images channel-wise
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Flatten()(d)
    d = Dropout(0.4)(d)
    out_layer = Dense(1, activation='sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same',
    kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define the standalone generator model
def define_generator(in_shape, vector_shape, latent_dim, n_classes=7):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    in_lat = Input(shape=latent_dim)
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    li = Dense(latent_dim)(li)
    li = Reshape([latent_dim])(li)
    gen = LeakyReLU(alpha=0.2)(in_lat)
    # merge image gen and label input
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    b = Flatten()(b)
    # Merge latent space variable input with network
    b = Concatenate()([b, gen, li])
    # Reshape to vector size
    n_nodes = (vector_shape[0] * vector_shape[1])
    b = Dense(n_nodes)(b)
    b = Reshape([vector_shape[0], vector_shape[1]])(b)
    # encoder-decoder LSTM model
    d = LSTM(200, activation='relu', return_sequences=True)(b)
    d = LSTM(200, activation='relu')(d)
    d = Dense(n_nodes)(d)
    out_layer = Reshape([vector_shape[0], vector_shape[1]])(d)
    # output


    # define model input & output
    model = Model([in_image, in_lat, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, vector_shape, latent_dim):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    in_lat = Input(shape=latent_dim)
    in_label = Input(shape=(1,))
    # connect the source image to the generator input
    gen_out = g_model([in_src, in_lat, in_label])
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out, in_label])
    # src image as input, generated image and classification output
    model = Model([in_src, in_lat, in_label], [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
	return [X1, X2, X3-1]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples):
	# unpack dataset
	img, vec, label = dataset
	# choose random instances
	ix = randint(0, img.shape[0], n_samples)
	# retrieve selected images
	X1, X2, X3 = img[ix], vec[ix], label[ix]
	# generate 'real' class labels 
	y = ones((n_samples, 1))
	return [X1, X2, X3], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, label, n_samples):
    # Generate latent space points
    z_input = generate_latent_points(latent_dim, n_samples)
    # generate fake instance
    X = g_model.predict([samples, z_input, label])
    # create 'fake' class labels 
    y = zeros((n_samples, 1))
    return X, y

# Overlay trajectories (data_y) to the image (data_x)
def create_trajectory(data_x, data_y, obs_len=10):
    # Calibration parameter to overlay for a 1280x360 resolution image
    K = np.array([[537.023764, 0, 640 , 0], 
                    [0 , 537.023764, 180, 0], 
                    [0, 0, 1, 0]])
    # Rotation matrix to obtain egocentric trajectory
    Rt = np.array([[0.028841, 0.007189, 0.999558, 1.481009],
                    [-0.999575,  0.004514,  0.028809,  0.296583],
                    [ 0.004305,  0.999964, -0.007316, -1.544537],
                    [ 0.      ,  0.      ,  0.      ,  1.      ]])

    # Resize data back to 1280x360
    data_x = cv2.resize(data_x, (1280,360))
    # Add column of ones for rotation matrix multiplication
    data_y = np.hstack((data_y, np.ones((len(data_y),1))))
    # Draw points
    for m in range(obs_len, data_y.shape[0]):
        # Rotation matrix multiplication of trajectory 
        A = np.matmul(np.linalg.inv(Rt), data_y[m, :].reshape(4, 1))
        # Egocentric view of trajectory
        B = np.matmul(K, A)
        # Circle location of trajectories 
        x = int(B[0, 0] * 1.0 / B[2, 0])
        y = int(B[1, 0] * 1.0 / B[2, 0])
        if (x < 0 or x > 1280 - 1 or y > 360 - 1):
            continue
        # Use opencv to overlay trajectories
        data_x = cv2.circle(data_x, (x, y), 3, (0, 0, 255), -1)
    return data_x

def generate_drvact_text(drvact):
    text = '[warnings] drvact label is not defined ...'
    if (drvact == 1):
        text = 'Go'
    elif (drvact == 2):
        text = 'Turn Left'
    elif (drvact == 3):
        text = 'Turn Right'
    elif (drvact == 4):
        text = 'U-turn'
    elif (drvact == 5):
        text = 'Left LC'
    elif (drvact == 6):
        text = 'Right LC'
    elif (drvact == 7):
        text = 'Avoidance'
    return text

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=1):
    # select a sample of input images
    [X_real_img, X_real_vec, X_label], _ = generate_real_samples(dataset, n_samples)
    # generate a batch of fake samples
    X_fake_vec, _ = generate_fake_samples(g_model, X_real_img, X_label, n_samples)
    # scale all pixels from [-1,1] to [0,1]
    X_real_img = (X_real_img + 1) / 2.0
    pyplot.figure(figsize=(32.0, 20.0))
    # plot real source images
    for i in range(n_samples):
        orig_image = (X_real_img[i]* 255).astype(np.uint8)
        orig_image = cv2.resize(orig_image, (1280,360))
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(orig_image)
    # plot generated target image
    for i in range(n_samples):
        fake_sample = create_trajectory((X_real_img[i]* 255).astype(np.uint8), X_fake_vec[i])
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(fake_sample)
    # plot real target image
    for i in range(n_samples):
        true_sample = create_trajectory((X_real_img[i]* 255).astype(np.uint8), X_real_vec[i])
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(true_sample)
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# train models
def train(d_model, g_model, gan_model, dataset, latent_dim, n_epochs=100, batch_size=14):
    # iteration number to output checkpoint
    display_iter = 5000
    # calculate the number of batches
    n_batch = int(len(dataset[0]) / batch_size)
    # calculate the number of training iterations
    n_steps = n_batch * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_real_img, X_real_vec, X_label], y_real = generate_real_samples(dataset, batch_size)
        # generate a batch of fake samples
        X_fake_vec, y_fake = generate_fake_samples(g_model, X_real_img, X_label, batch_size)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_real_img, X_real_vec, X_label], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_real_img, X_fake_vec, X_label], y_fake)
        X_lat = generate_latent_points(latent_dim, batch_size)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_real_img, X_lat, X_label], [y_real, X_real_vec])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % int(display_iter) == 0:
            summarize_performance(i, g_model, dataset)
            print('saved')

            
#%%
%%time
# load image data
dataset = load_real_samples('data/dual_condition_dataset_train.npz')
print('Loaded', dataset[0].shape, dataset[1].shape, dataset[2].shape)

#%%
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
vector_shape = dataset[1].shape[1:]
latent_dim = 512
# %%
# define the models
d_model = define_discriminator(image_shape, vector_shape)
g_model = define_generator(image_shape, vector_shape, latent_dim)
# %%
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape, vector_shape, latent_dim)

#%%
# train model
train(d_model, g_model, gan_model, dataset, latent_dim)

# %%
