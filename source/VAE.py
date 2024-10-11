import os
if __name__ == "__main__":
  # Lets not need to do this in command line, shall we?
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder






# Define the VAE
class VAE(tf.keras.Model):
  def __init__(self, parent, sessionId, labels, epochs, batch_size, outputFolder, outputEncoderPath, outputDecoderPath, outputSessionPath, outputType, latent_dim=2, encoderDecoderSizes=[], reluLayers=[], **kwargs):
    super(VAE, self).__init__(**kwargs)

    self.parent = parent
    self.sessionId = sessionId

    # VAE training loop settings
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.latent_dim = latent_dim
    self.epochs = epochs
    self.batch_size = batch_size

    self.labels = labels
    self.labelEncoder = None
    self.labelsEncoded = None
    self.labelOneHot = None
    self.encoders = {}
    self.decoders = {}
    self.encoderDecoderSizes = encoderDecoderSizes
    self.reluLayers = reluLayers
    self.outputFolder = outputFolder
    self.outputEncoderPath = outputEncoderPath
    self.outputDecoderPath = outputDecoderPath
    self.outputSessionFolder = outputSessionPath
    self.outputType = outputType

    self.labelEncoderPath = os.path.join(self.outputFolder, "label_encoder.pkl")
    self.labelsOneHotPath = os.path.join(self.outputFolder, "labels_one_hot.pkl")  # Path to save one-hot encoded labels

    if os.path.exists(self.labelsOneHotPath):
      self.labelOneHot = self.loadLabelsOneHot(self.labelsOneHotPath)
    self.prepShapes()

  def save(self):
    for x in range(len(self.encoderDecoderSizes)):
      size = self.encoderDecoderSizes[x]
      self.encoders[size].save( self.outputEncoderPath + "_" + str(size) + "." + self.outputType )
      self.decoders[size].save( self.outputDecoderPath + "_" + str(size) + "." + self.outputType )

    with open(self.labelEncoderPath, 'wb') as f:
      pickle.dump(self.labels, f)
    
    with open(self.labelsOneHotPath, 'wb') as f:
      pickle.dump(self.labelOneHot, f)  # Save one-hot encoded labels


  def getEncoderPath(self, size, isSession=False):
    if isSession:
      return os.path.join(self.outputSessionFolder, "vae_encoder_" + str(size) + "." + self.outputType)
    return self.outputEncoderPath + "_" + str(size) + "." + self.outputType
  def getDecoderPath(self, size, isSession=False):
    if isSession:
      return os.path.join(self.outputSessionFolder, "vae_decoder_" + str(size) + "." + self.outputType)
    return self.outputDecoderPath + "_" + str(size) + "." + self.outputType

  def saveEncoder(self, encoder, size=None, isSession=False):
    if size is None:
      for x in range(len(self.encoderDecoderSizes)):
        size = self.encoderDecoderSizes[x]
        encoder.save( self.getEncoderPath(size, isSession) )
    else:
      encoder.save( self.getEncoderPath(size, isSession) )
  def saveDecoder(self, decoder, size=None, isSession=False):
    if size is None:
      for x in range(len(self.encoderDecoderSizes)):
        size = self.encoderDecoderSizes[x]
        decoder.save( self.getDecoderPath(size, isSession) )
    else:
      decoder.save( self.getDecoderPath(size, isSession) )

  def saveSession(self,size=None):
    if not os.path.exists(self.outputSessionFolder):
      os.makedirs(self.outputSessionFolder)
    if size is None:
      for x in range(len(self.encoderDecoderSizes)):
        size = self.encoderDecoderSizes[x]
        self.saveEncoder( self.encoders[size], size, True )
        self.saveDecoder( self.decoders[size], size, True )
    else:
      self.saveEncoder( self.encoders[size], size, True )
      self.saveDecoder( self.decoders[size], size, True )


  def loadJson(self, jsonPath):
    with open(jsonPath, 'r') as f:
      return json.load(f)

  def loadEncoderLabels(self, labelEncoderPath):
    if not os.path.exists(labelEncoderPath):
      self.encodeLabels(self.labels)
      with open(labelEncoderPath, 'wb') as f:
        pickle.dump(self.labelEncoder, f)
    with open(labelEncoderPath, 'rb') as f:
      return pickle.load(f)
  
  def loadLabelsOneHot(self, labelsOneHotPath):
    if not os.path.exists(labelsOneHotPath):
      self.encodeLabels(self.labels)
      with open(labelsOneHotPath, 'wb') as f:
        pickle.dump(self.labelOneHot, f)
    with open(labelsOneHotPath, 'rb') as f:
      return pickle.load(f)

  def load(self, encoderDecoderSizes=None):
    if encoderDecoderSizes is None:
      encoderDecoderSizes = self.encoderDecoderSizes
    for x in range(len(self.encoderDecoderSizes)):
      size = encoderDecoderSizes[x]
      curEncoderPath = self.getEncoderPath(size)
      curDecoderPath = self.getDecoderPath(size)
      if not os.path.exists(curEncoderPath) :
        self.saveEncoder( self.encoders[size], size )
      if not os.path.exists(curDecoderPath) :
        self.saveDecoder( self.decoders[size], size )
      self.encoders[size] = models.load_model( curEncoderPath )
      self.decoders[size] = models.load_model( curDecoderPath )
    self.labels = self.loadEncoderLabels( self.labelEncoderPath )
    self.labelOneHot = self.loadLabelsOneHot( self.labelsOneHotPath )  # Load one-hot encoded labels

  def hasLoaded(self):
    if self.labelOneHot is None:
      self.load()
    if self.labelOneHot is None:
      return False
    return True

  def prepShapes(self, encoderDecoderSizes=None):
    wasNone = False
    if encoderDecoderSizes is None:
      wasNone = True
      encoderDecoderSizes = self.encoderDecoderSizes

    if self.labelOneHot is None:
      self.encodeLabels(self.labels)

    for x in range(len(encoderDecoderSizes)):
      size = encoderDecoderSizes[x]
      if wasNone and size not in self.encoderDecoderSizes:
        self.encoderDecoderSizes.append(size)
      self.parent.setNoTimerStatusText("Prepping Encoder & Decoders : "+ str(size))
      
      oneHotSize = len(self.labelOneHot[0])
      self.encoders[size] = self.build_encoder(self.latent_dim, (size, size, 1))
      self.decoders[size] = self.build_decoder(self.latent_dim, oneHotSize)
      self.parent.setNoTimerStatusText("Current Size : "+ str(size))


    # Encode labels as integers
  def encodeLabels(self, labels):
    if self.labelEncoder is None:
      self.labelEncoder = LabelEncoder()
    if self.labelsEncoded is None:
      self.labelsEncoded = self.labelEncoder.fit_transform(labels)

    # Convert labels to one-hot encoding
    self.labelOneHot = to_categorical(self.labelsEncoded)
    return self.labelEncoder, self.labelsEncoded

  # Define the encoder
  def build_encoder(self, latent_dim, toShape):
    inputs = layers.Input(shape=toShape)  # Specify the input size explicitly
    x = inputs

    for i in range(len(self.reluLayers)-1):
      x = layers.Conv2D(self.reluLayers[i], (3, 3), activation="relu", padding="same")(x)
      x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(self.reluLayers[-1], activation="relu")(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    return models.Model(inputs, [z_mean, z_log_var], name="shared_encoder")

  # Define the decoder
  def build_decoder(self, latent_dim, num_classes):
    latent_inputs = layers.Input(shape=(latent_dim,))

    # Initial Dense layer to transform latent space
    inputSize = 256
    x = layers.Dense((inputSize // 8) * (inputSize // 8) * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((inputSize // 8, inputSize // 8, 128))(x)
    
    # Sort and reverse the reluLayers to build the decoder in reverse order
    decodeOrder = self.reluLayers.copy()
    decodeOrder.sort(reverse=True)
    
    # Dynamic layers based on reluLayers
    for i in range(len(decodeOrder)):
      x = layers.Conv2D(decodeOrder[i], (3, 3), activation="relu", padding="same")(x)
      x = layers.UpSampling2D((2, 2))(x)

    # Output layer to match the number of classes in the one-hot encoded labels
    
    outputs = layers.Conv2DTranspose(num_classes, (3, 3), activation="softmax", padding="same")(x)
    print(outputs.shape)
    
    return models.Model(latent_inputs, outputs, name="decoder")

  def getEncoder(self, image):
    encoder = None
    decoder = None
    encoderSize = 0
    if type(image) == int:
      encoderSize = image
    else:
      encoderSize = image.shape[1]
    encoderKeys = list(self.encoders.keys())
    if encoderSize in encoderKeys:
      encoder = self.encoders[encoderSize]
    else:
      nearestSize = 0
      for x in range(len(encoderKeys)):
        size = encoderKeys[x]
        if size > encoderSize:
          break
        nearestSize = size
      encoder = self.encoders[nearestSize]
    
    decoderKeys = list(self.decoders.keys())
    if encoderSize in decoderKeys:
      decoder = self.decoders[encoderSize]
    else:
      nearestSize = 0
      for x in range(len(decoderKeys)):
        size = decoderKeys[x]
        if size > encoderSize:
          break
        nearestSize = size
      decoder = self.decoders[nearestSize]

    return encoder, decoder
  
  # Define the VAE loss function
  def vaeLoss(self, inputs, reconstructed_labels, z_mean, z_log_var, labels):
    # Reconstruction loss for images
    inputs_shape = tf.shape(inputs)
    restruct_shape = tf.shape(reconstructed_labels)
    reconstructed_labelForLoss = tf.image.resize(reconstructed_labels, [inputs_shape[1], inputs_shape[2]])
    reconstructed_labelForLoss = tf.reshape(reconstructed_labelForLoss, [inputs_shape[0], inputs_shape[1], inputs_shape[2], -1])
    reconstructed_labelForLoss = reconstructed_labelForLoss[:, :, :, 0]
    reconstructed_labelForLoss = tf.reshape(reconstructed_labelForLoss, [inputs_shape[0], inputs_shape[1], inputs_shape[2], 1])
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, reconstructed_labelForLoss))

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

    # Categorical cross-entropy loss for labels
    # Still can't get it right,
    #   Input sizes chance caue of session sizes, yet localizing the labels is a bit tricky
    #     I'll see where I get in the next week or two
    #label_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, reconstructed_labels))

    return reconstruction_loss + kl_loss# + label_loss

  @tf.function
  def trainStep(self, images, labels, encoder, decoder):
    with tf.GradientTape() as tape:
      z_mean, z_log_var = encoder(images)
      z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))
      reconstructed_labels = decoder(z)
      
      # Fit Reconstruction Labels to labels size
      #tfShape = tf.shape(images)
      #reconstructed_labels = tf.image.resize(reconstructed_labels, tfShape[1:3])
      #reconstructed_labels = tf.reshape(reconstructed_labels, [tfShape[0], tf.shape(reconstructed_labels)[1], tf.shape(reconstructed_labels)[2], 1])
      
      
      loss = self.vaeLoss( images, reconstructed_labels, z_mean, z_log_var, labels )

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return loss

  def checkOutputSize(self, size):
    decoderKeys = list(self.decoders.keys())
    nearestSize = 0
    for x in range(len(decoderKeys)):
      curSize = decoderKeys[x]
      if curSize > size:
        break
      nearestSize = curSize
    return nearestSize

  # Train the VAE on different scales
  # TODO : Add support for multiple image sets
  def train(self, imageSets=[], labels=[], epochs=None, batch_size=None):
    self.parent.setStatusText("Training VAE on different scales...")
    if len(imageSets) == 0:
      self.parent.setStatusText("Error: No image sets provided")
      return
    imageSetKeys = list(imageSets.keys())

    isExit = False
    epochText = ""
    curLoss = -1
    for x in imageSetKeys:
      images = imageSets[x]
      num_batches = len(images) // batch_size
      imageKeyText = "" if len(imageSetKeys) == 1 else "Size '"+str(x)+"'; "
      # Detect the size of the first image to determine which encoder to use
      
      encoder, decoder = self.getEncoder(images[0])
      if encoder is None:
        self.parent.setStatusText("Error: No encoder found")
        return

      for epoch in range(epochs):
        epochText = imageKeyText+"'Epoch " + str(epoch + 1) + " / " + str(epochs)
        for i in range(num_batches):
          if self.checkEscape() :
            isExit = True
            break
          batch_images = images[i * batch_size:(i + 1) * batch_size]
          batch_labels_one_hot = self.labelOneHot[i * batch_size:(i + 1) * batch_size]

          # Offset batch labels to full 
          # self.labelOneHot
          loss = self.trainStep(batch_images, batch_labels_one_hot, encoder, decoder)
          curLoss = loss.numpy()
          if self.parent:
            dispText = epochText + "; Batch " + str(i) + " of " + str(num_batches) + "; Loss : " + str(curLoss)
            print(dispText)
            self.parent.setNoTimerStatusText( dispText )
        self.saveSession(x)
        if isExit:
          self.parent.setStatusText("Exiting Training...")
          break
      if isExit:
        break

  def checkEscape(self):
    return self.parent.checkBreak()
      