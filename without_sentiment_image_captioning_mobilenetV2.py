
# coding: utf-8
'''
We have modified this code from the Image Captioning with visual attention
Link : https://www.tensorflow.org/tutorials/text/image_captioning

The captioning is evaluated in the Evaluation code for MobilenetV2 
'''

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[3]:


import tensorflow as tf

# import for splitting train test data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
# package to get a progress bar for saving vectors of each image in a .npy file
from tqdm import tqdm
from IPython.display import Image 



# In[4]:
# Code block to automatically download the annotations and images (MS-COCO dataset) from COCO page

annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_val2014.json'

name_of_zip = 'val2014.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
  image_zip = tf.keras.utils.get_file(name_of_zip,
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/val2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip)+'/val2014/'
else:
  PATH = os.path.abspath('.')+'/val2014/'



# In[5]:


# Read the json file
with open(annotation_file, 'r') as ann_file:
    annotations = json.load(ann_file)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    # the captions must have a <start> and <end> that is later tokenized. The result captions also come with these tokens
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)

    # The storing is based on captions, hence one image has multiple captions
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 150000 captions from the shuffled set
# Due to constraints on the processing we select the first 150000 caption,image pairs
num_examples = 150000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]



# In[7]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # mobilenet requires all the images in a 299*299 pixel format
    img = tf.image.resize(img, (299, 299))
    #using the preprocess_input function of the mobilenet
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, image_path


# In[8]:

# we can use the pretrained weights of the imagenet model to train on the COCO dataset
# we use include_top = False to extract the second last layer instead of last softmax layer
image_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                weights='imagenet')

# this gets the second last vector
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[9]:

# Get unique images to convert to numpy files
encode_train = sorted(set(img_name_vector))

# convert the sorted train set to tf.data dataset
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# parallel execution to convert each image Inceptionv3 compatible format
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

# image_dataset will have image_path and 299*299 resized image
for img, path in tqdm(image_dataset):
  # We call the model function to get the vector of shape 81x2048 
  batch_features = image_features_extract_model(img)
  # extracting the batches and keeping 1st dimension as is
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    # save the vectors as numpy files with the .npy extension by default  
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


# In[10]:


#The maximum size of captions to take the padding limit 
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[11]:


# Choose the top 5000 words from the vocabulary
top_k = 5000
# we tokenize the captions to only include top 5000 words, we exclude other infrequent words and replace them with <unk>, we also remove the punctuations 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
#text to sequence expects a sequence of words
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[12]:

# we change <pad> which is default word based on the longest caption with 0
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# In[13]:


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[14]:


# Pad each vector to the max_length of the captions
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[15]:


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# ## Split the data into training and testing

# In[16]:


# We use the train test split to split 80%-20% data of train itself
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)



# In[18]:


# This batch size should be perfectly divisible to the num of examples that are considered
BATCH_SIZE = 400

BUFFER_SIZE = 1000
# The vector embeddign size for deciding the RNN_GRU decoder 
embedding_dim = 256
# total units in GRU cell
units = 512
# Total number of terms in the vocabulary - 6409 total vocab_size
vocab_size = len(tokenizer.word_index) + 1
# total 150000 images hence the num_steps will be 300
num_steps = len(img_name_train) // BATCH_SIZE

tures_shape = 1280
attention_features_shape = 49

# In[19]:


# Load the numpy files each numpy file has a shape of 81x1280
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


# In[20]:

# the dataset will be a image path and caption pair
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# We sorted the encode train to store the numpy files hence we shuffle the dataset pair back according to batch size and form batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# In[21]:


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    
    # hidden_with_time_axis shape is batch_size, 1, hidden_size
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape is batch_size, 64, hidden_size
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector is batch_size, hidden_size
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[22]:


class CNN_Encoder(tf.keras.Model):
    # the input to the Encoder will be batch_sizex64x2048
    # this will change to dimensions batch_sizex64x256
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # input shape is batchx64x2048 while output is batchx64x256(embedding dimensions) with a dense layer
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        # the input to the dense layer will be the image tensor
        x = self.fc(x)
        # relu activation over the fully connected maintains the shape
        x = tf.nn.relu(x)
        return x


# In[23]:


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    # units in the RNN GRU cell = 512
    self.units = units

    # define Embedding layer with the vocab_size and embedding dimension 6409 and 256
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # define GRU cell with 256 units with return sequences set to true to give successive outputs to next unit
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # dense layers with output dim 512
    self.fc1 = tf.keras.layers.Dense(self.units)
    # dense layer with 6409 dimensions to get back the token numbers activated
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    # Call to define Attention object
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x)

    # shape will be batch_sizex1x(256+hiddensize)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #pass the concatenated output to GRU cell
    output, state = self.gru(x)

    # shape will be batch_size x max_length x hidden_size
    x = self.fc1(output)

    # collapsing 1st dimension
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape batch_size * max_length, vocab
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# In[24]:

# call to encoder and decoder classes
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[25]:

# set optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
# loss calculation
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)



# In[26]:

# set up of checkpoint and checkpoint manager
checkpoint_path = "./checkpoints/without_sentiment_mobilenetV2"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[27]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


# In[29]:


@tf.function
def train_step(img_tensor, target):
  loss = 0

  # set initial zeros with start tokens of 100 or batch_size
  hidden = decoder.reset_state(batch_size=target.shape[0])

  # Every time the dec_input will have the previous word predicted going as input
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

  with tf.GradientTape() as tape:
      # pass the image tensor through the encoder for embedding dimension based output
      features = encoder(img_tensor)
      # the loop runs for the total number of words in its corresponding caption
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          # calculate loss based on the current word predicted
          loss += loss_function(target[:, i], predictions)

          # using the predicted word as next unit input
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  # get all trainable variables that can be optimized with back prop
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)
  # apply weight optimization on trainable variables
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


# In[30]:


EPOCHS = 40

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    # dataset will have 2 parameters tensor and caption and batches
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # storing the epoch end loss value to plot later

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



# In[32]:
def evaluate(image):
    
    # decoder reset state with start tag
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    # run resize for mobilenet feed forward
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        #keep predicting till index to word mapping is <end> for end caption 
        if tokenizer.index_word[predicted_id] == '<end>':
            return result
    
        #else change the word to next word in decoder
        dec_input = tf.expand_dims([predicted_id], 0)

    # if no <end> tag is returned it continues to generate words till max length     
    return result

# In[34]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
# opening the image
pil_img = Image(img_name_val[rid])
display(pil_img)
