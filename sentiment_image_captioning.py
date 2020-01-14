
# coding: utf-8
'''
We have modified this code from the Image Captioning with visual attention
Link : https://www.tensorflow.org/tutorials/text/image_captioning

The captioning is evaluated in the Evaluation code for Inceptionv3 
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

# List to maintain the index of captions and image name paths
all_captions = []
all_img_name_vector = []
sentiment = []

# Load the senticap annotations
with open("senticap_dataset.json", 'r') as f:
    senti_annotations = json.load(f)

# form a similar file asthat of the validation captions of COCO
senti_data = {"annotations":[]}
for annot in senti_annotations['images']:
    img_id = int(annot['filename'].split('_')[-1].split('.')[0])
    for sen in annot['sentences']:
        senti_data["annotations"].append({"image_id":img_id,"caption":sen['raw'],"sentiment":sen["sentiment"]}) 

# collect all the uniques image_id
res = [v['image_id'] for v in senti_data['annotations']]
res = set(res)

# iterate on the annotations
for entry in senti_data['annotations']:
    caption = '<start> ' + entry['caption'] + ' <end>'
    all_captions.append(caption)
    img_id = entry['image_id']
    full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (img_id)
    senti = entry['sentiment']
    if senti == 0:
        senti = -1
    all_img_name_vector.append([full_coco_image_path,senti])
    
# iterate on the annotations of COCO calidation captions
for annot in annotations['annotations']:
    image_id = annot['image_id']
    if image_id in res:
        caption = '<start> ' + annot['caption'] + ' <end>'
        full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
        all_img_name_vector.append([full_coco_image_path,0])
        all_captions.append(caption)
        

# shuffle the arrays to avoid grouping of different captions with same images
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 20000 captions from the shuffled set out of 20002
num_examples = 20000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]


# In[7]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # inception v3 requires all the images in a 299*299 pixel format
    img = tf.image.resize(img, (299, 299))
    #using the preprocess_input function of the inception_v3 
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# In[8]:

# we can use the pretrained weights of the imagenet model to train on the COCO dataset
# we use include_top = False to extract the second last layer instead of last softmax layer
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
# this gets the second last vector
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# In[7]:
# img_name_vector has a file path and sentiment association
sentiment = img_name_vector
# remove sentiment after creating a copy
img_name_vector = [x[0] for x in img_name_vector]

# In[9]:

# Get unique images

encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# parallel execution to convert each image Inceptionv3 compatible format
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


for img, path in tqdm(image_dataset):
  # We call the model function to get the vector of shape 81x2048 
  batch_features = image_features_extract_model(img)
  
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


# In[16]:


# We use the train test split to split 80%-20% data of train itself
# here we split the sentiment data to maintain the sequence of pfile path caption and sentiment
img_name_train, img_name_val, cap_train, cap_val = train_test_split(sentiment,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)




# In[18]:
# This batch size should be perfectly divisible to the num of examples that are considered
BATCH_SIZE = 100
BUFFER_SIZE = 1000

# The vector embeddign size for deciding the RNN_GRU decoder 
embedding_dim = 256

# number of GRU units in the cell
units = 512

# Total number of terms in the vocabulary - 6409 total vocab_size
vocab_size = len(tokenizer.word_index) + 1

# total 20000 images hence the num_steps will be 200
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2052)
# These two variables represent that vector shape
features_shape = 2052
attention_features_shape = 64

# In[19]:


# Load the numpy files each numpy file has a shape of 64x2052
def map_func(img_name, cap,senti):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  # after loading the numpy files we make a similar vector with 4 more dimensions at axis 1
  comb_feats = np.zeros([img_tensor.shape[0],img_tensor.shape[1]+4], dtype=np.float32)
  comb_feats[:,:2048] = img_tensor
  # adding an encoding vale which corresponds to sentiments
  #Positive sentiment
  if senti == 1:
      result = [1,0,0,1]
  # negative sentiment
  elif senti == -1:
      result = [0,0,1,2]
  # neutral sentiment
  else:
      result = [0,1,0,0]
  comb_feats[:,2048:] = result
  # combine and return the new image vector of shape 64x2052
  return comb_feats,cap,senti


# In[20]:


dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in img_name_train], cap_train,[x[1] for x in img_name_train]))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(
          map_func, [item1,item2,item3], [tf.float32, tf.int32,tf.int32]),
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

    # score shape is batch_size, 64, hidden_size
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector is batch_size, hidden_size
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[22]:


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # input shape is batchx64x2052 while output is batchx64x256(embedding dimensions) with a dense layer
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

    # the shape will be batch_sizex1x256
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

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# In[26]:

# set up of checkpoint and checkpoint manager

checkpoint_path = "./checkpoints/sentiment"
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
      features = encoder(img_tensor)

      # pass the image tensor through the encoder for embedding dimension based output
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          # calculate loss based on the current word predicted
          loss += loss_function(target[:, i], predictions)

          # using the predicted word as next unit input
          dec_input = tf.expand_dims(target[:, i], 1)

  # calculate total loss based on number of words in caption
  total_loss = (loss / int(target.shape[1]))

  # get all trainable variables that can be optimized with back prop
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)
  # apply weight optimization on trainable variables
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


# In[30]:


EPOCHS = 100

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    # dataset will have 3 parameters tensor and caption,sentiment and batches
    for (batch, (img_tensor, target,senti)) in enumerate(dataset):
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

    # Extract the image to be tested
    temp_input = tf.expand_dims(load_image(image[0])[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    new_img_tensor_val = np.zeros([img_tensor_val.shape[0],img_tensor_val.shape[1],(img_tensor_val.shape[2]+4)],dtype=np.float32)
    # Only the first 2048 vectors will be returned
    # Give user input to decide the level of sentiment we need from the image
    # 0 for neutral, 1 for positive, -1 for negative
    new_img_tensor_val[:,:,:2048] = img_tensor_val
    if image[1] == 0:
        new_img_tensor_val[:,:,2048:] = [0,1,0,0]
    elif image[1] == 1:
        new_img_tensor_val[:,:,2048:] = [1,0,0,1]
    elif image[1] == -1:
        new_img_tensor_val[:,:,2048:] = [0,0,1,2]
    
    # form new image tensor 
    img_tensor_val = new_img_tensor_val
        
    # pass it through encoder
    features = encoder(img_tensor_val)

    # start token by default to all the captions in batch
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    
    # keep generating the caption till max length
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


# In[34]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
print("Real caption's sentiment is:",image[1])
# opening the image
pil_img = Image(image[0])
display(pil_img)



