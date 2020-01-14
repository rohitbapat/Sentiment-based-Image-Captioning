
# coding: utf-8

# In[53]:

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import os
import json
from IPython.display import Image

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap #COCOEval





# In[8]:

annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)

annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_val2014.json'

PATH = os.path.abspath('.')+'/val2014/'

# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 150000 captions from the shuffled set
num_examples = 150000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]


# In[105]:


#for annot in annotations['annotations']:
#    if annot['image_id'] == 455108:
#        caption = '<start> ' + annot['caption'] + ' <end>'
#        print(caption)


# In[4]:


#import keras
#from keras import backend as K
#from keras.layers.core import Dense, Activation
#from keras.optimizers import Adam
#from keras.metrics import categorical_crossentropy
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#from keras.models import Model
#from keras.applications import imagenet_utils
#from keras.layers import Dense,GlobalAveragePooling2D
#from keras.applications import MobileNet
##from keras.applications.mobilenet_v2 import preprocess_input
#import numpy as np
#from IPython.display import Image
#from keras.optimizers import Adam


# In[5]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, image_path

# In[6]:


image_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# In[7]:


# Get unique images
#encode_train = sorted(set(img_name_vector))
#
## Feel free to change batch_size according to your system configuration
#image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
#image_dataset = image_dataset.map(
#  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
#
#for img, path in image_dataset:
#    batch_features = image_features_extract_model(img)
#    batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
#
#    for bf, p in zip(batch_features, path):
#        path_of_feature = p.numpy().decode("utf-8")# Find the maximum length of any caption in our dataset
#        np.save(path_of_feature, bf.numpy())


# In[9]:


def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                  
                oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# In[10]:


img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)


# In[11]:


print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))


# In[12]:


BATCH_SIZE = 400
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from MobilenetV2 is (81, 1280)
# These two variables represent that vector shape
features_shape = 1280
attention_features_shape = 81


# In[13]:


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


# In[14]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[15]:


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[16]:


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[17]:


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# In[18]:


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[19]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# In[34]:


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


# In[35]:


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# In[20]:

# define a checkpoint with all the variables predefined from the graph
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)


# In[29]:

# Load a checkpoint manager
ckpt_manager = tf.train.CheckpointManager(ckpt,"./checkpoints/without_sentiment_mobilenetV2", max_to_keep=5)



# In[31]:

# get the latest checkpoint from the path
status = ckpt.restore(tf.train.latest_checkpoint("./checkpoints/without_sentiment_mobilenetV2"))


# In[32]:

# Confirmation that all the variables can be traced
status.assert_existing_objects_matched()


# In[33]:

# train the graph variables from the checkpoint
tf.train.list_variables(tf.train.latest_checkpoint("./checkpoints/without_sentiment_mobilenetV2"))

# In[34]:
# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
# plot_attention(image, result, attention_plot)
# opening the image
pil_img = Image(img_name_val[rid])
display(pil_img)




# In[35]:
# make result caption list for mobilenet
result_caption_list = []

for rid in range(len(img_name_val)):
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)
    
    result = ' '.join(result)
    result_cap = ' '.join(result.split()[:-1])

    image_id = int(img_name_val[rid].split('/')[-1].split('_')[-1].split('.')[0])
    result_caption_list.append({"image_id":image_id, "caption": result_cap})

# In[37]:

# dump the results into a json file

with open('results_without_sentiment_mobilenetV2.json', 'w') as f:
    json.dump(result_caption_list, f)

# In[36]:

# we have used the PycocoEvalCap from github to evaluate all the captions generated by the model
# Link:https://github.com/tylin/coco-caption/tree/master/pycocoevalcap
coco = COCO("annotations/captions_val2014.json")
cocoRes = coco.loadRes('results_without_sentiment_mobilenetV2.json')


cocoEval = COCOEvalCap(coco, cocoRes)

cocoEval.params['image_id'] = cocoRes.getImgIds()


a = cocoEval.evaluate()
    
    
    