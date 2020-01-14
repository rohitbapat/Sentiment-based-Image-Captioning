
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
import json
# package to get a progress bar for saving vectors of each image in a .npy file
from tqdm import tqdm
from IPython.display import Image 

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap #COCOEval




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
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []
sentiment = []

with open("senticap_dataset.json", 'r') as f:
    senti_annotations = json.load(f)
    
senti_data = {"annotations":[]}
for annot in senti_annotations['images']:
    img_id = int(annot['filename'].split('_')[-1].split('.')[0])
    for sen in annot['sentences']:
        senti_data["annotations"].append({"image_id":img_id,"caption":sen['raw'],"sentiment":sen["sentiment"]}) 

res = [v['image_id'] for v in senti_data['annotations']]
res = set(res)

for entry in senti_data['annotations']:
    caption = '<start> ' + entry['caption'] + ' <end>'
    all_captions.append(caption)
    img_id = entry['image_id']
    full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (img_id)
    senti = entry['sentiment']
    if senti == 0:
        senti = -1
    all_img_name_vector.append([full_coco_image_path,senti])
    

for annot in annotations['annotations']:
    image_id = annot['image_id']
    if image_id in res:
        caption = '<start> ' + annot['caption'] + ' <end>'
        full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
        all_img_name_vector.append([full_coco_image_path,0])
        all_captions.append(caption)
        

#for annot in annotations['annotations']:
#    caption = '<start> ' + annot['caption'] + ' <end>'
#    image_id = annot['image_id']
#    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
#
#    all_img_name_vector.append(full_coco_image_path)
#    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 20000 captions from the shuffled set
num_examples = 20000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]



# In[7]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# In[8]:


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# 
# In[7]:
sentiment = img_name_vector
img_name_vector = [x[0] for x in img_name_vector]


# In[10]:


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[11]:


# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[12]:


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# In[13]:


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[14]:


# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[15]:


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# ## Split the data into training and testing

# In[14]:
#print(img_name_vector[1])
#print(sentiment[1])
#sentiment_dict = {x[0]:x[1] for x in sentiment}

# In[16]:


# Create training and validation sets using an 80-20 split
#img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
#                                                                    cap_vector,
#                                                                    test_size=0.2,
#                                                                    random_state=0)
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

# Shuffle and batch
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


# In[22]:


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


# In[23]:


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


# In[24]:


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[25]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# In[32]:


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image[0])[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    new_img_tensor_val = np.zeros([img_tensor_val.shape[0],img_tensor_val.shape[1],(img_tensor_val.shape[2]+4)],dtype=np.float32)
    new_img_tensor_val[:,:,:2048] = img_tensor_val
    if image[1] == 0:
        new_img_tensor_val[:,:,2048:] = [0,1,0,0]
    elif image[1] == 1:
        new_img_tensor_val[:,:,2048:] = [1,0,0,1]
    elif image[1] == -1:
        new_img_tensor_val[:,:,2048:] = [0,0,1,2]
    
    img_tensor_val = new_img_tensor_val        

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


# In[33]:


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image[0]))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.show()
    
    
# In[20]:

# define a checkpoint with all the variables predefined from the graph
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)


# In[29]:

# Load a checkpoint manager
ckpt_manager = tf.train.CheckpointManager(ckpt,"./checkpoints/sentiment", max_to_keep=5)


# In[31]:

# get the latest checkpoint from the path
status = ckpt.restore(tf.train.latest_checkpoint("./checkpoints/sentiment"))


# In[32]:

# Confirmation that all the variables can be traced
status.assert_existing_objects_matched()


# In[33]:

# train the graph variables from the checkpoint
tf.train.list_variables(tf.train.latest_checkpoint("./checkpoints/sentiment"))



# In[37]:
# Trying with a random RID and genearting all its sentiment data , we also get its true sentiments
# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]

real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
# The model gives result based on the sentiment(0,1,-1) provided from input
result_0, attention_plot = evaluate([image[0],0])
result_1, attention_plot = evaluate([image[0],1])
result__1, attention_plot = evaluate([image[0],-1])

# Print all combinations
print ('Real Caption:', real_caption)
print ('Prediction Neutral Caption:', ' '.join(result_0))
print ('Prediction Positive Caption:', ' '.join(result_1))
print ('Prediction Negative Caption:', ' '.join(result__1))
print("Real caption's sentiment is:",image[1])


# display the image
print(image[0])
pil_img = Image(image[0])
display(pil_img)


# print all its related captions present as ground truth
img_id = int(image[0].split('_')[-1].split('.')[0])

for element in senti_data:
    for index in range(len(senti_data[element])):
        if senti_data[element][index]['image_id'] == img_id:
            for key in senti_data[element][index]:
                print(key)
                print(senti_data[element][index][key])
                
image_id = img_id
    
for index, element in enumerate(sentiment):
    if element[0] == image[0] and element[1] == 0:
        actual_neutral_cap = ' '.join(train_captions[index].split()[1:len(train_captions[index].split())-1])
        print("Actual Neutral Caption is:",actual_neutral_cap)
        
        
# In[35]:
           
# code block to generate and evaluate all positive sentiment captions
# For positive captions
result_caption_list_positive = []
actual_caption_list_positive = []

images_list_positive = []


for rid in tqdm(range(len(img_name_val))):
    image = img_name_val[rid]
    image_id = int(image[0].split('/')[-1].split('_')[-1].split('.')[0])
    
    
    # get all the captions from the annotations file 
    for index in range(len(senti_data['annotations'])):
        # if the image_id and the target sentiment matches then populate it into target caption and generate the model based caption for it
        if senti_data['annotations'][index]['image_id'] == image_id and senti_data['annotations'][index]['sentiment'] == 1:
            actual_caption_list_positive.append({"image_id":image_id,"id":rid,"caption":senti_data['annotations'][index]['caption']})
            images_list_positive.append({"file_name":image[0].split('/')[-1],"id":image_id})
            
            # generate a positive caption for the image
            result_positive, attention_plot = evaluate([image[0],1])
            result_positive = ' '.join(result_positive)
            result_positive_cap = ' '.join(result_positive.split()[:-1])
            
            # push image_id and generated caption into another json type variable
            result_caption_list_positive.append({"image_id":image_id, "caption": result_positive_cap})


# In[37]:

# dump the results to json file

with open('sentiment_results_positive.json', 'w') as f:
    json.dump(result_caption_list_positive, f)

actual_caption_list_positive_dict = {"images":images_list_positive,"annotations":actual_caption_list_positive}

# dump the ground truth to json file
with open('sentiment_actual_positive.json', 'w') as f:
    json.dump(actual_caption_list_positive_dict, f)
    
# we have used the PycocoEvalCap from github to evaluate all the captions generated by the model
# Link:https://github.com/tylin/coco-caption/tree/master/pycocoevalcap


# Convert to COCO objects 
coco = COCO("sentiment_actual_positive.json")
# COCO result object
cocoRes = coco.loadRes('sentiment_results_positive.json')

# evaluate the objects
cocoEval = COCOEvalCap(coco, cocoRes)

cocoEval.params['image_id'] = cocoRes.getImgIds()

a = cocoEval.evaluate()

# In[35]:
                
# For negative captions
result_caption_list_negative = []
actual_caption_list_negative = []

images_list_negative = []

for rid in tqdm(range(len(img_name_val))):
    image = img_name_val[rid]
    image_id = int(image[0].split('/')[-1].split('_')[-1].split('.')[0])
    
    for index in range(len(senti_data['annotations'])):
        # print(senti_data['annotations'][index])
        if senti_data['annotations'][index]['image_id'] == image_id and senti_data['annotations'][index]['sentiment'] == 0:
            actual_caption_list_negative.append({"image_id":image_id,"id":rid,"caption":senti_data['annotations'][index]['caption']})
            images_list_negative.append({"file_name":image[0].split('/')[-1],"id":image_id})
    
            result_negative, attention_plot = evaluate([image[0],-1])
            result_negative = ' '.join(result_negative)
            result_negative_cap = ' '.join(result_negative.split()[:-1])
            
            result_caption_list_negative.append({"image_id":image_id, "caption": result_negative_cap})

# In[37]:

with open('sentiment_results_negative.json', 'w') as f:
    json.dump(result_caption_list_negative, f)
    
actual_caption_list_negative_dict = {"images":images_list_negative,"annotations":actual_caption_list_negative}

with open('sentiment_actual_negative.json', 'w') as f:
    json.dump(actual_caption_list_negative_dict, f)
    
  
    
coco = COCO("sentiment_actual_negative.json")
cocoRes = coco.loadRes('sentiment_results_negative.json')


cocoEval = COCOEvalCap(coco, cocoRes)

cocoEval.params['image_id'] = cocoRes.getImgIds()


a = cocoEval.evaluate()

# In[35]:
                
# For neutral captions
result_caption_list_neutral = []
actual_caption_list_neutral = []

images_list_neutral = []

for rid in tqdm(range(len(img_name_val))):
    image = img_name_val[rid]
    image_id = int(image[0].split('/')[-1].split('_')[-1].split('.')[0])
    
    for index, element in enumerate(sentiment):
        
        if element[0] == image[0] and element[1] == 0:
            actual_neutral_cap = ' '.join(train_captions[index].split()[1:len(train_captions[index].split())-1])
            actual_caption_list_neutral.append({"image_id":image_id,"id":rid,"caption":actual_neutral_cap})
            images_list_neutral.append({"file_name":image[0].split('/')[-1],"id":image_id})
    
            result_neutral, attention_plot = evaluate([image[0],0])
            result_neutral = ' '.join(result_neutral)
            result_neutral_cap = ' '.join(result_neutral.split()[:-1])
            
            result_caption_list_neutral.append({"image_id":image_id, "caption": result_neutral_cap})


# In[37]:

with open('sentiment_results_neutral.json', 'w') as f:
    json.dump(result_caption_list_neutral, f)

actual_caption_list_neutral_dict = {"images":images_list_neutral,"annotations":actual_caption_list_neutral}

with open('sentiment_actual_neutral.json', 'w') as f:
    json.dump(actual_caption_list_neutral_dict, f)

coco = COCO("sentiment_actual_neutral.json")
cocoRes = coco.loadRes('sentiment_results_neutral.json')


cocoEval = COCOEvalCap(coco, cocoRes)

cocoEval.params['image_id'] = cocoRes.getImgIds()


a = cocoEval.evaluate() 


