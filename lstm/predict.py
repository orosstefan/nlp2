import tensorflow as tf
from rouge import Rouge
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import warnings
from tensorflow.keras.models import load_model
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

rouge = Rouge()
pre = pd.read_csv('dataset/preprocessed.csv')
cleaned_headlines = pre['Summary'].tolist()
text = pre['Text'].tolist()
summary = ['_START_ ' + str(doc) + ' _END_' for doc in cleaned_headlines]
pre['cleaned_text'] = pd.Series(text)
pre['cleaned_summary'] = pd.Series(summary)

max_text_len = 100
max_summary_len = 15


cleaned_text = np.array(pre['cleaned_text'])
cleaned_summary = np.array(pre['cleaned_summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if (len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

post_pre = pd.DataFrame({'text': short_text, 'summary': short_summary})

# Add sostok and eostok
post_pre['summary'] = post_pre['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

# Split Data
x_tr, x_val, y_tr, y_val = train_test_split(np.array(post_pre['text']), np.array(post_pre['summary']), test_size=0.1,
                                            random_state=0, shuffle=True)
# Tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))
thresh = 4

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
print("Total Coverage of rare words:", (freq / tot_freq) * 100)

# Tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
x_tokenizer.fit_on_texts(list(x_tr))

# One hot encode sequences
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

# Add padding until max length
x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

# Vocabulary size
x_voc = x_tokenizer.num_words + 1

print("Size of vocabulary in X = {}".format(x_voc))
# Tokenizer for reviews on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))
thresh = 6

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
print("Total Coverage of rare words:", (freq / tot_freq) * 100)
y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
y_tokenizer.fit_on_texts(list(y_tr))

# One hot encoding
y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

# Add padding
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

# Vocabulary size
y_voc = y_tokenizer.num_words + 1
print("Size of vocabulary in Y = {}".format(y_voc))

ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)

print("Size of vocabulary from the w2v model = {}".format(x_voc))

K.clear_session()

latent_dim = 300
embedding_dim = 200

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

model = load_model('lstm/best_model.h5')
encoder_inputs = model.input[0]
decoder_inputs = model.input[1]
encoder_outputs, state_h, state_c = model.layers[6].output

encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim), name="input_5")
decoder_lstm = model.layers[7]

dec_emb_layer = model.layers[5]
dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])
decoder_dense = model.layers[8]

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])
encoder_model.summary()

def decode_sequence(input_seq):

    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if ((i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if (i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString


for i in range(0, 50):
    original = seq2summary(y_tr[i])
    predicted = decode_sequence(x_tr[i].reshape(1, max_text_len))
    print("Review:", seq2text(x_tr[i]))
    print("Original summary:", original)
    print("Predicted summary:", predicted)
    score_rouge = rouge.get_scores(predicted, original)
    print("ROUGE: ", score_rouge[0]['rouge-1']['f'])
    print("\n")
