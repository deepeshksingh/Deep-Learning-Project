"""
Train convolutional network for sentiment analysis.
"""
import numpy as np
import data_helpers
from w2v import train_word2vec
from keras.models import load_model

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
np.random.seed(0)

dataSource = "kerasDataSet"
embeddingDim = 50
filterSizes = (3, 8)
numFilters = 10
dropoutProb = (0.5, 0.8)
hiddenDims = 50
batchSize = 64
numEpochs = 10
sequenceLength = 400
maxWords = 5000
minWordCount = 1
contextWindow = 10

def load_data(dataSource):
    assert dataSource in ["kerasDataSet", "local_dir"], "Unknown data source"
    if dataSource == "kerasDataSet":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxWords, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequenceLength, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequenceLength, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv

print("Loading Data")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(dataSource)

if sequenceLength != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequenceLength = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embeddingDim,
                                       minWordCount=minWordCount, context=contextWindow)

input_shape = (sequenceLength,)

model_input = Input(shape=input_shape)
z = Embedding(len(vocabulary_inv), embeddingDim, input_length=sequenceLength, name="embedding")(model_input)

z = Dropout(dropoutProb[0])(z)

conv_blocks = []
for sz in filterSizes:
    conv = Convolution1D(filters=numFilters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropoutProb[1])(z)
z = Dense(hiddenDims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

model.fit(x_train, y_train, batchSize=batchSize, epochs=numEpochs,
          validation_data=(x_test, y_test), verbose=2)

model.save('my_model.h5')

del model

model = load_model('my_model.h5')

print("X TEST", x_test)
print("Y Test", y_test)
test2 = model.predict(x_test)
print(test2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))