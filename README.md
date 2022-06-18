# NLP sentiment analysis of IMDB reviews dataset using a LSTM recurrent neural network

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We have a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, We are predicting the number of positive and negative reviews using either classification or deep learning algorithms.


We start by importing the main libraires that we will use:

the re module (for regular expression matching operations)
the nltk toolkit (for natural language operations)
the random module (for random number generation)
the numpy library (for arrays operations)
the pandas library (for data analysis)
the scipy.stats module (for statistics)
the seaborn library (for statistical data visualization)
the matplotlib.pyplot interface (for MATLAB-like plots)

Next, We've preprocessed the Data using Regex.
We perform Data Splitting and Tokenization . We split our DataFrame into a training and test lists. We use the train_test_split() function from the sklearn.model_selection module which allow to perform the splitting randomly with respect to the index of the DataFrame.
Next, we use the Tokenizer class from keras.preprocessing.text module to create a dictionary of the "dict_size" most frequent words present in the reviews (a unique integer is assigned to each word), and we print some of its attributes. The index of the Tokenizer is computed the same way no matter how many most frequent words we use later.

We use the texts_to_sequences() function of the Tokenizer class to convert the training reviews and test reviews to lists of sequences of integers.

We'd used the pad_sequences() function from keras.preprocessing.sequence module, we transform train and test tokens into 2D numpy arrays of shape.

We've imported some classes from Keras:

the Sequential class from the keras.models API (to group a linear stack of layers into a model)
the Embedding class from the keras.layers API (to turn positive integers (indexes) into dense vectors of fixed size)
the LSTM class from the keras.layers API (to apply a long short-term memory layer to an input)
the Dropout class from the keras.layers API (to apply dropout to an input)
the Dense class from the keras.layers API (to apply a regular densely-connected NN layer to an input)

In the LSTM model, we set the following parameters:

the output dimension of the Embedding layer (dimension of the vector space containing the word embeddings) is "output_dim"
the number of units of the LSTM layer is "units_lstm"
the dropout rate of the Dropout layer is "r"
the activation function of the final Dense layer is sigmoid (this is a natural choice since the output of the model should be a number between 0, for negative reviews, and 1, for positive reviews)

Then, We compile the model for training with the following parameters:

adam as optimizer to use during training process (a combination of gradient descent with momentum and RMSP)
binary cross-entropy (bce) between true labels and predicted labels as loss to minimise during training process
accuracy as metric to display during training process (how often predicted labels equal true labels)

Next, we test the trained model on a randomly chosen review from the test set. We displayed the original review, the sentiment predicted by the model with its probability, and the original (correct) sentiment

