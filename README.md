# Word2vec
  Generate batch for skip-gram model (word2vec_basic.py)
  Two loss functions to train word embeddings (loss_func.py)
  Parameter Tuning for word embeddings
  Applying learned word embeddings to word analogy task (word_analogy.py)

  word2vec_basic.py:
Usage:
  python word2vec_basic.py [cross_entropy | nce]
generate_batch(data, batch_size, num_skips, skip_window):
We use this function to generate batchsize of 128.
This function is called again and again in the train function and produces batches of size 128 for the corpus of size 100000.
It takes following arguments:
data which is the whole corpus.
batch_size variable which can be set in the main function.
num_skips which denotes number of time same center word has to be repeated for the outer words equal to skipwindow
  on left and right side of center word.
skipwindow defines the number of outer words on left and right side of center word.

we create window size as skip_window*2+1 (words on left and right plus center word)

first for loop creates a batchword_list which stores all words for the batch which are obtained by moving the global variable data_index
on the corpus.
we use this (data_index+1)%(len(data)) to bring back data_index to zero if it ever increases than the size of the corpus.

second for loop creates a list (list2) which store batch_size/num_skips number of lists each containing  indices for each window
of length  window size

third for loop creates list batch_ind which stores indices of center word repeated num_skips times. This list first center word starts from
value of skipwindow.

The fourth final for loop selects center word from batch_ind and corresponding outer word from list2 which is not a center word.

loss_func.py
This function is called from word2vec_basic and comprises of two loss functions forming two models.

cross_entropy_loss(inputs, true_w)
inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
we claculate A and B using formulas: A = log(exp({u_o}^T v_c)) and B = log(\sum{exp({u_w}^T v_c)})
as follows:
we take transpose of the embeddings and perform matrix multiplication with inputs
softmax_numerator is nothing but exponential of this multiplication
we perfom resum on axis 1 on softmax_numerator which adds cloumns of the softmax_numerator to produce softmax_denominator
log of softmax_denominator is our B. Note we have added 1e-10 in log to avoid log(0) condition.
log of softmax_numerator is our A. Note we have added 1e-10 in log to avoid log(0) condition.
we return tf.subtract(B,A) which subtracts each element of A from B.

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
biases: Biases for nce loss. Dimension is [Vocabulary, 1].
labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
samples: Word_ids for negative samples. Dimension is [num_sampled].
unigram_prob: Unigram probability. Dimesion is [Vocabulary].

we find outer words by using nce_weights and labels and similarly create noise words by using nce_weights and samples
Next we compute score of outer words and noise words which is multiplcation of transpose of above found matrices with input matrix.
we find negative bias by using biases and samples and outer word bias  using biases and labels this is done to make the shape
compatible so that addition of above found values can be done.

We convert the unigram_prob list given to us in to a tensor and use tf.gather to find the probability of negative words using
samples  and probability of true outer word using labels.

we then use the above found values to compute the formulas of provided pdf as follows
first_term =tf.subtract(U_score, tf.log(tf.scalar_mul(k,P_o)+1e-10))
second_term = tf.subtract(N_score,tf.log(tf.scalar_mul(k,P_n)+1e-10))
 we pass this terms to sigmoid function and find A as log of value obtained by passing first term to sigmoid.
 We take reduce sum of value obtained by passing second term to sigmoid. this adds all columns.
 We take log of this reduce subtracted from 1 to create B
 we pass A+B and put minus side outside the reduce sum in word2vec_basic file from where this function was called.

 word_analogy.py
 def cos_sim(v1,v2):
 takes two vectors as input and finds it's cosine similarity.
 def simillar_words(words,embeddings, v2):
 finds top similar words to the vector v2 by comparing it with all words of the entire data.
 we read the word_analogy_dev file line by line and separate examples and choices on '||' and use these to produce vectors to
 perform word analogy we subtract vectors of example group like left2 - left1 and take average of all example of each line and then find
 cosine simillarity with each of the words on the right side like cosine similarity between avg vector and each of the choice by forming
 vector of choices by substracting right2 and right in my program stored as right2 and words.
 in the output file we print on each line in this format
 <pair1> <pair2> <pair3> <pair4> <least_illustrative_pair> <most_illustrative_pair>
 the file name in program is word_anlogy_dev_predictions.txt
the prediction for different model can be found by commenting the model.
Care  has to be taken to delete this file before next execution as this file is opened in append mode.
