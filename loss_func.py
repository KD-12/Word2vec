import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    """
    we take transpose of the embeddings and perform matrix multiplication with inputs
    softmax_numerator is nothing but exponential of this multiplication
    we perfom resum on axis 1 on softmax_numerator which adds cloumns of the softmax_numerator to produce softmax_denominator
    log of softmax_denominator is our B. Note we have added 1e-10 in log to avoid log(0) condition.
    log of softmax_numerator is our A. Note we have added 1e-10 in log to avoid log(0) condition.
    we return tf.subtract(B,A) which subtracts each element of A from B.
    """


    true_w_transpose=tf.transpose(true_w) #transpose of embeddings
    mult_u_o_t_v_c=tf.matmul(true_w_transpose,inputs)
    softmax_numerator=tf.exp(mult_u_o_t_v_c)

    softmax_denominator=tf.reduce_sum(softmax_numerator,1) #summing over the batchsize

    A=tf.log(softmax_numerator + 1e-10 )
    B=tf.log(softmax_denominator + 1e-10 )



    return tf.subtract(B,A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    """
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
    """
    u_o=tf.nn.embedding_lookup(weights, labels)

    shape1 = weights.shape
    embedding_size = shape1[1]
    shape2 = sample.shape
    k=shape2[0]
    u_o = tf.reshape(u_o, [-1, embedding_size])

    q_w=tf.nn.embedding_lookup(weights, sample)

    u_o_transpose=tf.transpose(u_o)
    q_w_transpose=tf.transpose(q_w)
    mult_u_o_t_v_c = tf.matmul(inputs, u_o_transpose)

    mult_q_w_t_v_c = tf.matmul(inputs, q_w_transpose)


    neg_bias=tf.nn.embedding_lookup(biases,sample)
    pos_bias=tf.nn.embedding_lookup(biases,labels)
    neg_bias = tf.reshape(neg_bias, [-1])
    pos_bias = tf.reshape(pos_bias, [-1])
    U_score = tf.nn.bias_add(mult_u_o_t_v_c , pos_bias)

    N_score = tf.nn.bias_add(mult_q_w_t_v_c , neg_bias)


    unigram_prob_tensor=tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    P_n = tf.gather(unigram_prob_tensor,sample)
    P_o = tf.gather(unigram_prob_tensor,labels)
    P_o = tf.reshape(P_o, [-1])





    first_term =tf.subtract(U_score, tf.log(tf.scalar_mul(k,P_o)+1e-10))
    second_term = tf.subtract(N_score,tf.log(tf.scalar_mul(k,P_n)+1e-10))



    sigmoid_1 = tf.sigmoid(first_term)
    sigmoid_2 = tf.sigmoid(second_term)

    A = tf.log(sigmoid_1 + 1e-10)
    B = tf.reduce_sum(tf.log(1-sigmoid_2 + 1e-10),1)





    return tf.add(A,B)
