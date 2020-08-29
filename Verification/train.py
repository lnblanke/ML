# Train a CNN to recognize codes
# @Time: 8/19/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

import numpy as np
from PIL import Image
import random , cv2
import tensorflow.compat.v1 as tf
from captcha.image import ImageCaptcha

# Compat tensorflow to tf1
tf.disable_v2_behavior ()

number = [ '0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' ]
alphabet = [ 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , 'j' , 'k' , 'l' , 'm' , 'n' , 'o' , 'p' , 'q' , 'r' ,
             's' , 't' , 'u' , 'v' , 'w' , 'x' , 'y' , 'z' ]
ALPHABET = [ 'A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G' , 'H' , 'I' , 'J' , 'K' , 'L' , 'M' , 'N' , 'O' , 'P' , 'Q' , 'R' ,
             'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y' , 'Z' ]

set_size = 10

X = tf.placeholder ( tf.float32 , [ None , 60 * 160 ] )
Y = tf.placeholder ( tf.float32 , [ None , 4 * set_size ] )
keep_prob = tf.placeholder ( tf.float32 )

# Randomly generate verification codes
def generate_text ( set = number , length = 4 ) :
    image = ImageCaptcha ()

    text = ""

    for i in range ( length ) :
        char = random.choice ( set )
        text += char

    image.write ( text , "Code Training\\" + text + ".jpg" )

    image = np.array ( Image.open ( image.generate ( text ) ) )

    return text , image

# Use one-hot code to transform for text to vector
def text2vec ( text ) :
    vector = np.zeros ( 4 * set_size )

    for i , char in enumerate ( text ) :
        k = ord ( char ) - 48
        if k > 9 :
            k = ord ( char ) - 55

            if k > 35 :
                k = ord ( char ) - 61

        vector [ i * set_size + k ] = 1

    return vector

# Decrypt from vector to text
def vec2text ( vec ) :
    text = ""

    vector = vec [ 0 ]

    for i , char in enumerate ( vector ) :
        index = char % set_size

        if index < 10 :
            code = index + ord ( "0" )
        elif index < 35 :
            code = index - 10 + ord ( "A" )
        else :
            code = index - 35 + ord ( "a" )

        text += chr ( code )

    return text

# Generate next couple of codes
def getNext ( size = 128 ) :
    batchx = np.zeros ( [ size , 160 * 60 ] )
    batchy = np.zeros ( [ size , 4 * set_size ] )

    for i in range ( size ) :
        text , image = generate_text ()
        image = cv2.cvtColor ( image , cv2.COLOR_BGR2GRAY )

        batchx [ i , : ] = image.flatten ()
        batchy [ i , : ] = text2vec ( text )

    return batchx , batchy

# Create a conv with tensorflow
def cnn ( w_alpha = 0.01 , b_alpha = 0.1 ) :
    x = tf.reshape ( X , shape = [ -1 , 60 , 160 , 1 ] )
    w_c1 = tf.Variable ( w_alpha * tf.random_normal ( [ 3 , 3 , 1 , 32 ] ) )
    b_c1 = tf.Variable ( b_alpha * tf.random_normal ( [ 32 ] ) )

    conv1 = tf.nn.relu (
        tf.nn.bias_add ( tf.nn.conv2d ( x , w_c1 , strides = [ 1 , 1 , 1 , 1 ] , padding = 'SAME' ) , b_c1 ) )
    conv1 = tf.nn.max_pool ( conv1 , ksize = [ 1 , 2 , 2 , 1 ] , strides = [ 1 , 2 , 2 , 1 ] , padding = 'SAME' )
    conv1 = tf.nn.dropout ( conv1 , rate = 1 - keep_prob )

    w_c2 = tf.Variable ( w_alpha * tf.random_normal ( [ 3 , 3 , 32 , 64 ] ) )
    b_c2 = tf.Variable ( b_alpha * tf.random_normal ( [ 64 ] ) )
    conv2 = tf.nn.relu (
        tf.nn.bias_add ( tf.nn.conv2d ( conv1 , w_c2 , strides = [ 1 , 1 , 1 , 1 ] , padding = 'SAME' ) , b_c2 ) )
    conv2 = tf.nn.max_pool ( conv2 , ksize = [ 1 , 2 , 2 , 1 ] , strides = [ 1 , 2 , 2 , 1 ] , padding = 'SAME' )
    conv2 = tf.nn.dropout ( conv2 , rate = 1 - keep_prob )

    w_c3 = tf.Variable ( w_alpha * tf.random_normal ( [ 3 , 3 , 64 , 64 ] ) )
    b_c3 = tf.Variable ( b_alpha * tf.random_normal ( [ 64 ] ) )
    conv3 = tf.nn.relu (
        tf.nn.bias_add ( tf.nn.conv2d ( conv2 , w_c3 , strides = [ 1 , 1 , 1 , 1 ] , padding = 'SAME' ) , b_c3 ) )
    conv3 = tf.nn.max_pool ( conv3 , ksize = [ 1 , 2 , 2 , 1 ] , strides = [ 1 , 2 , 2 , 1 ] , padding = 'SAME' )
    conv3 = tf.nn.dropout ( conv3 , rate = 1 - keep_prob )

    w_d = tf.Variable ( w_alpha * tf.random_normal ( [ 8 * 20 * 64 , 1024 ] ) )
    b_d = tf.Variable ( b_alpha * tf.random_normal ( [ 1024 ] ) )
    dense = tf.reshape ( conv3 , [ -1 , w_d.get_shape ().as_list () [ 0 ] ] )
    dense = tf.nn.relu ( tf.add ( tf.matmul ( dense , w_d ) , b_d ) )
    dense = tf.nn.dropout ( dense , rate = 1 - keep_prob )

    w_out = tf.Variable ( w_alpha * tf.random_normal ( [ 1024 , 4 * set_size ] ) )
    b_out = tf.Variable ( b_alpha * tf.random_normal ( [ 4 * set_size ] ) )
    out = tf.add ( tf.matmul ( dense , w_out ) , b_out )

    return out

output = cnn ()

loss = tf.reduce_mean ( tf.nn.sigmoid_cross_entropy_with_logits ( logits = output , labels = Y ) )
optimizer = tf.train.AdamOptimizer ( learning_rate = 1e-3 ).minimize ( loss )
predict = tf.reshape ( output , [ -1 , 4 , set_size ] )
max_predict = tf.argmax ( predict , 2 )
max_real = tf.argmax ( tf.reshape ( Y , [ -1 , 4 , set_size ] ) , 2 )

correct = tf.equal ( max_predict , max_real )

accuracy = tf.reduce_mean ( tf.cast ( correct , tf.float32 ) )

saver = tf.train.Saver ()

# Train conv
with tf.Session () as sess :
    sess.run ( tf.global_variables_initializer () )

    step = -1

    while 1 :
        step += 1

        batchx , batchy = getNext ( 64 )

        _ , loss_ = sess.run ( [ optimizer , loss ] , feed_dict = { X : batchx , Y : batchy , keep_prob : .75 } )

        print ( "Step: %d  Loss: %f" % (step , loss_) )

        if step % 100 == 0 :
            batchx , batchy = getNext ( 100 )

            acc = sess.run ( accuracy , feed_dict = { X : batchx , Y : batchy , keep_prob : .75 } )

            print ( "Accuracy: %f" % (acc) )

            if acc > .9 :
                saver.save ( sess , "model/crack_capcha.model99" , global_step = step )
                break

    # Test

    step = 0
    correct = 0
    count = 100

    while step < count :
        text , img = generate_text ()

        img = cv2.cvtColor ( img , cv2.COLOR_BGR2GRAY )
        img = img.flatten ()

        predict = tf.math.argmax ( tf.reshape ( output , [ - 1 , 4 , set_size ] ) , 2 )
        label = sess.run ( predict , feed_dict = { X : [ img ] , keep_prob : 1 } )

        predict_text = vec2text ( label )

        print ( "step:{} 真实值: {}  预测: {}  预测结果: {}".format ( str ( step ) , text , predict_text ,
            "正确" if text.lower () == predict_text.lower () else "错误" ) )

        if text.lower () == predict_text.lower () :
            correct += 1

        step += 1

print ( "测试总数: {} 测试准确率: {}".format ( str ( count ) , str ( correct / count ) ) )
