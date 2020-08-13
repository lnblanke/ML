import mnist
from CNNclass import Conv

train_img = mnist.train_images()
train_lab = mnist.train_labels()

conv = Conv ( 8 )

output = conv.forward ( train_img [ 0 ] )

print ( output.shape )