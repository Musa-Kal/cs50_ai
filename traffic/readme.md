My intuition for this model shape was to start with a convolution layer then that is feed input layer directly and then through max pooling I reduce the image size and feed it into another input convolution layer then another max pooling to reduce the image size further and then feed it into a two dense layers, last of which being the output. The thought behind this was that the 1st convolution layer will allow the model to learn low level features like edges and corders and the pooling and next convolution layer combined will allow to model to learn higher level features like symbols and patterns on the sign basically having the same effect as zooming out and looking at the bigger picture and the last to dense layer are for the model to learn any remaining patterns or connections.

model 1:
first convolution layer has more filter then the second one:
1st layer - 64 filter convolution layer
2nd layer - 32 filter convolution layer
3rd layer - 128 dense layer
4th layer - 43 unit dense layer
lowest loss value achieved by the model was 0.0015

model 2:
first convolution layer has less filter then the second one:
1st layer - 30 filter convolution layer
2nd layer - 60 filter convolution layer
3rd layer - 128 dense layer
4th layer - 43 unit dense layer
lowest loss value achieved by the model was 6.1721e-04 which is the lowest of all models

model 3:
first convolution layer has less filter then the second one and there is only 1 dense layer which is the output layer:
1st layer - 64 filter convolution layer
2nd layer - 32 filter convolution layer
3th layer - 43 unit dense layer
lowest loss value achieved by the model was 0.0026 