# Neural-network-based-Classifier
The multilayer perceptron (MLP) is a type of deep feedforward neural network (DFN) that
maps the inputs to the outputs. It contains multiple fully connected layers (FCs), the Neurons
within these layers use nonlinear activation functions, enabling the network to capture
complex patterns in the data which makes MLPs powerful for tasks like classification,
regression, and pattern recognition.
# MLP architecture:
The model consists of three dense layers with a Rectifier Linear Unit (ReLU) , each followed by a dropout rate and an L2 regularization for a performance of
models by reducing overfitting. The output layer uses a sigmoid activation with a number of
units equal to the number of output classes which is equal to 64. For training and
optimization, the model utilizes the Adam optimizer with an initial learning rate of 0.0001 and
binary cross-entropy loss.
