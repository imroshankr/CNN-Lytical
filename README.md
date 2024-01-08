# CNN-Lytical
This project was divided into three key assignments, each progressively building on the complexity of the task and the neural network architecture. Here's an overview:

**MNIST Digit Classification**: We began with the classic problem of classifying handwritten digits using the MNIST dataset. The images were 28x28 pixels in grayscale, and the task was to categorize them into one of ten classes representing the digits 0 through 9. 
    To tackle this, I designed and implemented a feedforward neural network with multiple layers. The architecture involved linear layers interspersed with ReLU activation 
    functions and a final softmax layer to predict the digit classes. We used a cross-entropy loss function for training and experimented with different optimizers, such as 
    SGD and Adam.
    A significant challenge here was preventing overfitting, given the relatively simple nature of the images. We employed techniques like dropout and batch normalization to 
    improve generalization. Moreover, hyperparameter tuning was crucial to balance model complexity and training data characteristics, aiming for high accuracy on unseen 
    test data.
    


**CIFAR-10 Image Classification**: Next, we dealt with the CIFAR-10 dataset, which contains 32x32 pixel color images across 10 different categories, including animals and vehicles. The complexity here was notably higher than MNIST.
    For this, a convolutional neural network, or CNN, was the architecture of choice. CNNs leverage convolutional layers that apply filters to the images, capturing spatial 
    hierarchies and features essential for classification tasks. We used pooling layers to reduce dimensionality and fully connected layers for the final classification.

    The primary challenge was the complexity of the images. To enhance the robustness of the model, we implemented data augmentation strategies, such as random cropping and 
    flipping, which effectively increased the diversity of the training data. Additionally, learning rate annealing was crucial to ensure the model converged to a good solution
