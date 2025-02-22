# NeuralNetwork-HMA2
Question 1: Cloud Computing for Deep Learning
(a) Define elasticity and scalability in the context of cloud computing for deep learning.
(b) Compare AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio in terms of their deep learning capabilities. 

Question 2:Convolution Operations with Different Parameters
Task :Implement Convolution with Different Stride and Padding.
        VALID Padding → No padding, smaller output.
        SAME Padding → Keeps the output size similar to input by adding zeros.
        Stride 1 → Moves one step at a time.
        Stride 2 → Skips elements, reducing output size.

Question 3: CNN Feature Extraction with Filters and Pooling 
Task 1: Implement Edge Detection Using Convolution 
        Load the image in grayscale using cv2.imread().
        Apply the Sobel-X and Sobel-Y filters using cv2.filter2D().
        Display the results using matplotlib.pyplot.

Task 2: Implement Max Pooling and Average Pooling
        Create a random 4×4 matrix as input.
        Reshape it for TensorFlow (format: batch_size, height, width, channels).
        Apply max pooling (tf.nn.max_pool2d()) with a 2×2 filter and stride 2.
        Apply average pooling (tf.nn.avg_pool2d()) with a 2×2 filter and stride 2.
        Print results after converting tensors back to NumPy.

Summary
✅ Task 1: Edge Detection
        Used OpenCV (cv2) to apply Sobel filters for edge detection.
        Displayed original image, Sobel-X, and Sobel-Y images.
✅ Task 2: Pooling
        Used TensorFlow (tf.nn.max_pool2d & tf.nn.avg_pool2d).
        Demonstrated max pooling and average pooling on a random 4×4 matrix.

Question 4: Implementing and Comparing CNN Architectures
Task 1: Implement AlexNet Architecture
AlexNet is a deep convolutional neural network (CNN) designed for large-scale image classification. Below is a simplified implementation.
        Conv2D (96 filters, 11×11, stride=4): Extracts low-level features.
        MaxPooling (3×3, stride=2): Reduces spatial dimensions.
        Conv2D (256 filters, 5×5, stride=1, padding='same'): Extracts more detailed 
        features.
        MaxPooling (3×3, stride=2): Further reduces spatial size.
        Three more Conv2D layers: Increase feature depth.
        Final MaxPooling (3×3, stride=2): Reduces the feature map.
        Flatten: Converts the feature map into a vector.
        Dense (4096 neurons, ReLU) + Dropout(0.5): Fully connected layers to learn 
        high-level patterns.
        Output Layer (10 neurons, Softmax): Classifies into 10 categories.

Task 2: Implement a Residual Block and ResNet-like Model
ResNet uses skip connections to solve vanishing gradient problems in deep networks.
        Initial Conv2D (64 filters, 7×7, stride=2): Extracts features with a large 
        receptive field.
        Residual Block (2×):
        Two Conv2D layers (64 filters, 3×3).
        Skip connection adds input back to the output.
        Flatten: Converts the feature maps into a vector.
        Dense (128 neurons, ReLU): Fully connected layer.
        Output Layer (10 neurons, Softmax): Classifies into 10 categories.



