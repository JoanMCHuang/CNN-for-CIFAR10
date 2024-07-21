import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train_gray = tf.image.rgb_to_grayscale(x_train)
x_test_gray = tf.image.rgb_to_grayscale(x_test)

x_train_gray = x_train_gray / 255.0
x_test_gray = x_test_gray / 255.0

x_train_color = x_train / 255.0
x_test_color = x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
    
    return model

# Create and train grayscale model
model_gray = create_model((32,32,1), 10)
model_gray.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_gray.fit(x_train_gray, y_train, epochs=10, batch_size=64, validation_data=(x_test_gray, y_test))

# Create and train color model
model_color = create_model((32,32,3), 10)
model_color.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_color.fit(x_train_color, y_train, epochs=10, batch_size=64, validation_data=(x_test_color, y_test))

# Evaluate grayscale model
loss_gray, acc_gray = model_gray.evaluate(x_test_gray, y_test)
print('Grayscale model accuracy:', acc_gray)

# Evaluate color model
loss_color, acc_color = model_color.evaluate(x_test_color, y_test)
print('Color model accuracy:', acc_color)

