import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Load CSV files
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

# 2. Separate features and labels
train_x = train_df.drop('label', axis=1).values
train_y = train_df['label'].values

test_x = test_df.drop('label', axis=1).values
test_y = test_df['label'].values

# 3. Normalize pixel values and reshape
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)

# 4. Build the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 5. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(train_x, train_y, epochs=5, validation_split=0.1)

# 7. Evaluate
loss, acc = model.evaluate(test_x, test_y)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# 8. Prediction
labels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

predictions = model.predict(test_x[:1])
predicted_label = labels[np.argmax(predictions)]

print("Predicted Label:", predicted_label)
plt.imshow(test_x[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()



#optional but running





import numpy as np
import matplotlib.pyplot as plt

# Define label names
labels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# Choose the index of the test image you want to check
index = 9  # You can change this to any number between 0 and len(test_x)-1

# Make a prediction
prediction = model.predict(test_x[index:index+1])
predicted_label = labels[np.argmax(prediction)]

# Get the actual label
actual_label = labels[test_y[index]]

# Show both predicted and actual
print("Predicted Label:", predicted_label)
print("Actual Label   :", actual_label)

# Show the image
plt.imshow(test_x[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}")
plt.axis('off')
plt.show()




for i in range(5):
    prediction = model.predict(test_x[i:i+1])
    predicted_label = labels[np.argmax(prediction)]
    actual_label = labels[test_y[i]]

    print(f"Image {i+1}")
    print("Predicted Label:", predicted_label)
    print("Actual Label   :", actual_label)
    plt.imshow(test_x[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}")
    plt.axis('off')
    plt.show()




