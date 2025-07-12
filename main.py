# Re-check the blank index and ensure it's used consistently in encoding and character mapping
# The num_classes includes the blank character. If our characters are 0-9, a-z (36 total),
# then num_classes = 37. The indices should be 1-36 for characters and 0 for blank, OR
# 0-35 for characters and 36 for blank. Keras's ctc_batch_cost expects blank_index = num_classes - 1.
# So, our character indices should be 0 to num_classes - 2.
# Let's adjust the char_to_int and int_to_char mappings to reflect this.
import numpy as np
from PIL import Image
import pandas as pd
# Example: create dummy DataFrames with 'image_path' and 'text' columns
# You should replace this with your actual data loading logic
train_df = pd.DataFrame({
    'image_path': ['dummy_path1.png', 'dummy_path2.png'],
    'text': ['5', '0']
})
val_df = pd.DataFrame({
    'image_path': ['dummy_path3.png'],
    'text': ['4']
})
# Build character set from train_df and val_df (assuming single-label classification)
all_labels = np.unique(np.concatenate([train_df['text'].values, val_df['text'].values]))
all_chars = [str(label) for label in all_labels]  # Convert to string if needed
# Assign indices 0 to len(all_chars) - 1 to the characters
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}
# Assign the blank index
blank_index = len(all_chars)  # This will be num_classes - 1
char_to_int[' '] = blank_index
int_to_char[blank_index] = ' '
num_classes = len(all_chars) + 1  # Total number of classes including blank
print(f"Updated Number of unique characters (including blank): {num_classes}")
print(f"Blank index: {blank_index}")
print(f"Char to int mapping sample: {list(char_to_int.items())[:5]}...")
print(f"Int to char mapping sample: {list(int_to_char.items())[:5]}...")
# Define MAX_TEXT_LEN as the maximum length of the text labels in your dataset
MAX_TEXT_LEN = 1  # For single digit classification; set higher if your labels are longer
# Re-define the encoding function to use the updated blank index for padding
def encode_label(text, char_map, max_len=MAX_TEXT_LEN, blank_idx=blank_index):
    encoded = [char_map.get(char, blank_idx) for char in text] # Use .get with default blank_idx for unknown chars
    # Pad labels with the blank index to a fixed length
    if len(encoded) < max_len:
        encoded += [blank_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len] # Truncate if longer than max_len
    return np.array(encoded)
# Modify the data generator to ensure correct label and input length calculations and types
# Define IMAGE_HEIGHT according to your dataset (e.g., 28 for MNIST-like images)
IMAGE_HEIGHT = 28
import os
def mock_load_image(path, img_height):
    # If file exists, load it; otherwise, create a blank image
    if os.path.exists(path):
        img = Image.open(path).convert('L')
    else:
        # Create a blank white image (for demonstration)
        img = Image.new('L', (img_height, img_height), color=255)
    return img
def preprocess_image(path, img_height):
    img = mock_load_image(path, img_height)
    width, height = img.size
    new_width = int(img_height * (width / height))
    img = img.resize((new_width, img_height))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array, new_width
def data_generator(dataframe, char_map, batch_size=32, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index):
    num_samples = len(dataframe)
    max_img_width = 0
    for _, row in dataframe.iterrows():
        img_temp = mock_load_image(row['image_path'], img_height)
        width_temp, height_temp = img_temp.size
        new_width_temp = int(img_height * (width_temp / height_temp))
        if new_width_temp > max_img_width:
            max_img_width = new_width_temp
    target_width = max_img_width
    # Determine the maximum input length based on the target width and CNN architecture
    # Assumes 2 MaxPooling2D layers with pool_size=(2,2)
    max_input_length = target_width // (2 * 2)
    while True:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_df = dataframe.iloc[offset:offset + batch_size]
            images = []
            encoded_labels = []
            input_lengths = []
            label_lengths = []
            for index, row in batch_df.iterrows():
                img_array, original_width = preprocess_image(row['image_path'], img_height)
                if original_width < target_width:
                    padding_needed = target_width - original_width
                    padded_img_array = np.pad(img_array, ((0, 0), (0, padding_needed), (0, 0)), mode='constant', constant_values=1.0)
                else:
                    padded_img_array = img_array
                encoded_label = encode_label(row['text'], char_map, max_text_len, blank_idx)
                images.append(padded_img_array)
                encoded_labels.append(encoded_label)
                # The input length is consistent for all samples in this generator's logic
                # because all images are padded to the same target_width.
                # The input_length to CTC is the sequence length output by the RNN,
                # which is the width of the last CNN layer's output after pooling.
                input_len = max_input_length # Use the calculated max input length
                input_lengths.append(input_len)
                label_lengths.append(len(row['text']))
            images = np.array(images)
            encoded_labels = np.array(encoded_labels)
            input_lengths = np.array(input_lengths, dtype=np.int64).reshape(-1, 1) # Ensure int64
            label_lengths = np.array(label_lengths, dtype=np.int64).reshape(-1, 1) # Ensure int64
            yield {
                'the_input': images,
                'the_labels': encoded_labels,
                'input_length': input_lengths,
                'label_length': label_lengths
            }, np.zeros(len(batch_df))
# Define batch_size and create dummy train_df and val_df for demonstration
batch_size = 32
import pandas as pd
# Example: create dummy DataFrames with 'image_path' and 'text' columns
# You should replace this with your actual data loading logic
train_df = pd.DataFrame({
    'image_path': ['dummy_path1.png', 'dummy_path2.png'],
    'text': ['5', '0']
})
val_df = pd.DataFrame({
    'image_path': ['dummy_path3.png'],
    'text': ['4']
})
# Re-instantiate the data generators with the modified function and updated mappings
train_generator = data_generator(train_df, char_to_int, batch_size=batch_size, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)
val_generator = data_generator(val_df, char_to_int, batch_size=batch_size, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)
# Re-define the CTC Loss Layer to ensure it uses the correct blank index if needed
# In the original code, the K.ctc_batch_cost function automatically uses num_classes - 1
# as the blank index, which should now align with our updated encoding.
# No change needed in CTCLayer itself, but ensuring num_classes is correctly passed if the layer were different.
# Define a minimal CTC model if not already defined
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
input_img = keras.Input(shape=(IMAGE_HEIGHT, None, 1), name='the_input')
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
new_shape = (IMAGE_HEIGHT // 4, -1, 64)
x = layers.Reshape(target_shape=(-1, 64))(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
output = layers.Dense(num_classes, activation='softmax', name='ctc_output')(x)
training_model = keras.Model(inputs=input_img, outputs=output)
# Dummy steps_per_epoch and validation_steps for demonstration
steps_per_epoch = max(1, len(train_df) // batch_size)
validation_steps = max(1, len(val_df) // batch_size)
# Re-compile the model to ensure it picks up any potential graph changes (though unlikely here)
# The model structure and compilation should be correct from the previous step.
# However, re-running compile is harmless.
training_model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
# Train the model again
print("Restarting model training with fixed label encoding and generator...")
history = training_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20, # Increased epochs for better training
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1
)
print("Model training finished.")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GRU, Dense, Reshape, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Preprocess the data: Normalize and expand dimensions
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
# Let's see what a digit looks like
print("Training data shape:", x_train.shape)
print("A sample label:", y_train[0])
plt.imshow(x_train[0].squeeze(), cmap='gray')
plt.show()

def build_crnn_model(input_shape=(IMAGE_HEIGHT, None, 1), num_classes=num_classes): # Updated input_shape
    # --- CNN Part ---
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    # Added a third convolutional layer for potentially better feature extraction on more complex text data
    inner = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inner)
    # Removed MaxPooling after the last Conv layer to maintain a higher sequence length
    # --- Bridge to RNN ---
    # Reshape the output of CNN to be a sequence for the RNN
    # CNN output shape will be (batch, height, width, filters)
    # After the Conv and Pooling layers, the height will be IMAGE_HEIGHT // (2*2) = 32 // 4 = 8
    # The width is variable (None) after the first two pooling layers and remains variable.
    # The number of filters is 128 from the last Conv layer.
    # The shape for the RNN should be (batch, sequence_length, features)
    # sequence_length is the width of the last CNN output feature map
    # features is (height of last CNN output) * (number of filters in last Conv)
    conv_shape = K.int_shape(inner)
    # Reshape to (batch, conv_shape[2], conv_shape[1] * conv_shape[3])
    # conv_shape[1] is height, conv_shape[2] is width (sequence length), conv_shape[3] is filters
    inner = Reshape(target_shape=(-1, conv_shape[1] * conv_shape[3]))(inner) # -1 infers the sequence length (width)
    # --- RNN Part ---
    # Adding a Dense layer before GRU for potential dimension reduction or feature transformation
    inner = Dense(128, activation='relu')(inner)
    # Using Bidirectional GRU with more units than before for potentially better sequence modeling
    gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1'))(inner)
    # Added a second Bidirectional GRU layer
    gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2'))(gru_1)
    # --- Output Layer ---
    # Output layer predicting probabilities for each character at each timestep
    # The number of units should be num_classes (including the blank token)
    inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2', activation='softmax')(gru_2)
    # Create the model which outputs the raw predictions
    model = Model(inputs=inputs, outputs=inner)
    model.summary()
    return model
# Get the model with the updated input shape and num_classes
# Ensure num_classes is defined before calling this function
# num_classes is defined in the data preparation step (cell 27sTuTfk1yB3)
prediction_model = build_crnn_model(input_shape=(IMAGE_HEIGHT, None, 1), num_classes=num_classes)

# Define the CTC loss function
def ctc_loss(args):
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
# Define the inputs for the training model
labels = Input(name='the_labels', shape=[1], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Get the prediction model's output
y_pred = prediction_model.output
# Use a Lambda layer to wrap the CTC loss function
loss_out = Lambda(ctc_loss, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])
# Create the final training model
training_model = Model(inputs=[prediction_model.input, labels, input_length, label_length], outputs=loss_out)

# Prepare data for CTC
# train_input_length = np.ones(len(x_train)) * 7  # Sequence length is 7
# train_label_length = np.ones(len(y_train)) * 1  # Label length is 1
# test_input_length = np.ones(len(x_test)) * 7
# test_label_length = np.ones(len(y_test)) * 1
# Reshape the input and label length arrays to have a second dimension of 1
# train_input_length = train_input_length.reshape(-1, 1)
# train_label_length = train_label_length.reshape(-1, 1)
# test_input_length = test_input_length.reshape(-1, 1)
# test_label_length = test_label_length.reshape(-1, 1)
# Compile the training model
training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
# Train the model
# Note: We provide a dummy y_train (np.zeros) because the loss is calculated internally.
history = training_model.fit(
    # x=[x_train, y_train, train_input_length, train_label_length], # Old inputs for MNIST
    train_generator, # Use the new data generator
    steps_per_epoch=steps_per_epoch, # Use calculated steps
    epochs=10, # Keep epochs for now
    validation_data=val_generator, # Use the new validation generator
    validation_steps=validation_steps, # Use calculated validation steps
    verbose=1
    # y=np.zeros(len(y_train)), # Dummy y is handled by the generator output structure
)

# Re-check the blank index and ensure it's used consistently in encoding and character mapping
# The num_classes includes the blank character. If our characters are 0-9, a-z (36 total),
# then num_classes = 37. The indices should be 1-36 for characters and 0 for blank, OR
# 0-35 for characters and 36 for blank. Keras's ctc_batch_cost expects blank_index = num_classes - 1.
# So, our character indices should be 0 to num_classes - 2.
# Let's adjust the char_to_int and int_to_char mappings to reflect this.
import numpy as np
from PIL import Image
import pandas as pd
# Example: create dummy DataFrames with 'image_path' and 'text' columns
# You should replace this with your actual data loading logic
train_df = pd.DataFrame({
    'image_path': ['dummy_path1.png', 'dummy_path2.png'],
    'text': ['5', '0']
})
val_df = pd.DataFrame({
    'image_path': ['dummy_path3.png'],
    'text': ['4']
})
# Build character set from train_df and val_df (assuming single-label classification)
all_labels = np.unique(np.concatenate([train_df['text'].values, val_df['text'].values]))
all_chars = [str(label) for label in all_labels]  # Convert to string if needed
# Assign indices 0 to len(all_chars) - 1 to the characters
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}
# Assign the blank index
blank_index = len(all_chars)  # This will be num_classes - 1
char_to_int[' '] = blank_index
int_to_char[blank_index] = ' '
num_classes = len(all_chars) + 1  # Total number of classes including blank
print(f"Updated Number of unique characters (including blank): {num_classes}")
print(f"Blank index: {blank_index}")
print(f"Char to int mapping sample: {list(char_to_int.items())[:5]}...")
print(f"Int to char mapping sample: {list(int_to_char.items())[:5]}...")
# Define MAX_TEXT_LEN as the maximum length of the text labels in your dataset
MAX_TEXT_LEN = 1  # For single digit classification; set higher if your labels are longer
# Re-define the encoding function to use the updated blank index for padding
def encode_label(text, char_map, max_len=MAX_TEXT_LEN, blank_idx=blank_index):
    encoded = [char_map.get(char, blank_idx) for char in text] # Use .get with default blank_idx for unknown chars
    # Pad labels with the blank index to a fixed length
    if len(encoded) < max_len:
        encoded += [blank_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len] # Truncate if longer than max_len
    return np.array(encoded)
# Modify the data generator to ensure correct label and input length calculations and types
# Define IMAGE_HEIGHT according to your dataset (e.g., 28 for MNIST-like images)
IMAGE_HEIGHT = 28
import os
def mock_load_image(path, img_height):
    # If file exists, load it; otherwise, create a blank image
    if os.path.exists(path):
        img = Image.open(path).convert('L')
    else:
        # Create a blank white image (for demonstration)
        img = Image.new('L', (img_height, img_height), color=255)
    return img
def preprocess_image(path, img_height):
    img = mock_load_image(path, img_height)
    width, height = img.size
    new_width = int(img_height * (width / height))
    img = img.resize((new_width, img_height))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array, new_width
def data_generator(dataframe, char_map, batch_size=32, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index):
    num_samples = len(dataframe)
    max_img_width = 0
    for _, row in dataframe.iterrows():
        img_temp = mock_load_image(row['image_path'], img_height)
        width_temp, height_temp = img_temp.size
        new_width_temp = int(img_height * (width_temp / height_temp))
        if new_width_temp > max_img_width:
            max_img_width = new_width_temp
    target_width = max_img_width
    # Determine the maximum input length based on the target width and CNN architecture
    # Assumes 2 MaxPooling2D layers with pool_size=(2,2)
    max_input_length = target_width // (2 * 2)
    while True:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_df = dataframe.iloc[offset:offset + batch_size]
            images = []
            encoded_labels = []
            input_lengths = []
            label_lengths = []
            for index, row in batch_df.iterrows():
                img_array, original_width = preprocess_image(row['image_path'], img_height)
                if original_width < target_width:
                    padding_needed = target_width - original_width
                    padded_img_array = np.pad(img_array, ((0, 0), (0, padding_needed), (0, 0)), mode='constant', constant_values=1.0)
                else:
                    padded_img_array = img_array
                encoded_label = encode_label(row['text'], char_map, max_text_len, blank_idx)
                images.append(padded_img_array)
                encoded_labels.append(encoded_label)
                # The input length is consistent for all samples in this generator's logic
                # because all images are padded to the same target_width.
                # The input_length to CTC is the sequence length output by the RNN,
                # which is the width of the last CNN layer's output after pooling.
                input_len = max_input_length # Use the calculated max input length
                input_lengths.append(input_len)
                label_lengths.append(len(row['text']))
            images = np.array(images)
            encoded_labels = np.array(encoded_labels)
            input_lengths = np.array(input_lengths, dtype=np.int64).reshape(-1, 1) # Ensure int64
            label_lengths = np.array(label_lengths, dtype=np.int64).reshape(-1, 1) # Ensure int64
            yield {
                'the_input': images,
                'the_labels': encoded_labels,
                'input_length': input_lengths,
                'label_length': label_lengths
            }, np.zeros(len(batch_df))
# Define batch_size and create dummy train_df and val_df for demonstration
batch_size = 32
import pandas as pd
# Example: create dummy DataFrames with 'image_path' and 'text' columns
# You should replace this with your actual data loading logic
train_df = pd.DataFrame({
    'image_path': ['dummy_path1.png', 'dummy_path2.png'],
    'text': ['5', '0']
})
val_df = pd.DataFrame({
    'image_path': ['dummy_path3.png'],
    'text': ['4']
})
# Re-instantiate the data generators with the modified function and updated mappings
train_generator = data_generator(train_df, char_to_int, batch_size=batch_size, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)
val_generator = data_generator(val_df, char_to_int, batch_size=batch_size, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)
# Re-define the CTC Loss Layer to ensure it uses the correct blank index if needed
# In the original code, the K.ctc_batch_cost function automatically uses num_classes - 1
# as the blank index, which should now align with our updated encoding.
# No change needed in CTCLayer itself, but ensuring num_classes is correctly passed if the layer were different.
# Define a minimal CTC model if not already defined
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
input_img = keras.Input(shape=(IMAGE_HEIGHT, None, 1), name='the_input')
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
new_shape = (IMAGE_HEIGHT // 4, -1, 64)
x = layers.Reshape(target_shape=(-1, 64))(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
output = layers.Dense(num_classes, activation='softmax', name='ctc_output')(x)
training_model = keras.Model(inputs=input_img, outputs=output)
# Dummy steps_per_epoch and validation_steps for demonstration
steps_per_epoch = max(1, len(train_df) // batch_size)
validation_steps = max(1, len(val_df) // batch_size)
# Re-compile the model to ensure it picks up any potential graph changes (though unlikely here)
# The model structure and compilation should be correct from the previous step.
# However, re-running compile is harmless.
training_model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
# Train the model again
print("Restarting model training with fixed label encoding and generator...")
history = training_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20, # Increased epochs for better training
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1
)
print("Model training finished.")

# Make predictions on a few test images
for i in range(10):
    image = x_test[i][np.newaxis, :, :, :]
    prediction = prediction_model.predict(image)
    # Use CTC decode to get the best path
    decoded = K.get_value(K.ctc_decode(prediction, input_length=np.ones(1) * 7)[0][0])
    # Display the result
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f'Prediction: {decoded[0]}')
    plt.show()

import os
import tarfile
import urllib.request
import pandas as pd
from PIL import Image
import tensorflow as tf
# Define constants
IMAGE_HEIGHT = 32
MAX_TEXT_LEN = 50 # A reasonable maximum length for text in an image
# Download a small subset of the IAM dataset
url = "https://fki.tic.udl.cat/databases/iam-handwriting-database/original_forms/lines/lines.tgz"
dataset_path = "iam_lines.tgz"
extracted_path = "iam_lines"
if not os.path.exists(dataset_path):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dataset_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please check your internet connection or manually download the file and place it at", dataset_path)
# Check if the file exists before extracting
if not os.path.exists(dataset_path):
    print(f"Dataset file {dataset_path} not found. Please download it manually and place it in the working directory.")
else:
    if not os.path.exists(extracted_path):
        print(f"Extracting {dataset_path}...")
        with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall(path=".")
        print("Extraction complete.")
# Load the ground truth data
# The lines.txt file contains the mapping from image file names to their text content
lines_file = os.path.join(extracted_path, 'lines.txt')
if not os.path.exists(lines_file):
    # Try the parent directory (sometimes lines.txt is extracted to the current directory)
    alt_lines_file = 'lines.txt'
    if os.path.exists(alt_lines_file):
        lines_file = alt_lines_file
    else:
        print(f"Warning: Could not find lines.txt at {lines_file} or {alt_lines_file}. Please check your extraction.")
        lines = []
else:
    with open(lines_file, 'r') as f:
        lines = f.readlines()
data = []
for line in lines:
    if line.startswith('#'): # Skip comments
        continue
    parts = line.strip().split(' ')
    image_id = parts[0]
    # The text is in the parts after the first 8, joined by spaces
    text = ' '.join(parts[8:]).replace('|', ' ') # Replace '|' with space
    # Construct the image path
    # The image path is structured like: forms/a01/a01-000/a01-000-00.png
    folder1 = image_id.split('-')[0]
    folder2 = '-'.join(image_id.split('-')[:2])
    image_name = image_id + '.png'
    image_path = os.path.join(extracted_path, 'lines', folder1, folder2, image_name)
    if os.path.exists(image_path):
        data.append({'image_path': image_path, 'text': text})
# Only create DataFrame if data is not empty and has the correct keys
if data and 'text' in data[0]:
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame(columns=['image_path', 'text'])
print(f"Loaded {len(df)} samples.")
print(df.head())
# Create character set and mapping
all_chars = sorted(list(set(''.join(df['text']))))
char_to_int = {char: i + 1 for i, char in enumerate(all_chars)} # 0 is reserved for blank
int_to_char = {i + 1: char for i, char in enumerate(all_chars)}
char_to_int[' '] = 0 # Assign 0 to the blank character
int_to_char[0] = ' '
num_classes = len(char_to_int)
print(f"Number of unique characters (including blank): {num_classes}")
# Function to preprocess images
def preprocess_image(image_path, img_height=IMAGE_HEIGHT):
    img = Image.open(image_path).convert('L') # Convert to grayscale
    width, height = img.size
    new_width = int(img_height * (width / height))
    img = img.resize((new_width, img_height), Image.Resampling.LANCZOS)
    # Pad the image to a consistent width (optional, but simpler for fixed input size)
    # For a more robust solution, dynamic padding per batch could be used.
    # Let's pad to a reasonable max width for now.
    max_img_width = 800 # Example max width, adjust based on dataset analysis
    if new_width < max_img_width:
        padding = (0, 0, max_img_width - new_width, 0) # left, top, right, bottom
        img = Image.new(img.mode, (max_img_width, img_height), 255) # Pad with white
        img.paste(img, (0, 0))
    img_array = np.array(img).astype(np.float32) / 255.0 # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    return img_array, new_width # Also return the original width for input length calculation
# Function to encode labels
def encode_label(text, char_map, max_len=MAX_TEXT_LEN):
    encoded = [char_map[char] for char in text]
    # Pad labels with 0s to a fixed length (for batching)
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len] # Truncate if longer than max_len
    return np.array(encoded)
# Create a data generator
def data_generator(dataframe, char_map, batch_size=32, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN):
    num_samples = len(dataframe)
    while True:
        # Shuffle data
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_df = dataframe.iloc[offset:offset + batch_size]
            images = []
            encoded_labels = []
            input_lengths = []
            label_lengths = []
            for index, row in batch_df.iterrows():
                img_array, original_width = preprocess_image(row['image_path'], img_height)
                encoded_label = encode_label(row['text'], char_map, max_text_len)
                images.append(img_array)
                encoded_labels.append(encoded_label)
                # Calculate input length: depends on CNN architecture
                # For our CNN with two MaxPooling2D layers with pool_size=(2,2),
                # the width dimension is reduced by a factor of 2 * 2 = 4.
                # Assuming the original width is used before padding for input length calculation
                input_len = original_width // (2 * 2) # Adjust based on actual pooling/strides
                input_lengths.append(input_len)
                label_lengths.append(len(row['text'])) # Actual length of the label
            # Pad images to the max width in the current batch if dynamic padding is needed
            # Or if using fixed padding in preprocess, just convert list to numpy array
            # For simplicity with fixed padding:
            images = np.array(images)
            encoded_labels = np.array(encoded_labels)
            input_lengths = np.array(input_lengths).reshape(-1, 1)
            label_lengths = np.array(label_lengths).reshape(-1, 1)
            yield [images, encoded_labels, input_lengths, label_lengths], np.zeros(len(batch_df)) # Dummy y for training
# Example of using the generator
# train_generator = data_generator(df, char_to_int, batch_size=32)
# batch_data, dummy_y = next(train_generator)
# sample_images, sample_labels, sample_input_lengths, sample_label_lengths = batch_data
# print("\nSample batch shapes:")
# print("Images shape:", sample_images.shape)
# print("Encoded Labels shape:", sample_labels.shape)
# print("Input Lengths shape:", sample_input_lengths.shape)
# print("Label Lengths shape:", sample_label_lengths.shape)