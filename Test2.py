import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import random
import string
import sys
import editdistance # For CER calculation

# --- Constants ---
IMAGE_HEIGHT = 32
MAX_TEXT_LEN = 10 # Max text length for synthetic data
DATA_DIR = 'synthetic_captcha_dataset'
BATCH_SIZE = 32
EPOCHS = 20 # Number of training epochs

# --- Synthetic Data Generation ---
def generate_synthetic_data(num_samples=1000, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = []
    chars = string.ascii_lowercase + string.digits # Use lowercase letters and digits

    # Attempt to load a system font or use default
    try:
        if sys.platform == "win32":
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
        elif sys.platform == "darwin": # macOS
            font_path = "/Library/Fonts/Arial.ttf"
        else: # Linux and others
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf" # Example common font
        font = ImageFont.truetype(font_path, img_height - 5)
        print(f"Using system font: {font_path}")
    except IOError:
        print("System font not found. Using a default PIL font, text quality may be poor.")
        font = ImageFont.load_default()

    for i in range(num_samples):
        text_length = random.randint(1, max_text_len)
        text = ''.join(random.choice(chars) for _ in range(text_length))

        try:
            text_width = font.getbbox(text)[2]
        except AttributeError:
             text_width, _ = font.getsize(text)

        img_width = max(text_width + 10, 50)

        img = Image.new('L', (img_width, img_height), color=255)
        d = ImageDraw.Draw(img)

        try:
            d.text((5, 2), text, fill=0, font=font)
        except TypeError:
             d.text((5, 2), text, fill=0)

        image_name = f"synth_img_{i}.png"
        image_path = os.path.join(data_dir, image_name)
        img.save(image_path)

        data.append({'image_path': image_path, 'text': text})

    return pd.DataFrame(data)

# Generate the dataset
df = generate_synthetic_data()
print(f"Generated {len(df)} synthetic samples.")
display(df.head())

# Create character set and mapping
all_chars = sorted(list(set(''.join(df['text']))))
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}

# Assign the blank index (num_classes - 1)
blank_index = len(all_chars)
char_to_int[' '] = blank_index
int_to_char[blank_index] = ' '

num_classes = len(char_to_int)
print(f"Number of unique characters (including blank): {num_classes}")
print(f"Blank index: {blank_index}")


# --- Preprocessing and Data Generator ---
def preprocess_image(image_path, img_height=IMAGE_HEIGHT):
    img = Image.open(image_path).convert('L')
    width, height = img.size
    new_width = int(img_height * (width / height))
    img = img.resize((new_width, img_height), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array, new_width

def encode_label(text, char_map, max_len=MAX_TEXT_LEN, blank_idx=blank_index):
    encoded = [char_map.get(char, blank_idx) for char in text]
    if len(encoded) < max_len:
        encoded += [blank_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return np.array(encoded)

def data_generator(dataframe, char_map, batch_size=BATCH_SIZE, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index):
    num_samples = len(dataframe)

    max_img_width = 0
    for _, row in dataframe.iterrows():
        img_temp = Image.open(row['image_path']).convert('L')
        width_temp, height_temp = img_temp.size
        new_width_temp = int(img_height * (width_temp / height_temp))
        if new_width_temp > max_img_width:
            max_img_width = new_width_temp
    target_width = max_img_width

    max_input_length = target_width // (2 * 2) # Based on CNN architecture

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

                input_len = max_input_length
                input_lengths.append(input_len)

                label_lengths.append(len(row['text']))

            images = np.array(images)
            encoded_labels = np.array(encoded_labels)
            # Ensure input_lengths and label_lengths are 1D arrays
            input_lengths = np.array(input_lengths, dtype=np.int64)
            label_lengths = np.array(label_lengths, dtype=np.int64)

            yield {
                'the_input': images,
                'the_labels': encoded_labels,
                'input_length': input_lengths,
                'label_length': label_lengths
            }, np.zeros(len(batch_df))

# Split data and create generators
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].reset_index(drop=True)
val_df = df.iloc[train_size:].reset_index(drop=True)

train_generator = data_generator(train_df, char_to_int, batch_size=BATCH_SIZE, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)
val_generator = data_generator(val_df, char_to_int, batch_size=BATCH_SIZE, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE


# --- Model Architecture ---
def build_crnn_model(input_shape=(IMAGE_HEIGHT, None, 1), num_classes=num_classes):
    inputs = tf.keras.layers.Input(name='the_input', shape=input_shape, dtype='float32')

    inner = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inner)

    inner = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inner)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inner)

    inner = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inner)

    conv_shape = K.int_shape(inner)
    inner = tf.keras.layers.Reshape(target_shape=(-1, conv_shape[1] * conv_shape[3]))(inner)

    inner = tf.keras.layers.Dense(128, activation='relu')(inner)

    gru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1'))(inner)
    gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2'))(gru_1)

    inner = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense2', activation='softmax')(gru_2)

    model = tf.keras.models.Model(inputs=inputs, outputs=inner)
    return model

# Build the prediction model
prediction_model = build_crnn_model(num_classes=num_classes)
prediction_model.summary()


# --- CTC Loss and Training Setup ---
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return loss

# Define inputs for the training model
labels = tf.keras.layers.Input(name='the_labels', shape=(None,), dtype='float32')
input_length = tf.keras.layers.Input(name='input_length', shape=(1,), dtype='int64')
label_length = tf.keras.layers.Input(name='label_length', shape=(1,), dtype='int64')

# Get prediction model output
y_pred = prediction_model.output

# Use the custom CTC loss layer
ctc_layer = CTCLayer(name='ctc_loss')
loss_out = ctc_layer(labels, y_pred, input_length, label_length)

# Create the training model
training_model = tf.keras.models.Model(
    inputs=[prediction_model.input, labels, input_length, label_length],
    outputs=loss_out
)

# Compile the training model
training_model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
training_model.summary()

# Train the model
print("Starting model training...")
history = training_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1
)
print("Model training finished.")


# --- CTC Decoding ---
def decode_batch_predictions(pred, int_to_char_map):
    input_len = np.ones(pred.shape[0]) * K.int_shape(pred)[1]
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    decoded_text = []
    for res in results.numpy():
        # Filter out -1 and blank_index before mapping to characters
        text = "".join([int_to_char_map[char] for char in res if char != blank_index and char != -1])
        decoded_text.append(text)
    return decoded_text


# --- Evaluation ---
# Create a test data generator using the validation set for evaluation
test_generator = data_generator(val_df, char_to_int, batch_size=BATCH_SIZE, img_height=IMAGE_HEIGHT, max_text_len=MAX_TEXT_LEN, blank_idx=blank_index)

total_cer = 0
total_samples = 0

print("\nStarting evaluation on test set...")

for i in range(validation_steps):
    batch_data, _ = next(test_generator)
    images = batch_data['the_input']
    encoded_labels = batch_data['the_labels']
    label_lengths = batch_data['label_length']

    predictions = prediction_model.predict(images)
    decoded_texts = decode_batch_predictions(predictions, int_to_char)

    true_texts = []
    for j in range(encoded_labels.shape[0]):
        # Corrected access for label_lengths
        actual_label_length = label_lengths[j]
        true_encoded_label = encoded_labels[j][:actual_label_length]
        true_text = "".join([int_to_char[char] for char in true_encoded_label if char != blank_index])
        true_texts.append(true_text)

    batch_cer = 0
    for k in range(len(decoded_texts)):
        distance = editdistance.eval(true_texts[k], decoded_texts[k])
        cer = distance / max(len(true_texts[k]), 1)
        batch_cer += cer

    total_cer += batch_cer
    total_samples += len(decoded_texts)

average_cer = total_cer / total_samples if total_samples > 0 else 0
print(f"\nEvaluation complete.")
print(f"Average Character Error Rate (CER) on the test set: {average_cer:.4f}")


# --- Prediction on New Image (Example) ---
# Find the target width used during training
max_img_width_train = 0
for _, row in train_df.iterrows():
    img_temp = Image.open(row['image_path']).convert('L')
    width_temp, height_temp = img_temp.size
    new_width_temp = int(IMAGE_HEIGHT * (width_temp / height_temp))
    if new_width_temp > max_img_width_train:
        max_img_width_train = new_width_temp
TRAINING_TARGET_WIDTH = max_img_width_train


def predict_text_from_image(image_path, prediction_model, int_to_char_map, img_height=IMAGE_HEIGHT, target_width=TRAINING_TARGET_WIDTH):
    img = Image.open(image_path).convert('L')
    width, height = img.size
    new_width = int(img_height * (width / height))
    img = img.resize((new_width, img_height), Image.Resampling.LANCZOS)

    if new_width < target_width:
        padding_needed = target_width - new_width
        padded_img = Image.new(img.mode, (target_width, img_height), 255)
        padded_img.paste(img, (0, 0))
        img_array = np.array(padded_img).astype(np.float32) / 255.0
    else:
         # Resize if wider than target to maintain consistency, acknowledging potential distortion
         if new_width > target_width:
             print(f"Warning: Input image width ({original_width}) is greater than training target width ({target_width}). Resizing.")
             img = img.resize((target_width, img_height), Image.Resampling.LANCZOS)

         img_array = np.array(img).astype(np.float32) / 255.0


    img_array = np.expand_dims(img_array, axis=-1)
    image_batch = np.expand_dims(img_array, axis=0)

    predictions = prediction_model.predict(image_batch)
    decoded_texts = decode_batch_predictions(predictions, int_to_char_map)

    return decoded_texts[0]

# Example prediction on a sample image from the validation set
if not val_df.empty:
    test_image_path = val_df.iloc[0]['image_path']
    true_text = val_df.iloc[0]['text']
    predicted_text = predict_text_from_image(test_image_path, prediction_model, int_to_char)

    print(f"\nExample Prediction:")
    print(f"Test Image: {test_image_path}")
    print(f"True Text: {true_text}")
    print(f"Predicted Text: {predicted_text}")
else:
    print("\nNo validation data available for example prediction.")
