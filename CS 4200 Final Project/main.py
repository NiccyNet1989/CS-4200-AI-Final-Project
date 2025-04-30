import os
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
import keras

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from sklearn.utils import resample

# Download the dataset
path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")
print("Path to dataset files:", path)

# Find the CSV file in the downloaded directory
csv_file = None
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            csv_file = os.path.join(root, file)
            print(f"Found CSV file: {csv_file}")
            break
    if csv_file:
        break

if not csv_file:
    raise FileNotFoundError("No CSV file found in the dataset.")

# Load the full dataset
full_df = pd.read_csv(csv_file)

# Split into two DataFrames:
# 1. First column (labels: 0-25 representing A-Z)
labels_df = full_df.iloc[:, 0].to_frame(name='label')

# 2. Remaining 784 columns (28x28 pixels)
pixels_df = full_df.iloc[:, 1:785]

print("\nFull dataframe : " + str(full_df.shape))


def downsize_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return df.iloc[::n, :].reset_index(drop=True)


lables_downsized = downsize_data(labels_df, 20)
pixels_downsized = downsize_data(pixels_df, 20)

lables_downsized = lables_downsized['label'].values
pixels_downsized = pixels_downsized.values

# Convert the DataFrames to NumPy arrays
labels = labels_df['label'].values  # Shape: (num_samples,)
pixels = pixels_df.values  # Shape: (num_samples, 784)

# Reshape pixels to (num_samples, 28, 28, 1) for Keras
images = pixels.reshape(-1, 28, 28, 1).astype('float32')
images_downsized = pixels_downsized.reshape(-1, 28, 28, 1).astype('float32')

# Normalize pixel values to [0, 1] (optional but recommended)
images /= 255.0

print("\nDataframe downsized, Labels = " + str(lables_downsized.shape))
print("Dataframe downsized, Images = " + str(images_downsized.shape) + "\n")

# =================================================================
# Printing random entries from the full dataframe
pixels_scaled = pixels_df / 255.0


def reshapeImage(dataframe, index):
    sample_image = dataframe.iloc[index].values.reshape(28, 28)  # Reshape to 28x28
    return sample_image


def printImageAtIndex(index):
    charNum = int(labels_df.iloc[index, 0])
    sample_image = reshapeImage(pixels_scaled, index)
    plt.imshow(sample_image, cmap='gray')
    plt.title(chr(charNum + ord("A")))
    plt.colorbar()
    plt.grid(False)
    plt.show()


def labelToChar(index):
    charNum = int(labels_df.iloc[index, 0])
    return chr(charNum + ord("A"))


plt.figure(figsize=(10, 10))


def plot_random_entries():
    for i in range(25):
        selectedIndex = random.randint(0, len(labels_df) - 1)
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(reshapeImage(pixels_scaled, selectedIndex), cmap=plt.cm.binary)
        plt.xlabel(str(selectedIndex) + ", " + chr(int(labels_df.iloc[selectedIndex, 0]) + ord("A")))
    plt.show()


plot_random_entries()
plot_random_entries()
plot_random_entries()


# Demonstrating the class imbalance of the original dataframe
def countInstances(character):
    count = 0;

    for row in range(len(labels_df)):
        if labelToChar(row) == character:
            count += 1

    return count;


def get_instances_array():
    return_array = [0] * 26
    for index in range(0, len(labels_df)):
        return_array[int(labels_df.iloc[index, 0])] = return_array[int(labels_df.iloc[index, 0])] + 1
    return return_array


instances_count = np.array(get_instances_array())
instances_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
plt.bar(instances_label, instances_count)
plt.show()

# =================================================================
# Attempting to create balanced dataset

labels_balanced = []
images_balanced = []

classes, counts = np.unique(labels, return_counts=True)
lowest_count = np.min(counts)

for letter in classes:
    class_data = images[labels == letter]
    class_data_sample = resample(class_data, replace=False, n_samples=lowest_count)
    images_balanced.append(class_data_sample)
    labels_balanced.extend([letter] * lowest_count)

images_balanced = np.concatenate(images_balanced)
labels_balanced = np.array(labels_balanced)

print("Balanced Labels = " + str(labels_balanced.shape))
print("Balanced Images = " + str(images_balanced.shape))

# =================================================================
# Start building the model and feeding it data


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# model.fit(images, labels, epochs=3, validation_split=0.1)
# model.fit(images_downsized, lables_downsized, epochs=5, validation_split=0.1)
model.fit(images_balanced, labels_balanced, epochs=10, validation_split=0.1)

probability_model = tf.keras.Sequential([model])


# ===================================================================================
def largeIndent():
    print(
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


# Simple Drawing Program for Generating PNG Images
# Initialize pygame
pygame.init()

# Constants
WINDOW_SIZE = (560, 690)  # Increased height to accommodate both buttons
GRID_SIZE = 28
CELL_SIZE = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BUTTON_COLOR = (50, 50, 50)
BUTTON_HOVER = (70, 70, 70)

# Create the window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("28x28 Drawing App")

# Create a surface for the drawing area
drawing_area = pygame.Surface((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
drawing_area.fill(BLACK)

# Create a surface for the actual 28x28 image
image_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
test_image = None  # This will store the pixel data when saved

# Button rectangles
save_button_rect = pygame.Rect(
    WINDOW_SIZE[0] // 2 - 100,
    GRID_SIZE * CELL_SIZE + 20,
    200,
    50
)

clear_button_rect = pygame.Rect(
    WINDOW_SIZE[0] // 2 - 100,
    GRID_SIZE * CELL_SIZE + 90,
    200,
    50
)

# Font
font = pygame.font.SysFont('Arial', 24)

# Main loop
running = True
drawing = False
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        # Handle mouse button down
        elif event.type == MOUSEBUTTONDOWN:
            # Check if clicked on drawing area
            if event.pos[1] < GRID_SIZE * CELL_SIZE:
                drawing = True
                # Draw immediately on click (not just on drag)
                x, y = event.pos
                if 0 <= x < GRID_SIZE * CELL_SIZE and 0 <= y < GRID_SIZE * CELL_SIZE:
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    image_data[grid_y, grid_x] = 255
                    pygame.draw.rect(
                        drawing_area,
                        WHITE,
                        (grid_x * CELL_SIZE, grid_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )
            # Check if clicked on save button
            elif save_button_rect.collidepoint(event.pos):
                # Store the pixel data in test_image
                test_image = image_data.copy()

                # Save the image
                import matplotlib.pyplot as plt

                # plt.imsave('drawing.png', test_image, cmap='gray')
                # print("Image saved as drawing.png")
                largeIndent()
                print("Pixel data stored in test_image array:")
                test_image = test_image.reshape(-1, 28, 28, 1).astype('float32')
                test_image = test_image / 255

                testing_result = probability_model.predict(test_image)[0]
                print("\nModel predicts image is: " + "\"" + chr(
                    np.argmax(testing_result) + 65) + "\"" + "\n\tConfidence: " + str(
                    testing_result[np.argmax(testing_result)]) + "\n")

                for letter_index in range(26):
                    print(str(chr(letter_index + 65)) + " : " + str((testing_result[letter_index] * 100)) + "%")

            # Check if clicked on clear button
            elif clear_button_rect.collidepoint(event.pos):
                # Clear the drawing area and image data
                drawing_area.fill(BLACK)
                image_data.fill(0)

        # Handle mouse button up
        elif event.type == MOUSEBUTTONUP:
            drawing = False

        # Handle mouse motion
        elif event.type == MOUSEMOTION and drawing:
            x, y = event.pos
            # Only draw if within drawing area
            if 0 <= x < GRID_SIZE * CELL_SIZE and 0 <= y < GRID_SIZE * CELL_SIZE:
                grid_x = x // CELL_SIZE
                grid_y = y // CELL_SIZE

                # Set the pixel to white in the image data
                image_data[grid_y, grid_x] = 255

                # Draw on the enlarged display
                pygame.draw.rect(
                    drawing_area,
                    WHITE,
                    (grid_x * CELL_SIZE, grid_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    # Clear the screen
    screen.fill(GRAY)

    # Draw the grid lines
    for x in range(GRID_SIZE + 1):
        pygame.draw.line(
            screen,
            (50, 50, 50),
            (x * CELL_SIZE, 0),
            (x * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        )
    for y in range(GRID_SIZE + 1):
        pygame.draw.line(
            screen,
            (50, 50, 50),
            (0, y * CELL_SIZE),
            (GRID_SIZE * CELL_SIZE, y * CELL_SIZE)
        )

    # Draw the drawing area
    screen.blit(drawing_area, (0, 0))

    # Draw the save button
    save_button_color = BUTTON_HOVER if save_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, save_button_color, save_button_rect, border_radius=5)

    # Draw the clear button
    clear_button_color = BUTTON_HOVER if clear_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, clear_button_color, clear_button_rect, border_radius=5)

    # Draw the button texts
    save_text = font.render("Feed Model", True, WHITE)
    save_text_rect = save_text.get_rect(center=save_button_rect.center)
    screen.blit(save_text, save_text_rect)

    clear_text = font.render("Clear Canvas", True, WHITE)
    clear_text_rect = clear_text.get_rect(center=clear_button_rect.center)
    screen.blit(clear_text, clear_text_rect)

    # Update the display
    pygame.display.flip()

pygame.quit()
