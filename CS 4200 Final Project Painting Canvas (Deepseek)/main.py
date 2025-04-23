import pygame
import numpy as np
from pygame.locals import *
import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import random


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

# Verify the shapes
print("\nShapes:")
print(f"Full dataset: {full_df.shape}")
print(f"Labels DataFrame: {labels_df.shape}")
print(f"Pixels DataFrame: {pixels_df.shape}")

# Show some samples
print("\nFirst few labels:")
print(labels_df.head())

print("\nFirst few pixel values (first 5 columns):")
print(pixels_df.iloc[:, :5].head())







# Scale down pixel values
# Resultant values are between 0 : 1
pixels_df_scaled = pixels_df / 255.0
print("\nPixel integer values scaled down and saved to \"pixels_df_scaled\" ")

def reshapeImage(dataframe, index):
  sample_image = pixels_df_scaled.iloc[index].values.reshape(28, 28)  # Reshape to 28x28
  return sample_image

def printImageAtIndex(index):
  charNum = int(labels_df.iloc[index,0])
  sample_image = reshapeImage(pixels_df_scaled, index)
  plt.imshow(sample_image, cmap='gray')
  plt.title(chr(charNum+ord("A")))
  plt.colorbar()
  plt.grid(False)
  plt.show()


def labelToChar(index):
  charNum = int(labels_df.iloc[index,0])
  return chr(charNum + ord("A"))


def labelToInt(index):
  charNum = int(labels_df.iloc[index,0])
  return charNum

def countInstances(character):
  count = 0;

  for row in range(len(labels_df)):
    if labelToChar(row) == character:
      count += 1

  return count;


def getInstancesTable():
  firstRow = {'Letter' : ["A"], 'Count' : [countInstances('A')]}
  returnDF = pd.DataFrame(firstRow)
  for i in range(1,26):
    nextRow = {'Letter' : [chr(i+ord("A"))], 'Count' : [countInstances(chr(i+ord("A")))]}
    returnDF = pd.concat([returnDF, pd.DataFrame(nextRow)], ignore_index=True)
  return returnDF



#=================================================================
#Testing useful functions
#print(getInstancesTable())


















#===================================================================================
#Simple Drawing Program for Generating PNG Images
# Initialize pygame
# Initialize pygame
pygame.init()

# Constants
WINDOW_SIZE = (560, 630)  # 20x magnification for 28x28 grid plus space for button
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

# Button rectangle
button_rect = pygame.Rect(
    WINDOW_SIZE[0] // 2 - 100,
    GRID_SIZE * CELL_SIZE + 30,
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
            elif button_rect.collidepoint(event.pos):
                # Store the pixel data in test_image
                test_image = image_data.copy()

                # Save the image
                import matplotlib.pyplot as plt

                plt.imsave('drawing.png', test_image, cmap='gray')
                print("Image saved as drawing.png")
                print("Pixel data stored in test_image array:")
                print(test_image)

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
    button_color = BUTTON_HOVER if button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, button_rect, border_radius=5)

    # Draw the button text
    text_surface = font.render("Save Image", True, WHITE)
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)

    # Update the display
    pygame.display.flip()

pygame.quit()