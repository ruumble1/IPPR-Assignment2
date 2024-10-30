import os
import tensorflow as tf

# Define directories
positive_images_dir = r'Data/positive/images'
positive_masks_dir = r'Data/positive/masks'
negative_images_dir = r'Data/negative/images'
negative_masks_dir = r'Data/negative/masks'

# Function to load images and masks
def load_negative(images_dir, masks_dir):
    images = []
    masks = []
    
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))  # Adjust if masks have different format
            
            # Load image
            image = tf.keras.preprocessing.image.load_img(img_path, target_size=(448, 448))  # Resize as needed
            image = tf.keras.preprocessing.image.img_to_array(image) / 447.0  # Normalize
            
            # Load mask
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(448, 448), color_mode='grayscale')  # Load as grayscale
            mask = tf.keras.preprocessing.image.img_to_array(mask) / 447.0  # Normalize
            
            images.append(image)
            masks.append(mask)
    
    return tf.convert_to_tensor(images), tf.convert_to_tensor(masks)

def load_positive(images_dir, masks_dir):
    images = []
    masks = []
    
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))  # Adjust if masks have different format
            
            # Load image
            image = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))  # Resize as needed
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
            
            # Load mask
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(256, 256), color_mode='grayscale')  # Load as grayscale
            mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0  # Normalize
            
            images.append(image)
            masks.append(mask)
    
    return tf.convert_to_tensor(images), tf.convert_to_tensor(masks)


# Load positive data
positive_images, positive_masks = load_data(positive_images_dir, positive_masks_dir)

# Load negative data
negative_images, negative_masks = load_data(negative_images_dir, negative_masks_dir)

# Combine positive and negative data
images = tf.concat([positive_images, negative_images], axis=0)
masks = tf.concat([positive_masks, negative_masks], axis=0)

# Print shapes
print(f'Loaded images shape: {images.shape}')
print(f'Loaded masks shape: {masks.shape}')