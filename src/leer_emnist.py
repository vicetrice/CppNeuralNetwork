import numpy as np
import matplotlib.pyplot as plt

images_path = 'EMNIST_archive/emnist-balanced-train-images-idx3-ubyte'
labels_path = 'EMNIST_archive/emnist-balanced-train-labels-idx1-ubyte'

def load_images(filename):
    with open(filename, 'rb') as f:
        f.read(4) 
        n_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, rows, cols)
        return data

def load_labels(filename):
    with open(filename, 'rb') as f:
        f.read(4) 
        n_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def label_to_ascii(label):
    """
    EMNIST Balanced: etiquetas de 0 a 46
    - 0–9   -> dígitos '0'-'9'
    - 10–35 -> letras mayúsculas 'A'-'Z'
    - 36–45 -> letras minúsculas 'a'-'j' (Balanced no tiene todas)
    """
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 45:
        return chr(label - 36 + ord('a'))
    else:
        return '?'

def correct_orientation(img):
    flipped = np.fliplr(img)
    rotated = np.rot90(flipped, k=1)
    return rotated

images = load_images(images_path)
labels = load_labels(labels_path)

start_idx = 0  
end_idx = 100   
num_images = end_idx - start_idx
grid_size = int(np.ceil(np.sqrt(num_images))) 

plt.figure(figsize=(12, 12))
for j, i in enumerate(range(start_idx, end_idx)):

    img = images[i]
    label_ascii = label_to_ascii(labels[i])
    plt.subplot(grid_size, grid_size, j+1)
    plt.imshow(img, cmap='gray')
    plt.title(label_ascii, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
