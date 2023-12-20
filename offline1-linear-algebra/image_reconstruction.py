import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = np.dot(U[:, :k], np.dot(np.diag(Sigma[:k]), Vt[:k, :]))
    return A_k

image_path = 'image.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# target_size = 500
# image = cv2.resize(image, (target_size, target_size))

k_values = np.arange(1, min(image.shape), step=3)  

print(k_values[:10])
for k in k_values[:10]:
    approx_image = low_rank_approximation(image, k)

    plt.imshow(approx_image, cmap='gray')
    plt.title(f'k-rank Approximation (k={k})')
    plt.axis('off')
    plt.show()


# min k for clarity = 46