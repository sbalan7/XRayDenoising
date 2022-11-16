from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import addnoise as noise
import numpy as np
import cv2
import os


def add_noise(k, I, SNR):
    if k == 0:
        return noise.add_gaussian_noise(I, SNR)
    elif k == 1:
        return noise.add_poisson_noise(I, SNR)
    elif k == 2:
        return noise.add_salt_pepper_noise(I, SNR)
    return 'k somehow got an invalid value'


processed_list = [_ for _ in os.listdir('Training Images/') if _.startswith('proc_')]

data = []
for filename in processed_list:
    data.append(cv2.imread('Training Images/'+filename, cv2.IMREAD_UNCHANGED))
data = np.stack(data)

fig, ax = plt.subplots(figsize=(8, 8))
cmap_points = iter(plt.cm.Greens(np.linspace(0.2, 0.8, 5)))

for SNR in range(0, 50, 10):
    noise_type = np.random.randint(3, size=len(processed_list))
    noisy_images = []
    for _ in range(len(processed_list)):
        noisy_images.append(add_noise(noise_type[_], data[_], np.power(10, SNR/10)).flatten())
    noisy_data = np.stack(noisy_images)

    pca = PCA(n_components=17)
    pca.fit(noisy_data)
    c = next(cmap_points)
    ax.plot(np.cumsum(pca.explained_variance_ratio_), c=c, label=f'SNR={SNR} dB')
    ax.set_ylabel('explained variance ratio')
    ax.set_xlabel('# of components')

plt.title('Explained variance to # of components for noisy images')
plt.legend()
plt.show()

'''
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for filename in processed_list:
    I = cv2.imread('Images/'+filename, cv2.IMREAD_UNCHANGED)
    pca = PCA(n_components=25)
    pca.fit(I)
    axs[0].plot(pca.explained_variance_)
axs[0].set_title('# of PCA components for clean images')

snr10_random = np.random.randint(3, size=len(processed_list))
for i, filename in enumerate(processed_list):
    I = cv2.imread('Images/'+filename, cv2.IMREAD_UNCHANGED)
    I = add_noise(snr10_random[i], I, 10)
    pca = PCA(n_components=25)
    pca.fit(I)
    axs[1].plot(pca.explained_variance_)
axs[1].set_title('# of PCA components for images with SNR = 10 dB')

snr20_random = np.random.randint(3, size=len(processed_list))
for i, filename in enumerate(processed_list):
    I = cv2.imread('Images/'+filename, cv2.IMREAD_UNCHANGED)
    I = add_noise(snr20_random[i], I, 20)
    pca = PCA(n_components=25)
    pca.fit(I)
    axs[2].plot(pca.explained_variance_)
axs[2].set_title('# of PCA components for images with SNR = 20 dB')

plt.tight_layout()
plt.show()

'''