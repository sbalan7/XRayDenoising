import matplotlib.pyplot as plt
import numpy as np
import cv2

    
def add_gaussian_noise(image, SNR):
    h, w = image.shape
    sigma_X = np.var(image.flatten())
    var = sigma_X / SNR
    g = np.random.normal(0, np.sqrt(var), (h, w))
    g = g.reshape(h, w)
    noisy_image = image + g
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def add_poisson_noise(image, SNR):
    noisy_image = np.random.poisson(image/255.0 * SNR) / SNR * 255
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image
    
def add_salt_pepper_noise(image, SNR=100, sp_ratio=0.5):
    I = np.copy(image)
    contamination = 1/(SNR+1)
    rnd = np.random.rand(I.shape[0], I.shape[1])

    I[rnd < contamination * sp_ratio] = 0
    I[rnd > 1 - contamination * sp_ratio] = 255
    '''
    salt = np.ceil(contamination * image.size * sp_ratio)
    salt_loc = np.array([np.random.randint(0, i-1, int(salt)) for i in image.shape])
    I[salt_loc] = 255

    pepper = np.ceil(contamination * image.size * (1-sp_ratio))
    pepper_loc = np.array([np.random.randint(0, i-1, int(pepper)) for i in image.shape])
    I[pepper_loc] = 0
    '''
    return I

'''
SNR = 10
SNR_dB = 10 * np.log10(SNR)
filename = 'Images/proc_gridge_xray.tif'

I = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
I1 = add_gaussian_noise(I, SNR)
I2 = add_poisson_noise(I, SNR)
I3 = add_salt_pepper_noise(I, SNR)
'''
'''
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.gray()

axs[0, 0].imshow(I)
axs[0, 0].set_title('Original')
axs[0, 1].imshow(I1)
axs[0, 1].set_title(f'Gaussian, SNR = {SNR_dB} dB')
axs[1, 0].imshow(I2)
axs[1, 0].set_title(f'Poisson, SNR = {SNR_dB} dB')
axs[1, 1].imshow(I3)
axs[1, 1].set_title(f'S&P, SNR = {SNR_dB} dB')

[_.set_axis_off() for _ in axs.ravel()]
plt.tight_layout()
plt.show()
'''
'''
fig, axs = plt.subplots(1, 4, figsize=(15, 4.5))
plt.gray()

axs[0].imshow(I)
axs[0].set_title('Original')
axs[1].imshow(I1)
axs[1].set_title(f'Gaussian, SNR = {SNR_dB} dB')
axs[2].imshow(I2)
axs[2].set_title(f'Poisson, SNR = {SNR_dB} dB')
axs[3].imshow(I3)
axs[3].set_title(f'S&P, SNR = {SNR_dB} dB')

[_.set_axis_off() for _ in axs.ravel()]
plt.tight_layout()
plt.show()
'''
