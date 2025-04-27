import matplotlib.pyplot as plt
import numpy as np

def show_images(idx, ds):
    #idx = np.random.randint(len(ds['train']))
    x, y = ds[idx]
    x = x.cpu().squeeze(0).numpy()
    y = y.cpu().squeeze(0).numpy()
    plt.subplot(1,2,1)
    plt.imshow(x, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, cmap='gray')
    plt.show()

def show_random_images(ds, k= 5, figsize = (20,40)):
    fig, axes = plt.subplots(k, 2,figsize=figsize)
    for i in range(k):
        idx = np.random.randint(len(ds))
        for j in range(2):
            x = ds[idx][j].cpu().squeeze(0).numpy()
            axes[i][j].imshow(x,cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.show()


