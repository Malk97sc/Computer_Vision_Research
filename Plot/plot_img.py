import matplotlib.pyplot as plt

def plot_img(img, gray = True):
    if gray == False:
        plt.imshow(img)    
    else: 
        plt.imshow(img, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()