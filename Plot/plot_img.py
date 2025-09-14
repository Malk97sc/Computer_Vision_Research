import matplotlib.pyplot as plt

def plot_img(img, gray = True, size = (6, 8)):
    plt.figure(figsize = size)
    if gray == False:
        plt.imshow(img)    
    else: 
        plt.imshow(img, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()