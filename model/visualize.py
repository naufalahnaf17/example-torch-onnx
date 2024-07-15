import matplotlib.pyplot as plt

def visualize_data(data):
    plt.figure(figsize=(12,12))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(data[i][0].squeeze(0),cmap=plt.cm.binary)
        plt.title(f"Label : {data[i][1]}")
        plt.axis("off")
    plt.show()