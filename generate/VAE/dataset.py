from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def dataloader(batch_size=128, num_works=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST('./minist', train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST('./minist', train=False, transform=transform, download=True)

    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return mnist_train, mnist_test, classes
