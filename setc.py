import numpy as np
import torch
import torch.nn.functional as f
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from visdom import Visdom


# creating a neural network with 2 hidden layer convoluted and pooling layer
class Mnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 100)  # first layer
        self.linear2 = nn.Linear(100, 50)  # pooling layer
        self.final_linear = nn.Linear(50, 10)  # output layer

    def forward(self, image):
        x = image.view(-1, 28 * 28)  # flatten the image
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = self.final_linear(x)
        return f.log_softmax(x, dim=1)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

################################ Download MNIST dataset  ################################
train = datasets.MNIST('mnist_data', transform=transform, download=True, train=True,
                       target_transform = lambda y: torch.randint(0, 10, (1,)).item())  # random labels)
test = datasets.MNIST('mnist_data', transform=transform, download=True, train=False)

################################ Divide MNIST dataset  ################################
# diving the MNIST train data in 3 equal sets of A,B,C
traina, trainb, trainc = torch.utils.data.random_split(train, [20000, 20000, 20000])

# diving the MNIST Test data in 3 sets for 3 train sets
testa, testb, testc = torch.utils.data.random_split(test, [3330, 3330, 3340])

################################ Iterator for dataset  ################################
loadc = torch.utils.data.DataLoader(trainc, batch_size=24, shuffle=True)

# data loader of test sets
loadtestc = torch.utils.data.DataLoader(testc, batch_size=24, shuffle=True)


################################ Training Model  ################################
model = Mnet()
cec_loss = nn.CrossEntropyLoss()  # loss function
params = model.parameters()  # adjustable parameters and gradient
optimizer = optim.Adam(params=params, lr=0.001)

n_epochs = 3
n_iterations = 0
vis = Visdom()  # dynamic graphing window  # python -m visdom.server
vis_window = vis.line(np.array([0]), np.array([0]))  # initializing visdom


for e in range(n_epochs): # loop for epoch
    for i, (images, labels) in enumerate(loadc):  # loop on  all images in one batch
        images = Variable(images)  # pass images through autograd variable for it to create gradient
        labels = Variable(labels)  # pass labels through autograd variable for it to create gradient
        output = model(images)  # passing images to model

        model.zero_grad()  # initializing the gradient
        loss = cec_loss(output, labels)  # calculating loss
        loss.backward()  # back propagation
        optimizer.step()  # update the weights

        n_iterations += 1

        vis.line(np.array([loss.item()]), np.array([n_iterations]), win=vis_window, update='append')     # display on Visdom

################################  Accuracy Test  ################################
correct = 0
total = 0

# we do no need gradients to change the weight while testing, temporary turn all gradients_required to false
with torch.no_grad():
    for data in loadtestc:  # loop over test set
        images, labels = data
        output = model(images)  # pass images to model and get output
        for expected, actual in enumerate(output):    # loop over output to compare the actual and expected values
            if torch.argmax(actual) == labels[expected]:
                correct += 1
            else:
                print(torch.argmax(actual), labels[expected])  # print which values were not correctly identified
            total += 1

print("Accuracy", round(correct / total, 3))


