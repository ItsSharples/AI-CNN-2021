

import os
import torch
from torch import nn

import torch.nn.functional as F
from torchvision.datasets import ImageFolder
#from torch.utils.data import DataLoader
from torchvision import transforms

import timeit
import copy
import matplotlib.pyplot as plt
import numpy as np
#import numpy.plt as plt

class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self):
      super().__init__()
      
      self.size = 13 * 13 * 5 * 4
      
      #64x64x1
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      #32x32x10
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      #16x16x20
      self.conv2_drop = nn.Dropout2d(0.6)
      self.fc1 = nn.Linear(self.size, 32)
      self.fc2 = nn.Linear(32, 10)

    
    # self.layers = nn.Sequential(
    #   nn.Flatten(), # Turn into a 1D Structure
    #   nn.Linear(width * height * depth, 64), # Feed into a struct that takes the raw Image and spits out 64 values
    #   nn.ReLU(), # Rectify
    #   nn.Linear(64, 32), # Feed in again, down to 32 values
    #   nn.ReLU(), # Rectify
    #   nn.Linear(32, 10) # Feed in one last time, down to 10
    # )


  # def forward(self, x):
  #   '''Forward pass'''
  #   return self.layers(x)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def _main_(device = ""):
    print("Import Complete")
    
    n_epochs = 16
    batch_size_train = 4
    batch_size_test = 4
    learning_rate = 0.01
    momentum = 0.5
    
    log_interval = 5
    
    torch.backends.cudnn.enabled = True
    # Set fixed random number seed
    random_seed = 42
    torch.manual_seed(random_seed)
    print("Seeded Torch")
    
      # Prepare CIFAR-10 dataset
    #dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    
    data_transforms = {
    'train' : transforms.Compose([
        transforms.Resize(64),
          transforms.CenterCrop(64),
          transforms.Grayscale(),
          transforms.ToTensor()
      ]),
    'test' : transforms.Compose([
            transforms.Resize(64),
          transforms.CenterCrop(64),
          transforms.Grayscale(),
          transforms.ToTensor()
      ])
    
    }
    
    location = r"E:\Work\AI\Images"
    #dataset = ImageFolder(location, transform = data_transform)
    #trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    
    
    image_datasets = {x: ImageFolder(os.path.join(location, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
    #                                              shuffle=True, num_workers=4)
    #              for x in ['train', 'val']}
    
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size_train,
                                                  shuffle=True, num_workers=1)
    
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size_test,
                                                  shuffle=True, num_workers=1)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print("Loaded Dataset")
    
    mean = torch.zeros(1)
    std = torch.zeros(1)
    N_CHANNELS = 1
    print('Computing mean and std...')
    for inputs, _labels in train_loader:
        for i in range(N_CHANNELS):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(image_datasets['train']))
    std.div_(len(image_datasets['train']))
    print(mean, std)
    print("Found Mean and STD")
    
    if device == "":
        # Work out if we can use the GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Initialize the MLP
    model = MLP()
    model.to(device)
    model.cuda()
    # Define the loss function and optimizer
    #criterion = F.nll_los
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    
    
    # Data
    train_losses = []
    train_counter = []
    test_losses = []
    #test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    x_prev = 0
    
    def train(epoch, device, model):
        model.train()  # Set model to training mode

        
        len_dataset= len(train_loader.dataset)
        # Iterate over the DataLoader for training data
        for batch_no, data in enumerate(train_loader, 0):
          
            # Get inputs
            inputs, targets = data
            # Send to the right place
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = model(inputs)
            
            # Compute loss
            #loss = criterion(outputs, targets)
            loss = F.nll_loss(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            #current_loss += loss.item()
            if batch_no % log_interval == 0:
                print(f"Epoch: {epoch}, Batch: {batch_no}")
                #print('Loss after mini-batch %5d: %.3f' %
                #      (i + 1, current_loss / 500))
                train_losses.append(loss.item())
                train_counter.append((batch_no*batch_size_train) + ((epoch-1)*len_dataset))
                torch.save(model.state_dict(), r'.\results\model.pth')
                torch.save(optimizer.state_dict(), r'.\results\optimizer.pth')
              
    def test(device, model):
        model.eval()  # Set model to training mode
        
        test_loss = 0
        correct = 0
        
        len_dataset = len(test_loader.dataset)
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                test_loss += F.nll_loss(output, targets, size_average=False).item()
                prediction = output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(targets.data.view_as(prediction)).sum()
        test_loss /= len_dataset
        test_losses.append(test_loss)
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len_dataset} ({100. * correct / len_dataset:.0f}%)\n')

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
    
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])
    
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
    
    test(device, model)
    for epoch in range(1, n_epochs + 1):
        train(epoch, device, model)
        test(device, model)
    #train_model(model, criterion, optimizer, scheduler)
    visualize_model(model)
    # for epoch in range(1,6):
    #     print(f"Starting Epoch {epoch}")
    #     epoch_func(device)
        
      
    print("Training Complete")
    
    plt.plot(train_losses)
    plt.plot(test_losses)
    print(train_losses)
    print(train_counter)
    print(test_losses)
    
if __name__ == '__main__':
    _main_()
     