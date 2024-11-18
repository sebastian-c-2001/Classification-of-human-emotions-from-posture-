# bibliotecile
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot
from skimage import io, color
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

cale = 'mpii_dataset_17000.csv'

def read_cell(csv_filename, row_index, column_index):
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == row_index:
                return row[column_index]
transform = transforms.Compose([transforms.ToTensor(), ])
device = torch.device('cuda')
print(device)

a = pd.read_csv(cale,delimiter=',')
# for i in range(0,100,1):
#
#     print("Poza de index ",i, " este ", a.iloc[i, 1])
# print("Poza de index 5578",a.iloc[5578, 1] )
# print(a.shape)

# def plot_image_with_labels(data,model=None,nr=2,nc=3,figsize=(15,15),training_data=False):
#   if training_data:
#     fig,axs = plt.subplots(nrows=nr,ncols=nc,figsize=figsize)
#     for item in data:
#         image, label = item
#
#         for i in range(nr*nc):
#           image_plot = axs[i//nc, i%nc].imshow(np.squeeze(image[i][0]),cmap='viridis')
#
#           label_p = label[i] * 224
#           axs[i//nc,i%nc].scatter(label_p[0::2],label_p[1::2],s=10,c='r')
#           axs[i//nc, i%nc].axis('off')
#         break
#     plt.show();
#
#   # Define a transform to normalize the image only testing image.
#   else:
#     transform = transforms.Compose([
#       transforms.ToTensor()])
#     fig,axs = plt.subplots(nrows=nr,ncols=nc,figsize=figsize)
#     if (nr*nc) <= len(data) or 1==1 :
#       for i in range(nr*nc):
#         x_test = shuffle(data)
#         image = x_test[i]
#         image = transform(image)
#         # Pass the image through the model to get the predicted keypoints
#         with torch.no_grad():
#           model.eval()
#           output = model(image.unsqueeze(0).to(device))  # add a batch dimension
#
#         output  = output.squeeze().cpu().numpy() * 255 + 255
#         # Reshape the predicted keypoints into a numpy array
#         label_p = output.reshape(-1, 2)
#         image_plot = axs[i//nc, i%nc].imshow(image.permute(1, 2, 0),cmap='gray')
#         axs[i//nc,i%nc].scatter(label_p[:,0],label_p[:,1],s=10,c='r')
#         axs[i//nc, i%nc].axis('off')
#       #PROGRAMUL BUN
#         print(1)
#         plt.show();
#         break

class MyDataset(Dataset):
    def __init__(self, annotation_file, transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform

        # self.target_transform = target_transform

    # This method return as whole length of dataset
    def __len__(self):
        #return len(self.img_labels)
        return 16896
    # And in this function to return as single data instance by index point.
    def __getitem__(self, index):
        img_list = []
       # print(self.img_labels.iloc[index, 1])
        # image = Image.open(str(self.img_labels.iloc[index, 1])).convert('RGB')
        image_path = self.img_labels.iloc[index, 1]  # Convert image path to string

        #print("Indexul este ", index, "Calea este ", image_path)

        image = Image.open(image_path).convert('RGB')

        for i in range(0, 32, 1):
            x = float(self.img_labels.iloc[index, i + 2])
            # x = x.astype(np.float32)
            if x != -1:
                x = x / 224
            # x=x.astype(np.float32)
            img_list.append(x)

        if self.transform:
            image = self.transform(image)
            img_list = torch.tensor(img_list)
        return image, img_list


antrenare = MyDataset(cale, transform=transform)

print(antrenare[5578])

train_loader = DataLoader(antrenare, batch_size=64, shuffle=True)


class DFKModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding='valid')
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=1, padding='valid')
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=1, padding='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(5, 5), stride=1, padding='valid')
        self.conv4_4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(5, 5), stride=1, padding='same')
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.dropout1 = nn.Dropout(0.5)
        # 96800
        #204800
        self.linear1 = nn.Linear(in_features=51200, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=500)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=500, out_features=250)
        self.dropout3 = nn.Dropout(0.5)
        self.linear4 = nn.Linear(in_features=250, out_features=32)

        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool3(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4_4(x))
        x = self.maxpool4(x)

        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.relu(self.linear3(x))
        x = self.dropout3(x)
        x= F.relu(self.linear4(x))
        return x


model = DFKModel()

# See model architacture look like.
#print(model)
#
# from torchview import draw_graph
# model_graph = draw_graph(DFKModel(), input_size=(1,3,224,224),expand_nested=True,save_graph=True)


model.to(device)

criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

num_epochs = 200
print("Incepe antrenarea")

best_loss = float('inf')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images).to(device)
        loss = torch.sqrt(criterion(outputs, labels))
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Print Epoch, Step and Loss value.
        # if (i+1) % 100 == 67:

        if (i+1) == len(train_loader):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                   loss.item()))
            if loss.item() < best_loss:
                # Update best_loss
                best_loss = loss.item()
                # Save the model
                torch.save(model.state_dict(), 'best_model_8sc.pth')
                print("Model saved with loss: {:.4f}".format(best_loss))


print("Gata antrenarea \n Se salveaza modelul... ")

torch.save(model.state_dict(), 'model.pth')

# # Partea de testing
# # #
# def predict(image_path, model, device):
#     img=io.imread(image_path);
#     img=img.astype(np.float32)
#
#
# model = DFKModel()  # Assuming DFKModel is your model class
# model.load_state_dict(torch.load('best_model.pth'))
#
# if torch.cuda.is_available():
#     model.cuda()
#
# def grayscale_to_rgb_1d(grayscale_image):
#
#   return [[intensity, intensity, intensity] for intensity in grayscale_image]
#
# # Example usage
#
# # rgb_image = grayscale_to_rgb_1d(grayscale_image)
# # print(rgb_image)
#
#
# # Calculați ieșirea modelului
# #output = model(example_input)
#
# name="096640930.jpg"
#
# image=io.imread(name)
#
# image=image.astype(np.float32)
# list=[]
#
# # image1=io.imread("sebi0.jpeg")
# # image1 = grayscale_to_rgb_1d(image1)
# # image1 =image1.astype(np.float32)
#
# for i in range(0,224,1):
#     for j in range (0,224,1):
#         image[i,j]=image[i,j]/255.0
#         list1=[image[i,j],image[i,j],image[i,j]]
#         list.append(list1)
#
# # for i in range(0, 224, 1):
# #     for j in range(0, 224, 1):
# #         image1[i, j] = image1[i, j]/ 255.0
# #         list.append(image1[i, j])
#
#
# X_test=np.vstack(list)
# X_test=X_test.reshape(-1,224,224,3)
#
# #X_test=X_test.cuda()
# #Convert the numpy array to a PyTorch tensor
# #X_test_tensor = torch.tensor(X_test)
#
# # Move the tensor to the GPU if available
# # if torch.cuda.is_available():
# #     X_test_tensor = X_test_tensor.('cuda')
# #     print(1)
#
# #plot_image_with_labels(data=X_test,model=model,nr=2,nc=2,figsize=(15,15))
#
# image = X_test[0]
# image = transform(image)
#
#
#
# with torch.no_grad():
#     model.eval()
#     output = model(image.unsqueeze(0).to(device))  # add a batch dimension
# print(output*224)
#
# make_dot(model(image.unsqueeze(0).to(device)), params=dict(model.named_parameters())).render("model_schema",  format="pdf")
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# # Load the image
# image = mpimg.imread(name)
# x_points=[]
# y_points=[]
# for i in range (32):
#     if i%2 == 0:
#         x_points.append(float(output[0][i])*224)
#     else:
#         y_points.append(float(output[0][i])*224)
#
#
# # Define your x and y coordinates as lists
#
# # Plot the image
# plt.imshow(image)
#
# # Plot the points with markers and labels (optional)
# for i, (x, y) in enumerate(zip(x_points, y_points)):
#     if x>0 and y>0:
#         plt.plot(x, y, 'o', markersize=5, color='red')
#
# for i in range (0,17000,1):
#     if str(read_cell(cale,i,1)) == name:
#
#         for j in range (2, 34,2):
#             x_de_normat=int(float(read_cell(cale,i,j)))
#             y_de_normat=int(float(read_cell(cale,i,j+1)))
#             plt.plot(x_de_normat, y_de_normat, 'o', markersize=4, color='yellow')
#         break
# # Add labels and title (optional)
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Image with Points")
#
# # Display the plot
# plt.legend()
# plt.show()
