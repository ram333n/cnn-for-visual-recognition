import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from einops.layers.torch import Rearrange

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def G_train(G, D, batch_size, z_dim, criterion, G_optimizer):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(batch_size, z_dim).to(device)
    y = torch.ones(batch_size, 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def D_train(x, G, D, z_dim, criterion, D_optimizer):

    D.zero_grad()
    batch_size = x.shape[0]
    # train discriminator on real
    x_real, y_real = x, torch.ones(batch_size, 1, 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake
    z = torch.randn(batch_size, z_dim).to(device)

    with torch.no_grad():
        x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
        Rearrange('c h w -> c (h w)')
    ])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    z_dim = 64
    mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

    G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    G_lr = 0.0002
    D_lr = 0.0002
    G_optimizer = optim.Adam(G.parameters(), lr=G_lr)
    D_optimizer = optim.Adam(D.parameters(), lr=D_lr)

    n_epoch = 200
    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            D_losses.append(D_train(x, G, D, z_dim, criterion, D_optimizer))

            if batch_idx % 2 == 0:
                G_losses.append(G_train(G, D, batch_size, z_dim, criterion, G_optimizer))

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        # Save trained parameters of model
        torch.save(G.state_dict(), 'G.pth')
        torch.save(D.state_dict(), 'D.pth')

        if epoch % 10 == 0:
            # Visualize generated data
            z = torch.randn(64, z_dim).to(device)

            # Generate image from z
            generated_images = G(z)

            # Make the images as grid
            generated_images = make_grid(generated_images.view(-1, 1, 28, 28), nrow=8, normalize=True)
            os.makedirs('./results', exist_ok=True)
            # Save the generated torch tensor models to disk
            save_image(generated_images, f'./results/gen_img{epoch}.png')
