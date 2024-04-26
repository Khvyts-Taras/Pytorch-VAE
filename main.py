from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.utils import save_image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 150
lr = 0.001
epochs = 10
img_size = 128

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_size, img_size)),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = '/data/CelebA'
dataset = CelebA(root=data_dir, split='train', transform=transform, download=True)

batch_size = 32
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):
	def __init__(self, latent_dim):
		super(Encoder, self).__init__()
		def _DownBlock(inp_n, out_n):
			block = [nn.Conv2d(inp_n, out_n, kernel_size=3, stride=1, padding=1),
					 nn.LeakyReLU(0.1),
					 nn.MaxPool2d(2)]

			return block

		self.encoder = nn.Sequential(
			#3, 128, 128
			*_DownBlock(3, 16),
			#16, 64, 64
			*_DownBlock(16, 32),
			#32, 32, 32
			*_DownBlock(32, 64),
			#64, 16, 16
			*_DownBlock(64, 128),
			#128, 8, 8
		)

		self.fc_mean = nn.Linear(128*8*8, latent_dim)
		self.fc_log_var = nn.Linear(128*8*8, latent_dim)

	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)

		mean = self.fc_mean(x)
		log_var = self.fc_log_var(x)
		return mean, log_var


class Decoder(nn.Module):
	def __init__(self, latent_dim):
		super(Decoder, self).__init__()
		def _UpBlock(inp_n, out_n):
			block = [nn.ConvTranspose2d(inp_n, out_n, kernel_size=4, stride=2, padding=1),
					 nn.LeakyReLU(0.1)]

			return block

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 128*8*8),
			nn.LeakyReLU(0.1),

			nn.Unflatten(1, (128, 8, 8)),
			#128, 8, 8
			*_UpBlock(128, 64),
			#64, 16, 16
			*_UpBlock(64, 32),
			#32, 32, 32
			*_UpBlock(32, 16),
			#16, 64, 64
			*_UpBlock(16, 3),
			#3, 128, 128
			nn.Tanh()
		)

	def forward(self, v):
		x = self.decoder(v)
		return x


class VAE(nn.Module):
	def __init__(self, latent_dim):
		super(VAE, self).__init__()
		self.encoder = Encoder(latent_dim).to(device)
		self.decoder = Decoder(latent_dim).to(device)

	def reparametrization(self, mean, log_var):
		epsilon = torch.rand_like(mean).to(device)
		v = mean + torch.exp(0.5*log_var) * epsilon
		return v

	def forward(self, x):
		mean, log_var = self.encoder(x)
		v = self.reparametrization(mean, log_var)
		res = self.decoder(v)

		return res, mean, log_var


rec_k = 0.01
kld_k = 10
def loss_f(x, x_hat, mean, log_var):
	rec_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return rec_loss*rec_k + KLD*kld_k


model = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

model.train()
for epoch in range(epochs):
	epoch_loss = 0
	for i, (img, _) in enumerate(tqdm(dataloader, desc=f'Epoch {(epoch+1)}/{epochs}')):
		optimizer.zero_grad()
		img = img.to(device)
		rec, mean, log_var = model(img)
		loss = loss_f(rec, img, mean, log_var)

		epoch_loss += loss.item()

		loss.backward()
		optimizer.step()

		if i%100 == 0:
			with torch.no_grad():
				torch.manual_seed(1)
				noise = torch.rand(batch_size, latent_dim).to(device)
				generated_images = model.decoder(noise)/2+0.5

			save_image(generated_images, f'images/image_{epoch}_{i}.png')
			torch.save(model.state_dict(), f'models/vae_{epoch}_{i}.pt')

	print(f'Epoc: {epoch+1}, Loss: {epoch_loss/len(dataloader.dataset)}')
