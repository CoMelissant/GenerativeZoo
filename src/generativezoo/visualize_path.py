import torch
from models.Diffusion.Diffusion import *
import numpy as np
from config import data_raw_dir, models_dir
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToPILImage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

class mod_Sampler():
    def __init__(self, betas, sqrt_one_minus_alphas_cumprod, sqrt_one_by_alphas, posterior_variance, timesteps):
        self.betas = betas
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.sqrt_one_by_alphas = sqrt_one_by_alphas
        self.posterior_variance = posterior_variance
        self.timesteps = timesteps
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = extract_time_index(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_time_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_one_by_alphas_t = extract_time_index(self.sqrt_one_by_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_one_by_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract_time_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

# get entire mnist dataset as a numpy array
def get_mnist_dataset():
    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=60000, 
                                 shuffle=True,
                                 pin_memory=True)
    for batch_idx, (data, target) in enumerate(training_loader):
        return data.numpy(), target.numpy()
    
reverse_transform = Compose([
     #Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     #Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.float32)),
     #ToPILImage(),
])

beta_start = 0.0001
beta_end = 0.02
timesteps = 300
image_size = 28
num_channels = 1
n_samples = 6000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = os.path.join(models_dir, "DDPM.pt")

dataset, target = get_mnist_dataset()
dataset = dataset.reshape(60000, 784)

model = DDPM(n_features=image_size, in_channels=num_channels, channel_scale_factors=(1, 2, 4,)).to(device)
model.load_pretrained_weights(checkpoint_dir)

scheduler = LinearScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
forward_diffusion = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)
sampler = mod_Sampler(betas=scheduler.betas, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, sqrt_one_by_alphas=scheduler.sqrt_one_by_alphas, posterior_variance=scheduler.posterior_variance, timesteps=timesteps)

samples = sampler.sample(model=model, image_size=image_size, batch_size=1000, channels=num_channels)
for i in tqdm(range(1, n_samples//1000)):
    aux = sampler.sample(model=model, image_size=image_size, batch_size=1000, channels=num_channels)
    for j in range(len(aux)):
        samples[j] = np.concatenate((samples[j], aux[j]), axis=0)

pca = PCA(n_components=1)
pca.fit(dataset)
map = np.zeros((n_samples, timesteps),dtype=np.float32)
k = 0
for i in range(len(samples)-1,-1,-1):
    path = samples[i].reshape(n_samples, 784)
    embeddings = pca.transform(path)
    map[:,k] = embeddings[:,0]
    k = k + 1

plt.figure(figsize=(30,10))
for i in range(map.shape[0]-2):
    plt.plot(map[i,:], linewidth=0.2, color='green')
plt.plot(map[map.shape[0]-2,:], linewidth=2, color='red')
plt.plot(map[map.shape[0]-1,:], linewidth=2, color='blue')
plt.xlabel('Timesteps')
# remove y axis
plt.yticks([])
plt.savefig('./Timesteps.png')
plt.close()

# save blue and red samples
plt.figure(figsize=(5,5))
plt.imshow(((samples[-1][-2]+1)/2).reshape(28,28), cmap='gray')
plt.savefig('./Blue.png')
plt.close()

plt.figure(figsize=(5,5))
plt.imshow(((samples[-1][-1]+1)/2).reshape(28,28), cmap='gray')
plt.savefig('./Red.png')
plt.close()

for i in range(0, map.shape[1], 20):
    # range should be maximum and minimum of map in this histogram
    plt.hist(map[:,i], bins=100)
    # range of x axis should be maximum and minimum of map
    plt.xlim(np.min(map), np.max(map))
    plt.ylim(0, 0.1*map.shape[0])
    # remove axis
    plt.yticks([])
    plt.title('Timestep: ' + str(i))
    plt.savefig('./Histograms_' + str(i) + '.png')
    plt.close()



