import torch
from tqdm import tqdm

from context_unet import ContextUnet
from sprite_dataset import SpriteDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class Pipeline:
    def __init__(self):
        # network hyperparameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
        self.timesteps = 500
        self.n_feat = 64 # 64 hidden dimension feature
        self.n_cfeat = 5 # context vector is of size 5. Obviously should match the sprite_labels shape
        self.height = 16 # 16x16 image. Obviously should match the sprites shape
        self.save_dir = './weights/'
        self.batch_size = 100
        self.construct_noise()

        # create neural network
        self.nn_model = ContextUnet(in_channels=3, n_feat=self.n_feat, n_cfeat=self.n_cfeat, height=self.height).to(self.device)

    def visualise_context_sample(self, matrix):
        # mix of defined context
        ctx = torch.tensor(matrix).float().to(self.device)
        samples, _ = self.sample_ddpm_context(ctx.shape[0], ctx)
        self.show_images(samples)

    def visualise_random_sample(self):
        # visualize samples with randomly selected context
        plt.clf()
        ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=self.device).float()
        samples, intermediate = self.sample_ddpm_context(32, ctx)
        animation_ddpm_context = self.plot_sample(intermediate,32,4, self.save_dir, "ani_run", None)
        plt.show() 

    def load_data(self):
        print("-- loading custom dataset")
        transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
        dataset = SpriteDataset("./data/sprites_1788_16x16.npy", "./data/sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        print("-- loading done")

    def load_pretraining(self, filename="context_model_pretrained"):
        # load in pretrain model weights and set to eval mode
        self.nn_model.load_state_dict(torch.load(f"{self.save_dir}/{filename}.pth", map_location=self.device))
        self.nn_model.eval() 
        print("Loaded in Context Model")

    def train(self, n_epoch=32):
        # training hyperparameters
        lrate=1e-3
        optim = torch.optim.Adam(self.nn_model.parameters(), lr=lrate)

        # training with context code
        # set into train mode
        self.nn_model.train()

        for ep in range(n_epoch):
            print(f'epoch {ep}')
            
            # linearly decay learning rate
            optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
            
            pbar = tqdm(self.dataloader, mininterval=2 )
            for x, c in pbar:   # x: images  c: context
                optim.zero_grad()
                x = x.to(self.device)
                c = c.to(x)
                
                # randomly mask out c
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(self.device)
                c = c * context_mask.unsqueeze(-1)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, self.timesteps + 1, (x.shape[0],)).to(self.device)
                x_pert = self.perturb_input(x, t, noise)
                
                # use network to recover noise
                pred_noise = self.nn_model(x_pert, t / self.timesteps, c=c)
                
                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                
                optim.step()
            # save model periodically
            if ep%4==0 or ep == int(n_epoch-1):
                self.save_model(f"context_model_{ep}")
        # save model finally
        self.save_model("context_model")

    def save_model(self, file_name):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        torch.save(self.nn_model.state_dict(), self.save_dir + f"{file_name}.pth")
        print('saved model at ' + self.save_dir + f"{file_name}.pth")

    # helper function: perturbs an image to a specified noise level
    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise
    
    def construct_noise(self):
        # construct DDPM noise schedule based on these hyperparameters
        beta1 = 1e-4
        beta2 = 0.02
        self.b_t = (beta2 - beta1) * torch.linspace(0, 1, self.timesteps + 1, device=self.device) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse of the algoritm)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise
    
    # sample with context using standard algorithm
    @torch.no_grad()
    def sample_ddpm_context(self, n_sample, context, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, self.height, self.height).to(self.device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(self.timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / self.timesteps])[:, None, None, None].to(self.device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = self.nn_model(samples, t, c=context)    # predict noise e_(x_t,t, ctx)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate==0 or i==self.timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    def plot_sample(self, x_gen_store,n_sample,nrows,save_dir, fn,  w):
        ncols = n_sample//nrows
        sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
        nsx_gen_store = self.norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
        
        # create gif of images evolving over time, based on x_gen_store
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
        def animate_diff(i, store):
            print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
            plots = []
            for row in range(nrows):
                for col in range(ncols):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
            return plots
        ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=False, frames=nsx_gen_store.shape[0]) 
        return ani
    
    def norm_all(self, store, n_t, n_s):
        # runs unity norm on all timesteps of all samples
        nstore = np.zeros_like(store)
        for t in range(n_t):
            for s in range(n_s):
                nstore[t,s] = self.unorm(store[t,s])
        return nstore
    
    def unorm(self, x):
        # unity norm. results in range of [0,1]
        # assume x (h,w,3)
        xmax = x.max((0,1))
        xmin = x.min((0,1))
        return(x - xmin)/(xmax - xmin)
    
    def show_images(self, imgs, nrow=2):
        _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
        plt.show()