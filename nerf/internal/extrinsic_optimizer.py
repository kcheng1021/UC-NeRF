import torch
import numpy as np

class BrightnessCorrection(torch.nn.Module):
    def __init__(self, n_views, model_sky=False, n_dim=4):
        super().__init__()

        self.latent_code = torch.nn.Parameter(torch.zeros(size=(n_views, n_dim), dtype=torch.float32), requires_grad=True)
        self.model_sky = model_sky
        if model_sky:
            self.sky_latent_code = torch.nn.Parameter(torch.zeros(size=(n_views, 4), dtype=torch.float32), requires_grad=True)
        self.brightness_MLP = BrightnessMLP()

    
    def forward(self, indices: torch.tensor = None):
        indices = indices.squeeze().to(torch.long)
        latent_code = self.latent_code[indices]
        affine_transformation = self.brightness_MLP(latent_code).view(indices.shape[0], 3, 4)
        if self.model_sky:
            sky_latent_code = self.sky_latent_code[indices]
            affine_transformation_sky = self.brightness_MLP(sky_latent_code).view(indices.shape[0], 3, 4)

            return affine_transformation, affine_transformation_sky
        
        return affine_transformation
    
class BrightnessMLP(torch.nn.Module):
    def __init__(self, D=3, W=256, input_ch=4, output_ch=12, use_viewdirs=False):
        super(BrightnessMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.use_viewdirs = use_viewdirs

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, W)] + [torch.nn.Linear(W, W) for i in range(D-1)]
        )

        self.output_linear = torch.nn.Linear(W, output_ch)

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = torch.nn.functional.relu(x)

        outputs = self.output_linear(x)

        return outputs