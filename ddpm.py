import torch
import numpy as np


class DDPMSampler:

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120
    ):
        # Beta schedule used during training (variance schedule)
        # Values taken from Stable Diffusion v1 configuration
        # Betas are linearly spaced in sqrt-space, then squared
        self.betas = torch.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_training_steps,
            dtype=torch.float32
        ) ** 2

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # \bar{α}_t = product_{i=1..t} α_i
        # Cumulative product used in forward and reverse diffusion
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Constant tensor used when t = 0 (ᾱ_{-1} = 1)
        self.one = torch.tensor(1.0)

        # Random number generator for reproducible noise sampling
        self.generator = generator

        # Total number of diffusion steps used during training
        self.num_train_timesteps = num_training_steps

        # Default timesteps for inference (reverse order)
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        )

    def set_inference_timesteps(self, num_inference_steps=50):
        # Reduce the number of timesteps for faster inference
        # We uniformly subsample training timesteps
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps

        # Select timesteps and reverse them for denoising
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round()[::-1].copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        # Compute previous timestep index in the reduced inference schedule
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        # Compute variance σ_t^2 used when sampling x_{t-1}
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        )

        # β_t as defined in DDPM formulation
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Variance term from DDPM reverse process
        # See equations (6) and (7) in the DDPM paper
        variance = (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        ) * current_beta_t

        # Clamp to avoid numerical issues when taking log / sqrt
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """
        Controls how much noise is added to an input image (img2img use case).

        strength ≈ 1 → start from pure noise (ignore input image)
        strength ≈ 0 → minimal noise (preserve input image)
        """
        # Number of initial diffusion steps to skip
        start_step = self.num_inference_steps - int(
            self.num_inference_steps * strength
        )

        # Truncate timesteps to start from a later noise level
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor
    ):
        # Perform one reverse diffusion step: x_t → x_{t-1}
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Compute alpha / beta terms for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Predict original clean sample x_0 from predicted noise
        # See equation (15) in the DDPM paper
        pred_original_sample = (
            latents - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5

        # 3. Compute coefficients for combining x_0 and x_t
        # See equation (7) in the DDPM paper
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** 0.5 * current_beta_t
        ) / beta_prod_t

        current_sample_coeff = (
            current_alpha_t ** 0.5 * beta_prod_t_prev
        ) / beta_prod_t

        # 4. Compute the mean μ_t of p(x_{t-1} | x_t)
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # 5. Add stochastic noise according to variance schedule
        variance = 0
        if t > 0:
            device = model_output.device

            # Sample standard Gaussian noise
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype
            )

            # Scale noise by predicted variance
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # Final sampling step:
        # x_{t-1} = μ_t + σ_t * ε
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Add noise to clean samples according to forward diffusion process
        # q(x_t | x_0)

        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device,
            dtype=original_samples.dtype
        )

        timesteps = timesteps.to(original_samples.device)

        # Compute sqrt(ᾱ_t)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # Compute sqrt(1 − ᾱ_t)
        sqrt_one_minus_alpha_prod = (
            1 - alphas_cumprod[timesteps]
        ) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0)
        # Equation (4) in the DDPM paper:
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        )

        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples
