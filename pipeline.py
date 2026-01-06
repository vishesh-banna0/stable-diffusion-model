import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# Image resolution (Stable Diffusion operates in latent space at 1/8 resolution)
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        # Validate img2img strength
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Utility to move models back to idle device (e.g., CPU) when not in use
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize RNG for reproducibility
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # ---------------------------------------------------------
        # TEXT ENCODING (CLIP)
        # ---------------------------------------------------------
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Encode conditional prompt
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            # Encode unconditional prompt (for classifier-free guidance)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # Concatenate contexts for a single forward pass
            # Shape: (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Encode prompt without CFG
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        # Move CLIP back to idle device
        to_idle(clip)

        # ---------------------------------------------------------
        # SAMPLER SETUP
        # ---------------------------------------------------------
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # ---------------------------------------------------------
        # LATENT INITIALIZATION (txt2img or img2img)
        # ---------------------------------------------------------
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            # Preprocess input image
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )

            # Rescale pixel values from [0, 255] → [-1, 1]
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # Add batch dimension and convert to channels-first
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Sample noise for VAE encoder
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )

            # Encode image into latent space
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise according to img2img strength
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # txt2img: start directly from pure Gaussian noise
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # ---------------------------------------------------------
        # DIFFUSION SAMPLING LOOP
        # ---------------------------------------------------------
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Compute sinusoidal timestep embedding
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare UNet input
            model_input = latents

            if do_cfg:
                # Duplicate latents so conditional and unconditional
                # predictions can be computed in one forward pass
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise ε_θ(x_t, t, c)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Split conditional and unconditional predictions
                output_cond, output_uncond = model_output.chunk(2)

                # Apply classifier-free guidance
                model_output = (
                    cfg_scale * (output_cond - output_uncond) + output_uncond
                )

            # Compute x_{t-1} from x_t using the sampler
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # ---------------------------------------------------------
        # DECODE LATENTS TO IMAGE SPACE
        # ---------------------------------------------------------
        
        decoder = models["decoder"]
        decoder.to(device)

        # Decode latents back to RGB image
        images = decoder(latents)

        to_idle(decoder)

        # Rescale from [-1, 1] → [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # Convert to HWC format and uint8
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
    

def rescale(x, old_range, new_range, clamp=False):
    # Linearly rescale tensor values from old_range to new_range
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep):
    # Compute sinusoidal timestep embedding (as in Transformer positional encoding)
    # Output shape: (1, 320)

    freqs = torch.pow(
        10000,
        -torch.arange(start=0, end=160, dtype=torch.float32) / 160
    )

    # Multiply timestep with frequencies
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # Concatenate cosine and sine embeddings
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
