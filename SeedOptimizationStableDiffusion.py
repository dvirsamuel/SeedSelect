import torch
from diffusers import StableDiffusionPipeline
from typing import Callable, List, Optional, Union
import torch.nn.functional as F


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_sd_tensors(imgs, preprocess):
    imgs = F.interpolate(imgs, size=preprocess[0].size, mode="bicubic")
    imgs = preprocess[-1](imgs)
    return imgs


class SeedOptimizationStableDiffusion(StableDiffusionPipeline):

    def __init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)
        # enable checkpointing
        self.text_encoder.gradient_checkpointing_enable()
        self.unet.enable_gradient_checkpointing()
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        # freeze diffusion parameters
        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())
        freeze_params(self.text_encoder.parameters())

    def decode_latents_tensors(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def apply(self,
              prompt: Union[str, List[str]] = None,
              text_embeddings=None,
              clip_img_centroid=None,
              img_seed=None,
              clip_model=None,
              clip_transform=None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 20,
              guidance_scale: float = 1,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
              latents: Optional[torch.FloatTensor] = None,
              callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
              callback_steps: Optional[int] = 1,
              run_raw_sd=False
              ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if text_embeddings is None:
            text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        if img_seed is None:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        else:
            latents = img_seed

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        losses = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if not run_raw_sd and i >= num_inference_steps - 3:
                    # 8. Post-processing
                    image = self.decode_latents_tensors(latents)
                    image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                    image_pil = self.numpy_to_pil(image_numpy)
                    image_pp = image.float()

                    image_preproceesed = transform_sd_tensors(image_pp, clip_transform)
                    clip_img_features = clip_model.get_image_features(image_preproceesed)
                    clip_img_features_normed = clip_img_features / clip_img_features.norm(dim=-1, keepdim=True)

                    # Calculate the cosine distance
                    cosine_sim = clip_img_features_normed @ clip_img_centroid.T
                    clip_loss = 1 - cosine_sim.mean()
                    losses += [clip_loss]
                elif i == num_inference_steps - 1:
                    image = self.decode_latents_tensors(latents)
                    image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                    image_pil = self.numpy_to_pil(image_numpy)

        return latents, losses, image_pil[0]
