import torch
import os
from diffusers import FluxControlPipeline
from crosslift.guidance.base import VisualGuidance

PROMPT = ("A 3D render of an object with a wireframe quad mesh, equally sized quads, large "
          "quads, straight lines, only quads, evenly spaced quadrilateral lines, anti-aliased "
          "lines, sharp, no shading, technical CAD illustration, white background.")


class FluxGuidance(VisualGuidance, method_name="flux"):
    def __init__(self, cfg, device=None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.cfg = cfg

    def get_images(self, **kwargs):
        """Given a depth map, use Flux to generate an image of a quad mesh"""
        cfg = self.cfg
        device = self.device
        depths = kwargs.get("depths")
        if os.path.exists(os.path.join(cfg.run_name, "targets/guidance_images.pt")) and \
            not cfg.guidance.overwrite_target:
            images = torch.load(os.path.join(cfg.run_name, "targets/guidance_images.pt"))
        else:
            pipe = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to(device)
            prompt = [PROMPT] * cfg.render.num_views
            prompt_emb, pooled_emb, text_ids = pipe.encode_prompt(prompt, device=device)
            # Delete encoders to save memory
            pipe.text_encoder = None
            pipe.text_encoder_2 = None
            pipe.tokenizer = None
            pipe.tokenizer_2 = None
            torch.cuda.empty_cache()
            if len(prompt) > cfg.guidance.batch_size:
                all_images = []
                generator = torch.Generator().manual_seed(cfg.seed)
                for i in range(0, len(prompt), cfg.guidance.batch_size):
                    images_batch = pipe(
                        control_image=[depths[i:i+cfg.guidance.batch_size]],
                        prompt_embeds=prompt_emb[i:i+cfg.guidance.batch_size],
                        pooled_prompt_embeds=pooled_emb[i:i+cfg.guidance.batch_size],
                        guidance_scale=10,
                        generator=generator,
                        num_inference_steps=30,
                        output_type="pt",
                    ).images.to(torch.float32)
                    all_images.append(images_batch)
                images = torch.cat(all_images, dim=0)
            else:
                images = pipe(
                    control_image=[depths],
                    prompt_embeds=prompt_emb,
                    pooled_prompt_embeds=pooled_emb,
                    guidance_scale=10,
                    generator=torch.Generator().manual_seed(cfg.seed),
                    num_inference_steps=30,
                    output_type="pt",
                ).images.to(torch.float32)
            # Cache the generated torch images
            torch.save(images, os.path.join(cfg.run_name, "targets/guidance_images.pt"))
        # Save images
        self.save_images(images)

        return images
