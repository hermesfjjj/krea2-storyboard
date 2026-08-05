"""
Krea 2 Turbo Storyboard Generator
ComfyUI Custom Node for generating storyboard scenes with Krea 2 Turbo model
"""

import os
import json
import time
import uuid
from pathlib import Path
import folder_paths
from aiohttp import web
from server import PromptServer

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBFOLDER = "krea2-storyboard"


class Krea2StoryboardLoader:
    """Load Krea 2 Turbo models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (sorted([f for f in os.listdir(os.path.join(folder_paths.models_dir, "diffusion_models", "krea2")) 
                                     if f.endswith(('.safetensors', '.ckpt'))]),),
                "clip_name": (sorted([f for f in os.listdir(os.path.join(folder_paths.models_dir, "text_encoders")) 
                                     if f.endswith(('.safetensors', '.ckpt'))]),),
                "vae_name": (sorted([f for f in os.listdir(os.path.join(folder_paths.models_dir, "vae")) 
                                    if f.endswith(('.safetensors', '.ckpt'))]),),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_models"
    CATEGORY = "Krea2/Storyboard"
    
    def load_models(self, unet_name, clip_name, vae_name):
        from comfy.sd import load_diffusion_model
        from comfy.clip_vision import load as load_clip_vision
        from nodes import VAELoader, CLIPLoader
        
        # Load UNET
        unet_path = os.path.join(folder_paths.models_dir, "diffusion_models", "krea2", unet_name)
        model = load_diffusion_model(unet_path)
        
        # Load CLIP
        clip_loader = CLIPLoader()
        clip = clip_loader.load_clip(clip_name, "krea2", "default")[0]
        
        # Load VAE
        vae_loader = VAELoader()
        vae = vae_loader.load_vae(vae_name)[0]
        
        return (model, clip, vae)


class Krea2StoryboardCharacterSheet:
    """Upload 2 character reference images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_a": ("IMAGE",),
                "character_b": ("IMAGE",),
            },
            "optional": {
                "character_a_name": ("STRING", {"default": "Character A", "multiline": False}),
                "character_b_name": ("STRING", {"default": "Character B", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("character_a", "character_b", "name_a", "name_b")
    FUNCTION = "process_characters"
    CATEGORY = "Krea2/Storyboard"
    
    def process_characters(self, character_a, character_b, character_a_name="Character A", character_b_name="Character B"):
        return (character_a, character_b, character_a_name, character_b_name)


class Krea2StoryboardPromptBuilder:
    """Build storyboard prompt with 15 scenes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_1": ("STRING", {"multiline": True, "default": ""}),
                "scene_2": ("STRING", {"multiline": True, "default": ""}),
                "scene_3": ("STRING", {"multiline": True, "default": ""}),
                "scene_4": ("STRING", {"multiline": True, "default": ""}),
                "scene_5": ("STRING", {"multiline": True, "default": ""}),
                "scene_6": ("STRING", {"multiline": True, "default": ""}),
                "scene_7": ("STRING", {"multiline": True, "default": ""}),
                "scene_8": ("STRING", {"multiline": True, "default": ""}),
                "scene_9": ("STRING", {"multiline": True, "default": ""}),
                "scene_10": ("STRING", {"multiline": True, "default": ""}),
                "scene_11": ("STRING", {"multiline": True, "default": ""}),
                "scene_12": ("STRING", {"multiline": True, "default": ""}),
                "scene_13": ("STRING", {"multiline": True, "default": ""}),
                "scene_14": ("STRING", {"multiline": True, "default": ""}),
                "scene_15": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "style_prefix": ("STRING", {"default": "Pixar-quality 3D animated movie, DreamWorks quality, stylized animation, expressive faces, soft lighting, global illumination, vibrant colors, ultra detailed, masterpiece", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality, deformed, ugly, bad anatomy, bad hands, missing fingers", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("positive_prompts", "negative_prompt", "scene_count")
    FUNCTION = "build_prompts"
    CATEGORY = "Krea2/Storyboard"
    
    def build_prompts(self, scene_1, scene_2, scene_3, scene_4, scene_5,
                      scene_6, scene_7, scene_8, scene_9, scene_10,
                      scene_11, scene_12, scene_13, scene_14, scene_15,
                      style_prefix="", negative_prompt=""):
        
        scenes = [scene_1, scene_2, scene_3, scene_4, scene_5,
                  scene_6, scene_7, scene_8, scene_9, scene_10,
                  scene_11, scene_12, scene_13, scene_14, scene_15]
        
        # Filter out empty scenes
        valid_scenes = [s for s in scenes if s.strip()]
        
        # Build full prompts with style prefix
        full_prompts = []
        for scene in valid_scenes:
            if style_prefix:
                full_prompt = f"Krea2Edit, {scene}, {style_prefix}"
            else:
                full_prompt = f"Krea2Edit, {scene}"
            full_prompts.append(full_prompt)
        
        # Join with separator for batch processing
        positive_prompts = "|||".join(full_prompts)
        
        return (positive_prompts, negative_prompt, len(valid_scenes))


class Krea2StoryboardGenerator:
    """Generate storyboard images using Krea 2 Turbo"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "character_a": ("IMAGE",),
                "character_b": ("IMAGE",),
                "positive_prompts": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 512, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "use_upscale": ("BOOLEAN", {"default": True}),
                "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.1}),
                "ref_boost": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard_images",)
    FUNCTION = "generate_storyboard"
    CATEGORY = "Krea2/Storyboard"
    OUTPUT_NODE = True
    
    def generate_storyboard(self, model, clip, vae, character_a, character_b,
                           positive_prompts, negative_prompt, width, height,
                           seed, steps, cfg, denoise, use_upscale=True,
                           upscale_factor=1.5, ref_boost=4.0, lora_strength=1.0):
        
        from nodes import KSampler, VAEDecode, EmptySD3LatentImage, LatentUpscaleBy
        from comfy.samplers import KSampler as ComfyKSampler
        
        # Split prompts by separator
        prompts = positive_prompts.split("|||")
        prompts = [p.strip() for p in prompts if p.strip()]
        
        all_images = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating scene {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Create latent image
            latent = EmptySD3LatentImage().generate(width, height, 1)[0]
            
            # Encode prompt with character references
            # Using simple conditioning for now - can be enhanced with Krea2EditGroundedEncode
            positive = clip.encode(prompt, pooled_output=False)[0]
            negative = clip.encode(negative_prompt, pooled_output=False)[0]
            
            # First pass - generate base image
            seed_i = seed + i
            samples = ComfyKSampler().sample(
                model=model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                seed=seed_i,
                steps=steps,
                cfg=cfg,
                sampler_name="euler",
                scheduler="simple",
                denoise=denoise
            )[0]
            
            # Optional upscale pass
            if use_upscale:
                upscaled = LatentUpscaleBy().upscale(samples, "nearest-exact", upscale_factor)[0]
                samples = ComfyKSampler().sample(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent_image=upscaled,
                    seed=seed_i,
                    steps=4,
                    cfg=cfg,
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=0.4
                )[0]
            
            # Decode to image
            image = VAEDecode().decode(samples, vae)[0]
            all_images.append(image)
        
        # Stack all images
        import torch
        storyboard = torch.cat(all_images, dim=0)
        
        return (storyboard,)


class Krea2StoryboardPreview:
    """Preview and save storyboard"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "storyboard"}),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview_storyboard"
    CATEGORY = "Krea2/Storyboard"
    OUTPUT_NODE = True
    
    def preview_storyboard(self, images, filename_prefix="storyboard"):
        from nodes import SaveImage
        
        results = []
        for i in range(images.shape[0]):
            img = images[i:i+1]
            filename = f"{filename_prefix}_{i+1:03d}"
            SaveImage().save_images(img, filename_prefix=filename)
            results.append({"filename": f"{filename}.png", "subfolder": "", "type": "output"})
        
        return {"results": results}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Krea2StoryboardLoader": Krea2StoryboardLoader,
    "Krea2StoryboardCharacterSheet": Krea2StoryboardCharacterSheet,
    "Krea2StoryboardPromptBuilder": Krea2StoryboardPromptBuilder,
    "Krea2StoryboardGenerator": Krea2StoryboardGenerator,
    "Krea2StoryboardPreview": Krea2StoryboardPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2StoryboardLoader": "Krea2 Storyboard Loader",
    "Krea2StoryboardCharacterSheet": "Krea2 Character Sheet",
    "Krea2StoryboardPromptBuilder": "Krea2 Storyboard Prompts",
    "Krea2StoryboardGenerator": "Krea2 Storyboard Generator",
    "Krea2StoryboardPreview": "Krea2 Storyboard Preview",
}
