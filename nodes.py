"""
Krea2 Storyboard Generator - ComfyUI Custom Node
Simple and stable version
"""

import os
import torch


class Krea2StoryboardGenerator:
    """Generate storyboard images using Krea 2 Turbo"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": ("STRING", {"default": "krea2_turbo_fp8_scaled.safetensors"}),
                "clip_name": ("STRING", {"default": "qwen3vl_4b_fp8_scaled.safetensors"}),
                "vae_name": ("STRING", {"default": "qwen_image_vae.safetensors"}),
                "character_a": ("IMAGE",),
                "character_b": ("IMAGE",),
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
                "width": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 512, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "style_prefix": ("STRING", {"default": "Pixar-quality 3D animated movie, DreamWorks quality, stylized animation, expressive faces, soft lighting, global illumination, vibrant colors, ultra detailed, masterpiece", "multiline": True}),
            },
            "optional": {
                "lora_name": ("STRING", {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "use_upscale": ("BOOLEAN", {"default": True}),
                "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.1}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality, deformed, ugly, bad anatomy, bad hands, missing fingers", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard_images",)
    FUNCTION = "generate_storyboard"
    CATEGORY = "Krea2"
    OUTPUT_NODE = True
    
    def generate_storyboard(self, unet_name, clip_name, vae_name,
                           character_a, character_b,
                           scene_1, scene_2, scene_3, scene_4, scene_5,
                           scene_6, scene_7, scene_8, scene_9, scene_10,
                           scene_11, scene_12, scene_13, scene_14, scene_15,
                           width, height, seed, steps, cfg, denoise, style_prefix,
                           lora_name="None", lora_strength=1.0,
                           use_upscale=True, upscale_factor=1.5,
                           negative_prompt="blurry, low quality, deformed"):
        
        try:
            # Import ComfyUI nodes
            from nodes import UNETLoader, CLIPLoader, VAELoader
            from nodes import KSampler, VAEDecode, EmptySD3LatentImage
            from nodes import LatentUpscaleBy
        except ImportError as e:
            print(f"[Krea2Storyboard] Import error: {e}")
            return (torch.zeros(1, height, width, 3),)
        
        print(f"[Krea2Storyboard] Loading models...")
        
        try:
            # Load UNET
            model = UNETLoader().load_unet(unet_name, "default")[0]
            
            # Load CLIP
            clip = CLIPLoader().load_clip(clip_name, "krea2", "default")[0]
            
            # Load VAE
            vae = VAELoader().load_vae(vae_name)[0]
        except Exception as e:
            print(f"[Krea2Storyboard] Model loading error: {e}")
            return (torch.zeros(1, height, width, 3),)
        
        # Load LoRA if specified
        if lora_name and lora_name != "None":
            print(f"[Krea2Storyboard] Loading LoRA: {lora_name}")
            try:
                from nodes import LoraLoader
                lora_result = LoraLoader().load_lora(lora_name, lora_strength, lora_strength, model, clip)
                model = lora_result[0]
                clip = lora_result[1]
            except Exception as e:
                print(f"[Krea2Storyboard] Warning: Could not load LoRA: {e}")
        
        # Collect valid scenes
        scenes = []
        for i, scene in enumerate([scene_1, scene_2, scene_3, scene_4, scene_5,
                                   scene_6, scene_7, scene_8, scene_9, scene_10,
                                   scene_11, scene_12, scene_13, scene_14, scene_15], 1):
            if scene.strip():
                scenes.append((i, scene.strip()))
        
        if not scenes:
            print("[Krea2Storyboard] Error: No scenes provided!")
            return (torch.zeros(1, height, width, 3),)
        
        print(f"[Krea2Storyboard] Generating {len(scenes)} scenes...")
        
        all_images = []
        
        for scene_num, scene_text in scenes:
            print(f"[Krea2Storyboard] Scene {scene_num}: {scene_text[:50]}...")
            
            # Build prompt
            prompt = f"Krea2Edit, {scene_text}, {style_prefix}"
            
            try:
                # Encode prompt
                positive = clip.encode(prompt, pooled_output=False)[0]
                negative = clip.encode(negative_prompt, pooled_output=False)[0]
                
                # Create empty latent
                latent = EmptySD3LatentImage().generate(width, height, 1)[0]
                
                # First pass - generate base image
                scene_seed = seed + scene_num
                samples = KSampler().sample(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    seed=scene_seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=denoise
                )[0]
                
                # Optional upscale pass
                if use_upscale:
                    upscaled = LatentUpscaleBy().upscale(samples, "nearest-exact", upscale_factor)[0]
                    samples = KSampler().sample(
                        model=model,
                        positive=positive,
                        negative=negative,
                        latent_image=upscaled,
                        seed=scene_seed,
                        steps=4,
                        cfg=cfg,
                        sampler_name="euler",
                        scheduler="simple",
                        denoise=0.4
                    )[0]
                
                # Decode to image
                image = VAEDecode().decode(samples, vae)[0]
                all_images.append(image)
                
            except Exception as e:
                print(f"[Krea2Storyboard] Error generating scene {scene_num}: {e}")
                continue
        
        if not all_images:
            print("[Krea2Storyboard] Error: No images generated!")
            return (torch.zeros(1, height, width, 3),)
        
        # Stack all images
        storyboard = torch.cat(all_images, dim=0)
        
        print(f"[Krea2Storyboard] Done! Generated {len(all_images)} images.")
        
        return (storyboard,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Krea2StoryboardGenerator": Krea2StoryboardGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2StoryboardGenerator": "Krea2 Storyboard Generator",
}
