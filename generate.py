#!/usr/bin/env python3
"""
Krea2 Storyboard Generator - Standalone Script
Generate storyboard scenes using Krea 2 Turbo model
"""

import os
import json
import torch
from pathlib import Path

# Try to import ComfyUI modules
try:
    import folder_paths
    from nodes import (
        KSampler, VAEDecode, EmptySD3LatentImage, 
        LatentUpscaleBy, SaveImage, LoadImage
    )
    from comfy.sd import load_diffusion_model
    from comfy.clip_vision import load as load_clip_vision
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("Warning: ComfyUI not found. Running in standalone mode.")


class Krea2StoryboardGenerator:
    """Generate storyboard scenes with Krea 2 Turbo"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.model = None
        self.clip = None
        self.vae = None
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            "models": {
                "unet": "krea2_turbo_fp8_scaled.safetensors",
                "clip": "qwen3vl_4b_fp8_scaled.safetensors",
                "vae": "qwen_image_vae.safetensors"
            },
            "loras": [
                {"name": "krea2_identity_edit_v1_2.safetensors", "strength": 1.0},
                {"name": "krea2_kidsdrawing.safetensors", "strength": 1.0}
            ],
            "generation": {
                "width": 1280,
                "height": 720,
                "steps": 35,
                "cfg": 1.0,
                "denoise": 1.0,
                "seed": 0,
                "sampler": "euler",
                "scheduler": "simple"
            },
            "upscale": {
                "enabled": True,
                "factor": 1.5,
                "steps": 4,
                "denoise": 0.4
            },
            "style": {
                "prefix": "Krea2Edit",
                "quality": "Pixar-quality 3D animated movie, DreamWorks quality, stylized animation, expressive faces, soft lighting, global illumination, vibrant colors, ultra detailed, masterpiece"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_models(self):
        """Load all required models"""
        if not COMFYUI_AVAILABLE:
            print("Error: ComfyUI not available")
            return False
        
        try:
            # Load UNET
            unet_path = os.path.join(
                folder_paths.models_dir, 
                "diffusion_models", 
                "krea2",
                self.config["models"]["unet"]
            )
            print(f"Loading UNET: {unet_path}")
            self.model = load_diffusion_model(unet_path)
            
            # Load CLIP
            from nodes import CLIPLoader
            clip_loader = CLIPLoader()
            self.clip = clip_loader.load_clip(
                self.config["models"]["clip"],
                "krea2",
                "default"
            )[0]
            
            # Load VAE
            from nodes import VAELoader
            vae_loader = VAELoader()
            self.vae = vae_loader.load_vae(self.config["models"]["vae"])[0]
            
            print("All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def load_character_image(self, image_path):
        """Load character reference image"""
        if not COMFYUI_AVAILABLE:
            print(f"Would load character: {image_path}")
            return None
        
        try:
            loader = LoadImage()
            return loader.load_image(image_path)[0]
        except Exception as e:
            print(f"Error loading character image: {e}")
            return None
    
    def build_prompt(self, scene_prompt, character_a_name="", character_b_name=""):
        """Build full prompt with style and character names"""
        prefix = self.config["style"]["prefix"]
        quality = self.config["style"]["quality"]
        
        # Add character names if provided
        character_part = ""
        if character_a_name or character_b_name:
            characters = []
            if character_a_name:
                characters.append(character_a_name)
            if character_b_name:
                characters.append(character_b_name)
            character_part = ", ".join(characters) + ","
        
        # Build full prompt
        full_prompt = f"{prefix} {character_part} {scene_prompt}, {quality}"
        return full_prompt.strip()
    
    def generate_scene(self, prompt, seed=None, width=None, height=None):
        """Generate a single scene"""
        if not COMFYUI_AVAILABLE:
            print(f"Would generate scene: {prompt[:50]}...")
            return None
        
        # Use defaults from config if not specified
        if seed is None:
            seed = self.config["generation"]["seed"]
        if width is None:
            width = self.config["generation"]["width"]
        if height is None:
            height = self.config["generation"]["height"]
        
        try:
            # Create empty latent
            latent = EmptySD3LatentImage().generate(width, height, 1)[0]
            
            # Encode prompt
            positive = self.clip.encode(prompt, pooled_output=False)[0]
            negative = self.clip.encode("", pooled_output=False)[0]
            
            # First pass
            samples = KSampler().sample(
                model=self.model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                seed=seed,
                steps=self.config["generation"]["steps"],
                cfg=self.config["generation"]["cfg"],
                sampler_name=self.config["generation"]["sampler"],
                scheduler=self.config["generation"]["scheduler"],
                denoise=self.config["generation"]["denoise"]
            )[0]
            
            # Optional upscale
            if self.config["upscale"]["enabled"]:
                upscaled = LatentUpscaleBy().upscale(
                    samples, 
                    "nearest-exact", 
                    self.config["upscale"]["factor"]
                )[0]
                
                samples = KSampler().sample(
                    model=self.model,
                    positive=positive,
                    negative=negative,
                    latent_image=upscaled,
                    seed=seed,
                    steps=self.config["upscale"]["steps"],
                    cfg=self.config["generation"]["cfg"],
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=self.config["upscale"]["denoise"]
                )[0]
            
            # Decode to image
            image = VAEDecode().decode(samples, self.vae)[0]
            return image
            
        except Exception as e:
            print(f"Error generating scene: {e}")
            return None
    
    def generate_storyboard(self, scenes, character_a_path=None, character_b_path=None,
                           character_a_name="", character_b_name="", output_dir="output"):
        """Generate complete storyboard from list of scenes"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load character images if provided
        character_a = None
        character_b = None
        
        if character_a_path:
            character_a = self.load_character_image(character_a_path)
        if character_b_path:
            character_b = self.load_character_image(character_b_path)
        
        # Generate each scene
        all_images = []
        base_seed = self.config["generation"]["seed"]
        
        for i, scene in enumerate(scenes):
            print(f"\n=== Generating Scene {i+1}/{len(scenes)} ===")
            print(f"Prompt: {scene[:80]}...")
            
            # Build full prompt
            full_prompt = self.build_prompt(scene, character_a_name, character_b_name)
            print(f"Full prompt: {full_prompt[:100]}...")
            
            # Generate with unique seed
            scene_seed = base_seed + i
            image = self.generate_scene(full_prompt, seed=scene_seed)
            
            if image is not None:
                all_images.append(image)
                
                # Save individual scene
                filename = f"scene_{i+1:03d}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save using ComfyUI's SaveImage
                if COMFYUI_AVAILABLE:
                    SaveImage().save_images(
                        image, 
                        filename_prefix=f"scene_{i+1:03d}"
                    )
                
                print(f"✓ Saved: {filepath}")
            else:
                print(f"✗ Failed to generate scene {i+1}")
        
        # Create combined storyboard image
        if all_images:
            print(f"\n=== Creating Storyboard Grid ===")
            storyboard = torch.cat(all_images, dim=0)
            
            # Save combined storyboard
            if COMFYUI_AVAILABLE:
                SaveImage().save_images(
                    storyboard,
                    filename_prefix="storyboard_complete"
                )
            
            print(f"✓ Storyboard saved with {len(all_images)} scenes")
        
        return all_images


def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Krea2 Storyboard Generator")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--scenes", nargs="+", help="List of scene prompts")
    parser.add_argument("--scenes-file", help="File containing scene prompts (one per line)")
    parser.add_argument("--character-a", help="Path to character A image")
    parser.add_argument("--character-b", help="Path to character B image")
    parser.add_argument("--character-a-name", default="Character A", help="Name for character A")
    parser.add_argument("--character-b-name", default="Character B", help="Name for character B")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load scenes
    scenes = []
    
    if args.scenes:
        scenes = args.scenes
    elif args.scenes_file:
        with open(args.scenes_file, 'r', encoding='utf-8') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        # Example scenes
        scenes = [
            "Character A walking through a magical forest",
            "Character B appearing from behind a tree",
            "Both characters discovering a hidden portal",
            "Entering the portal together",
            "Arriving in a colorful fantasy world",
            "Meeting friendly creatures",
            "Exploring a crystal cave",
            "Finding a treasure chest",
            "Opening the chest to reveal magical items",
            "Character A using magic powers",
            "Character B flying with magical wings",
            "Both characters dancing in the rain",
            "Watching a beautiful sunset",
            "Returning through the portal",
            "Waving goodbye to new friends"
        ]
        print("Using example scenes. Provide --scenes or --scenes-file for custom scenes.")
    
    print(f"\n{'='*50}")
    print("Krea2 Storyboard Generator")
    print(f"{'='*50}")
    print(f"Scenes: {len(scenes)}")
    print(f"Character A: {args.character_a or 'Not provided'}")
    print(f"Character B: {args.character_b or 'Not provided'}")
    print(f"Output: {args.output}")
    print(f"{'='*50}\n")
    
    # Create generator
    generator = Krea2StoryboardGenerator(args.config)
    
    # Load models
    if COMFYUI_AVAILABLE:
        if not generator.load_models():
            print("Failed to load models. Exiting.")
            return
    
    # Generate storyboard
    generator.generate_storyboard(
        scenes=scenes,
        character_a_path=args.character_a,
        character_b_path=args.character_b,
        character_a_name=args.character_a_name,
        character_b_name=args.character_b_name,
        output_dir=args.output
    )
    
    print(f"\n{'='*50}")
    print("Storyboard generation complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
