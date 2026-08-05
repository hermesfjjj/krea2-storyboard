# Krea2 Storyboard Generator

A ComfyUI custom node for generating professional storyboard scenes using Krea 2 Turbo model.

## Features

- 🎬 Generate 15 storyboard scenes at once
- 👥 Support for 2 character reference images
- 🖼️ 16:9 aspect ratio (cinematic)
- ⚡ Krea 2 Turbo fast generation
- 🎨 Pixar-quality 3D animation style

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/krea2-storyboard.git
```

2. Install required models (see below)

3. Restart ComfyUI

## Required Models

### Diffusion Model
- `krea2_turbo_fp8_scaled.safetensors` (12.2 GB)
- Place in: `ComfyUI/models/diffusion_models/krea2/`

### Text Encoder (CLIP)
- `qwen3vl_4b_fp8_scaled.safetensors` (4.8 GB)
- Place in: `ComfyUI/models/text_encoders/`

### VAE
- `qwen_image_vae.safetensors` (242 MB)
- Place in: `ComfyUI/models/vae/`

### LoRAs (Optional)
- `krea2_identity_edit_v1_2.safetensors` (for character consistency)
- `krea2_kidsdrawing.safetensors` (for kids drawing style)
- Place in: `ComfyUI/models/loras/krea2/`

## Nodes

### 1. Krea2 Storyboard Loader
Load all required models (UNET, CLIP, VAE).

### 2. Krea2 Character Sheet
Upload 2 character reference images for consistent character generation.

### 3. Krea2 Storyboard Prompts
Input 15 scene descriptions for your storyboard.

### 4. Krea2 Storyboard Generator
Main generation node - creates all storyboard scenes.

### 5. Krea2 Storyboard Preview
Preview and save generated storyboard images.

## Usage

1. **Load Models**: Use "Krea2 Storyboard Loader" node
2. **Upload Characters**: Use "Krea2 Character Sheet" node to upload 2 character images
3. **Write Prompts**: Use "Krea2 Storyboard Prompts" node to enter 15 scene descriptions
4. **Generate**: Connect to "Krea2 Storyboard Generator" and run
5. **Preview**: Use "Krea2 Storyboard Preview" to view results

## Example Prompt Format

```
Scene 1: Character A is walking through a magical forest, discovering a hidden portal
Scene 2: Character B appears from behind a tree, looking surprised
Scene 3: Both characters enter the portal together
...
```

## Style Keywords

Add these to your prompts for best results:
- `Pixar-quality 3D animated movie`
- `DreamWorks quality`
- `stylized animation`
- `expressive faces`
- `soft lighting`
- `global illumination`
- `vibrant colors`

## Resolution Options

- 1280×720 (16:9) - Default, good balance
- 1920×1080 (16:9) - Full HD, higher quality
- 1024×576 (16:9) - Faster generation

## Tips

1. **Character Consistency**: Use clear, well-lit character reference images
2. **Prompt Length**: Keep prompts detailed but concise (50-100 words)
3. **Seed**: Use same seed for consistent style across scenes
4. **Steps**: 35 steps is good balance between quality and speed

## License

MIT License
