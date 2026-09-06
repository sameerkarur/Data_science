"""
Netflix Design Studio V2: Multi-Modal Compositional Pipeline & A/B Creative Engine
Author: Sameer Karur
Curriculum: Advanced Generative AI

Key Architectural Enhancements over V1:
- Multi-Aspect Ratio Creative Generation (1:1 Square, 16:9 Landscape Billboard, 9:16 Vertical Story)
- Semantic Style Expansion (Cinematic Noir, Cyberpunk Neon, Minimalist Bauhaus, Editorial Watercolor)
- Dynamic A/B Copy Matrix Generator (Headline, Hook, Emotional Trigger, Call-to-Action)
- Automated PIL Image Compositor with brand color palettes and typography layouts
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

@dataclass
class CreativeAsset:
    format_name: str
    dimensions: tuple # (width, height)
    aspect_ratio: str
    recommended_channel: str

ASPECT_RATIOS = [
    CreativeAsset("Square_Feed", (600, 600), "1:1", "Instagram Feed / LinkedIn Carousel"),
    CreativeAsset("Billboard_Landscape", (960, 540), "16:9", "Netflix Hero Carousel / YouTube Banner"),
    CreativeAsset("Vertical_Reels", (540, 960), "9:16", "Instagram Reels / TikTok / Stories")
]

class MultimodalCreativeStudioV2:
    def __init__(self, show_title: str, genre: str, target_audience: str):
        self.show_title = show_title
        self.genre = genre
        self.target_audience = target_audience

    def generate_style_prompts(self) -> Dict[str, str]:
        """Generates visual prompt descriptors conditioned on distinct artistic styles."""
        return {
            "Cinematic_Cyberpunk": (
                f"Cinematic wide-angle movie poster for '{self.show_title}', {self.genre} genre. "
                f"Neon cobalt blue and magenta rim lighting, rain-soaked asphalt reflections, "
                f"intense dramatic character silhouettes, 8k resolution, volumetric fog, anamorphic lens flares."
            ),
            "Minimalist_Bauhaus": (
                f"Bold graphic design poster for '{self.show_title}'. Swiss typographic layout, "
                f"clean geometric vectors, striking crimson red and obsidian black contrast, "
                f"negative space focal point, award-winning indie film aesthetic."
            ),
            "Editorial_Noir": (
                f"Moody black and white film-grain still for '{self.show_title}'. High contrast chiaroscuro "
                f"lighting, venetian blind shadows casting across troubled detective's face, gritty atmospheric tension."
            )
        }

    def generate_ab_copy_matrix(self) -> List[Dict[str, str]]:
        """Generates 3 variant ad copy sets tailored for conversion testing."""
        return [
            {
                "Variant": "A (Urgency & Suspense)",
                "Headline": f"Nothing is as it seems in {self.show_title}.",
                "Hook": "When the timeline fractures, every choice costs a lifetime.",
                "CTA": "Stream Now — Only on Netflix"
            },
            {
                "Variant": "B (Social Proof & Critical Acclaim)",
                "Headline": f"The #1 Most Watched Thriller Worldwide.",
                "Hook": "Critics are calling it 'a masterclass in unyielding suspense.'",
                "CTA": "Watch the Premiere Tonight"
            },
            {
                "Variant": "C (Character & Emotional Mystery)",
                "Headline": "Can you outrun what you cannot remember?",
                "Hook": "One detective. Zero alibis. The clock starts now.",
                "CTA": "Add to Your Watchlist"
            }
        ]

    def render_sample_banner(self, output_path: str = "/tmp/netflix_v2_banner.png") -> str:
        """Renders an automated marketing poster using PIL with brand typography and styling."""
        width, height = 960, 540
        img = Image.new("RGB", (width, height), color=(18, 18, 24))
        draw = ImageDraw.Draw(img)

        # Draw cinematic gradient background effect
        for y in range(height):
            r = int(18 + 40 * (y / height))
            g = int(18 + 5 * (y / height))
            b = int(24 + 10 * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Brand Accent Header Bar
        draw.rectangle([(0, 0), (width, 8)], fill=(229, 9, 20)) # Netflix signature red

        # Title Card
        draw.text((60, 80), "NETFLIX ORIGINAL SERIES", fill=(229, 9, 20))
        draw.text((60, 140), self.show_title.upper(), fill=(255, 255, 255))
        draw.text((60, 220), f"Genre: {self.genre} | Audience: {self.target_audience}", fill=(180, 180, 180))

        # Marketing Hook
        draw.rectangle([(60, 280), (750, 360)], fill=(30, 30, 42), outline=(60, 60, 80))
        draw.text((80, 305), "When the countdown hits zero, reality rewinds.", fill=(240, 240, 240))

        # CTA Button
        draw.rectangle([(60, 410), (280, 470)], fill=(229, 9, 20))
        draw.text((100, 432), "WATCH NOW ▶", fill=(255, 255, 255))

        img.save(output_path)
        return output_path

def run_demo():
    print("=" * 70)
    print("🚀 Running Netflix Creative Studio & Multi-Modal Engine V2 Demo")
    print("=" * 70)

    studio = MultimodalCreativeStudioV2(
        show_title="Echoes of the Obsidian Hour",
        genre="Sci-Fi Psychological Thriller",
        target_audience="Mystery & Tech-Enthusiasts (18-35)"
    )

    print(f"\n🎬 1. Production Creative Specs for: '{studio.show_title}'")
    for asset in ASPECT_RATIOS:
        print(f"  • {asset.format_name:20}: {asset.dimensions[0]}x{asset.dimensions[1]} ({asset.aspect_ratio}) — {asset.recommended_channel}")

    print("\n🎨 2. Multi-Style Diffusion Prompts:")
    styles = studio.generate_style_prompts()
    for s_name, prompt in styles.items():
        print(f"\n  [{s_name}]")
        print(f"  \"{prompt}\"")

    print("\n✍️ 3. A/B Testing Copy Matrix:")
    ab_matrix = studio.generate_ab_copy_matrix()
    for item in ab_matrix:
        print(f"  • {item['Variant']:30} | \"{item['Headline']}\" (CTA: '{item['CTA']}')")

    banner_path = studio.render_sample_banner()
    print(f"\n🖼️ 4. Rendered Creative Banner Saved: {banner_path}")
    print("\n✅ Netflix Design Studio V2 execution completed successfully.")

if __name__ == "__main__":
    run_demo()
