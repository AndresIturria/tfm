from pathlib import Path
from PIL import Image

def resize_and_crop_images():
    # Define source and destination folders
    data_folder = Path("data")
    cropped_folder = Path("cropped")
    cropped_folder.mkdir(exist_ok=True)

    target_size = (224, 224)

    # Loop through each species folder
    for species_folder in data_folder.iterdir():
        if species_folder.is_dir():
            # Create corresponding folder in 'cropped'
            target_species_folder = cropped_folder / species_folder.name
            target_species_folder.mkdir(parents=True, exist_ok=True)

            # Process PNG and JPG images
            for img_file in species_folder.glob("*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        with Image.open(img_file) as img:
                            original_size = img.size

                            # Convert to RGB to avoid mode issues
                            img = img.convert("RGB")

                            # Center crop to square if needed
                            min_side = min(img.size)
                            left = (img.width - min_side) // 2
                            top = (img.height - min_side) // 2
                            right = left + min_side
                            bottom = top + min_side
                            img_cropped = img.crop((left, top, right, bottom))

                            # Resize to 224x224
                            img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)

                            # Generate filename with .png extension
                            target_file_name = img_file.stem + ".png"
                            target_file_path = target_species_folder / target_file_name

                            # Save as PNG
                            img_resized.save(target_file_path, format="PNG")

                            print(f"Processed {img_file} | Original: {original_size} -> Saved: {target_file_path}")

                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    resize_and_crop_images()
