from PIL import Image

def change_black_to_white(image_path, output_path):
    # Open the image
    with Image.open(image_path) as img:
        # Convert the image to 'LA' mode (Luminance and Alpha)
        img = img.convert("LA")

        # Load the pixel data
        pixels = img.load()

        # Iterate over each pixel
        for x in range(img.width):
            for y in range(img.height):
                # Get the grayscale and alpha values
                grayscale, alpha = pixels[x, y]

                # Change black to white while preserving alpha
                if grayscale == 0:
                    pixels[x, y] = (255, alpha)

        # Save the modified image
        img.save(output_path)

# Example usage
change_black_to_white('media/target.png', 'media/target2.png')
