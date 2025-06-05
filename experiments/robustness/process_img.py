import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

def create_output_path(input_path, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    return output_path



def scale_image(input_path: str, output_dir: str, width: int = None, height: int = None, factor: float = None, interpolation = Image.Resampling.BILINEAR) -> str:
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path}")

    with Image.open(input_path) as img:
        original_width, original_height = img.size
        new_width, new_height = original_width, original_height # Default to original size

        if factor is not None:
            new_width = int(original_width * factor)
            new_height = int(original_height * factor)
            op_suffix = f"scaled_factor_{factor:.2f}"
        elif width is not None and height is not None:
                new_width = width
                new_height = height
                op_suffix = f"scaled_{width}x{height}"
        elif width is not None:
            new_width = width
            new_height = int(original_height * (width / original_width))
            op_suffix = f"scaled_w{width}"
        elif height is not None:
            new_height = height
            new_width = int(original_width * (height / original_height))
            op_suffix = f"scaled_h{height}"

        new_width = max(1, new_width)
        new_height = max(1, new_height)

        scaled_img = img.resize((new_width, new_height), resample=interpolation)
        output_path = create_output_path(input_path, output_dir)
        scaled_img.save(output_path)
        return output_path

def rotate_image(input_path: str, output_dir: str, angle: float, expand: bool = False, fillcolor = None) -> str:
    

    with Image.open(input_path) as img:
        if img.mode == 'RGBA' and fillcolor is not None:
            if isinstance(fillcolor, tuple) and len(fillcolor) == 3:
                fillcolor = (*fillcolor, 0) 
            elif isinstance(fillcolor, str):
                
                from PIL import ImageColor
                rgb = ImageColor.getrgb(fillcolor)
                fillcolor = (*rgb, 0)


        rotated_img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=fillcolor)
        output_path = create_output_path(input_path, output_dir)
        rotated_img.save(output_path)
        return output_path

def convert_to_grayscale(input_path: str, output_dir: str) -> str:
   

    with Image.open(input_path) as img:
        
        grayscale_img = img.convert('L')
        output_path = create_output_path(input_path, output_dir)
        
        grayscale_img.save(output_path)
        return output_path
def apply_blur(input_path: str, output_dir: str, radius: float = 2.0) -> str:
    

    with Image.open(input_path) as img:
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        output_path = create_output_path(input_path, output_dir)
        blurred_img.save(output_path)
        return output_path

if __name__ == "__main__":
    directory_path="/home/cty/WatermarkmmRAG/datasets/watermark_images/optimization/llava"
    output_directory="/home/cty/WatermarkmmRAG/datasets/watermark_images_attackd/optimization_all"
    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)
        scale_path = scale_image(img_path, output_directory, factor=1.5, interpolation=Image.Resampling.BICUBIC)
        rotate_path = rotate_image(scale_path, output_directory, angle=45, expand=True, fillcolor='white')
        #grayscale_path = convert_to_grayscale(rotate_path, output_directory)
        blur_path = apply_blur(rotate_path, output_directory, radius=2.0)
    