import cv2
import numpy as np
import shutil
from pathlib import Path

def practice_image_operation(img):

    results={}

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    #Rotate
    M = cv2.getRotationMatrix2D(center, 30, 1)
    results["rotated_30"] = cv2.warpAffine(img, M, (w, h))

    #Crop
    results["cropped"] = img[50:250, 50:250]

    #Flip
    results["flip_horizontal"] = cv2.flip(img, 1)
    results["flip_vertical"] = cv2.flip(img, 0)

    #B&W
    results["gray"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #RGB
    red = img.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0
    results["red_only"] = red

    green = img.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    results["green_only"] = green

    blue = img.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    results["blue_only"] = blue

    #Blur
    results["gaussian_blur"] = cv2.GaussianBlur(img, (15, 15), 0)
    results["median_blur"] = cv2.medianBlur(img, 15)

    #Zoom
    results["scaled_up"] = cv2.resize(img, None, fx=1.5, fy=1.5)
    results["scaled_down"] = cv2.resize(img, None, fx=0.5, fy=0.5)

    return results


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


def list_images(folder):
    #All image files in a folder (not subfolders), sorted, extension check ignores case
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def process_image_folder(input_dir, output_dir, num_images=None):
    #Runs every operation on each image and saves into output_dir/<operation>/<image name>
    #num_images=None processes all of them
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Output folder must be different from input folder")

    images = list_images(input_dir)
    if num_images is not None:
        images = images[:num_images]
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    processed = 0
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # forces 3 channels so the RGB ops work on gray/png images too
        if img is None:
            print(f"Skipping unreadable file: {path.name}")
            continue

        for op_name, result in practice_image_operation(img).items():
            op_dir = output_dir / op_name
            op_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(op_dir / path.name), result)

        processed += 1
        print(f"[{processed}/{len(images)}] {path.name}")

    print(f"Done. {processed} images saved to {output_dir}")
    return processed


def create_practice_folder(source_dir, practice_dir, num_images=None):
    #Copies images into a separate practice folder so the real dataset is never touched
    source_dir = Path(source_dir)
    practice_dir = Path(practice_dir)

    if practice_dir.resolve() == source_dir.resolve():
        raise ValueError("Practice folder must be different from the dataset folder")

    images = list_images(source_dir)
    if num_images is not None:
        images = images[:num_images]

    practice_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        shutil.copy2(path, practice_dir / path.name)

    print(f"Copied {len(images)} images to {practice_dir}")
    return practice_dir

if __name__ == "__main__":
    base = Path(__file__).parent

    #1. Copy a few images out of the real dataset
    practice_input = create_practice_folder(base / "train" / "images", base / "practice" / "input", num_images=5)

    #2. Run all operations on the copies only
    process_image_folder(practice_input, base / "practice" / "output", num_images=5)


