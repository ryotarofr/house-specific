import house_specific
from PIL import Image

image_path = "/path/to/img.webp"
img = Image.open(image_path).convert("L")

# 画像データをバイト配列に変換
img_data = list(img.getdata())
width, height = img.size

barcode_regions = house_specific.detect_barcode_regions(img_data, width, height)

for region in barcode_regions:
    print(f"Barcode region - x_start: {region.x_start}, x_end: {region.x_end}, y_start: {region.y_start}, y_end: {region.y_end}")
