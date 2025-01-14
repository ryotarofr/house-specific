# house-specific

Package for locating the coordinates of a barcode in an image

## Why create it?

In using Zbar, we sometimes have trouble recognizing images that contain a lot of noise (low performance scanners or images that have been scanned multiple times).
We have found that when we import an image of a barcode that has been enlarged, the recognition rate increases slightly.
We have also created a hypothesis for a process to improve recognition accuracy by compiling the results of testing existing barcode readers.

1. improve image preprocessing
2. decode barcodes by enlarging them
3. create our own decoding logic

This library is a prototype to realize 2. We needed to show the procedure for obtaining the exact coordinate position of a barcode from the entire image.


## Set Up

### Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Continue? (y/N) y
# ...
# 1) Proceed with standard installation (default - just press enter)
# 2) Customize installation
# 3) Cancel installation
# >1
```

```bash
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc
```

```bash
sudo apt install cargo
cargo --version
# cargo 1.75.0
```

```bash
sudo apt install rustup
rustup --version
# rustup 1.26.0 (2024-04-01)
```


```bash
sudo apt install cargo
cargo --version
# cargo 1.82.0 (8f40fc59f 2024-08-21)
```

```bash
sudo apt install rustup
rustup --version
# rustup 1.27.1 (54dd3d00f 2024-04-24)
```


### Python

```bash
sudo apt install python3-pip
pip3 --version
# pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)
```

```bash
sudo apt install -y libfontconfig1-dev
```

#### Virtual environment (required at build time)

```bash
sudo apt install python3.12-venv
```

Executed only for the first time

```bash
python3 -m venv .venv  # .venv is the name of the virtual environment (optional)
```

```bash
source .venv/bin/activate
```

At the end of the virtual environment

```bash
deactivate
```

#### Package creation (performed in a virtual environment)

```bash
pip install maturin
```

(Execute the following command for the first time only)

```bash
maturin init
```

```bash
? 🤷 Which kind of bindings to use?
  📖 Documentation: https://maturin.rs/bindings.html ›
❯ pyo3 <<< Select this
  cffi
  uniffi
  bin
```

build command

```bash
maturin develop
```

## generate `.whl`(Recommended for local development)

Wheel needs to be generated to make the format executable only in python environment when deploying Docker image.

```bash
maturin build --release --strip --manylinux off
```

## Overview

Detects features on horizontal lines by adapting a one-dimensional Discrete Fourier Transform (DFT) to the image (x-direction).

### Application of Discrete Fourier Transform

Convert the binarized line to a vector of complex numbers (Complex<f32>) and set the real part to binary data.
Use `FftPlanner` in `rustfft` to create a forward FFT plan with appropriate size.

### Frequency Component Analysis

Calculate the amplitude of each frequency component (excluding index 0, which is the DC component). This is obtained by computing the square root of the sum of the squares of the real and imaginary parts.
Examining the sum of the amplitudes represents the overall frequency content of the section.
Any section whose sum of amplitudes exceeds a certain threshold is considered to contain significant frequency components (barcode patterns).

### verification

Measure the processing time with the following sample code. Roughly less than 1 second.

```py
import time
import house_specific
from PIL import Image

# Start measuring processing time
start_time = time.time()

image_path = "./sample.webp"
img = Image.open(image_path).convert("L")

# Convert image data to byte array (S3 standard format)
img_data = list(img.getdata())
width, height = img.size

# Call the main function
barcode_regions = house_specific.detect_barcode_regions(img_data, width, height)

# End of processing time measurement
end_time = time.time()

# Show Results
for region in barcode_regions:
    print(f"Barcode region - x_start: {region.x_start}, x_end: {region.x_end}, y_start: {region.y_start}, y_end: {region.y_end}")

# Output processing time
print(f"processing time: {end_time - start_time:.4f} s")

```

```bash
Barcode region - x_start: 1161, x_end: 1458, y_start: 100, y_end: 150
Barcode region - x_start: 1134, x_end: 1458, y_start: 1850, y_end: 1900
processing time: 0.5448 s
```

The coordinates of two bar codes (setting decision and identifier) are acquired.

