use image::{ImageBuffer, Luma};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Represents a region in the image that is identified as a barcode.
#[pyclass]
#[derive(Debug, Clone)]
struct BarcodeRegion {
    #[pyo3(get)]
    x_start: u32,
    #[pyo3(get)]
    x_end: u32,
    #[pyo3(get)]
    y_start: u32,
    #[pyo3(get)]
    y_end: u32,
}

const VERTICAL_SECTIONS: u32 = 60;
const HORIZONTAL_SECTIONS: u32 = 100;
const SECTION_HEIGHT: u32 = 50;
const THRESHOLD: f32 = 50.0;
const CONSECUTIVE_THRESHOLD: usize = 5;

/// Detects barcode-like regions in a grayscale image using frequency analysis.
///
/// # Arguments
///
/// * `img` - A reference to the grayscale image buffer
///
/// # Returns
///
/// A vector of `BarcodeRegion` containing detected regions
///
/// # Example
///
/// ```
/// use barcode_detector::{detect_barcode_regions, BarcodeRegion};
/// use image::GrayImage;
///
/// let img = GrayImage::new(800, 600);
/// let regions = detect_barcode_regions(&img);
/// for region in regions {
///     println!("{:?}", region);
/// }
/// ```
#[pyfunction]
fn detect_barcode_regions(img_data: Vec<u8>, width: u32, height: u32) -> Vec<BarcodeRegion> {
    let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width, height, img_data)
        .expect("Failed to create image buffer");

    let is_ratio = width <= height;
    let sections_per_width = if is_ratio {
        VERTICAL_SECTIONS
    } else {
        HORIZONTAL_SECTIONS
    };
    let section_width = width / sections_per_width;
    let sections_per_height = (height / SECTION_HEIGHT) as usize;

    let mut barcode_regions = Vec::new();
    let mut planner = FftPlanner::<f32>::new();

    for section_index_y in 0..sections_per_height {
        let section_y_start = section_index_y as u32 * SECTION_HEIGHT;

        // Calculate the amplitude of each horizontal section
        let section_magnitudes = compute_section_magnitudes(
            &img,
            section_y_start,
            section_width,
            sections_per_width,
            &mut planner,
        );

        // Detects high amplitude areas as barcode areas
        detect_regions(
            &section_magnitudes,
            section_y_start,
            section_width,
            &mut barcode_regions,
        );
    }

    barcode_regions
}

/// Computes the magnitude of each section's frequency response along a specified horizontal line.
///
/// # Arguments
///
/// * `img` - A reference to the grayscale image buffer
/// * `section_y_start` - The y-coordinate to start from
/// * `section_width` - Width of each section
/// * `sections_per_width` - Number of sections across the width
/// * `planner` - FFT planner to use for frequency analysis
fn compute_section_magnitudes(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    section_y_start: u32,
    section_width: u32,
    sections_per_width: u32,
    planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
    let mut section_magnitudes = Vec::new();

    for section_index_x in 0..sections_per_width {
        let section_x_start = section_index_x * section_width;

        let section_line: Vec<u8> = (0..section_width)
            .map(|x| img.get_pixel(section_x_start + x, section_y_start + SECTION_HEIGHT / 2)[0])
            .collect();

        let binary_line: Vec<f32> = section_line
            .iter()
            .map(|&pixel| if pixel > 128 { 1.0 } else { 0.0 })
            .collect();

        let mut input: Vec<Complex<f32>> =
            binary_line.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); input.len()];

        let fft = planner.plan_fft_forward(input.len());
        fft.process(&mut input);
        output.copy_from_slice(&input);

        let section_magnitude: f32 = output
            .iter()
            .skip(1)
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .sum();

        section_magnitudes.push(if section_magnitude > THRESHOLD {
            section_magnitude
        } else {
            0.0
        });
    }

    section_magnitudes
}

/// Detects contiguous regions of high frequency magnitude that likely indicate barcodes.
///
/// # Arguments
///
/// * `section_magnitudes` - Vector of magnitudes for each section
/// * `section_y_start` - Starting y-coordinate of the section
/// * `section_width` - Width of each section
/// * `barcode_regions` - Vector to store detected regions
fn detect_regions(
    section_magnitudes: &[f32],
    section_y_start: u32,
    section_width: u32,
    barcode_regions: &mut Vec<BarcodeRegion>,
) {
    let mut consecutive_count = 0;
    let mut start_index = None;

    for (section_index, &magnitude) in section_magnitudes.iter().enumerate() {
        if magnitude > 0.0 {
            if consecutive_count == 0 {
                start_index = Some(section_index);
            }
            consecutive_count += 1;

            if consecutive_count >= CONSECUTIVE_THRESHOLD {
                if let Some(start) = start_index {
                    let end = section_index;
                    barcode_regions.push(BarcodeRegion {
                        x_start: start as u32 * section_width,
                        x_end: (end + 1) as u32 * section_width,
                        y_start: section_y_start,
                        y_end: section_y_start + SECTION_HEIGHT,
                    });
                }
            }
        } else {
            consecutive_count = 0;
            start_index = None;
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn house_specific(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_barcode_regions, m)?)?;
    Ok(())
}
