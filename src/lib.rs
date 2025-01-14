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
const SECTION_HEIGHT: u32 = 5;
const THRESHOLD: f32 = 50.0;
const CONSECUTIVE_THRESHOLD: usize = 5;
const MAX_WHITE_BLACK_WIDTH: usize = 10;

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

    // merge same pos "y"
    merge_barcode_regions(&mut barcode_regions);

    // merge current pos "y" and next pos "y"
    merge_regions_if_y_matches(&mut barcode_regions);

    barcode_regions
}

/// Detects character-like regions in a grayscale image by leveraging barcode detection logic.
///
/// # Arguments
///
/// * `img_data` - A vector of `u8` representing the grayscale image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Returns
///
/// A vector of `BarcodeRegion` representing detected character regions.
///
/// # Example
///
/// ```
/// use barcode_detector::{detect_character_regions, BarcodeRegion};
///
/// let img_data = vec![0; 800 * 600]; // Example grayscale image data
/// let width = 800;
/// let height = 600;
///
/// let regions = detect_character_regions(img_data, width, height);
/// for region in regions {
///     println!("{:?}", region);
/// }
/// ```
#[pyfunction]
fn detect_character_regions(img_data: Vec<u8>, width: u32, height: u32) -> Vec<BarcodeRegion> {
    // Detect barcode-like regions using the barcode detection logic
    let mut barcode_regions = detect_barcode_regions(img_data, width, height);

    // Adjust the detected regions for better alignment and scaling
    adjust_regions(&mut barcode_regions, width, height);

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

        // Check the width of the black and white area
        if contains_large_white_black_regions(&binary_line, MAX_WHITE_BLACK_WIDTH) {
            section_magnitudes.push(0.0);
            continue;
        }

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

/// Checks if a binary line contains any white or black region
/// with a width greater than the specified maximum width.
///
/// # Arguments
///
/// * `binary_line` - A slice of `f32` values representing a binary line,
///   where 1.0 indicates a "white" pixel and 0.0 indicates a "black" pixel.
/// * `max_width` - The maximum allowable width for a continuous white or black region.
///
/// # Returns
///
/// Returns `true` if any region of white or black exceeds the specified maximum width,
/// otherwise returns `false`.
///
/// # Example
///
/// ```rust
/// let binary_line = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let max_width = 2;
/// let result = contains_large_white_black_regions(&binary_line, max_width);
/// assert_eq!(result, true); // The black region exceeds the maximum width of 2.
/// ```
///
/// # Notes
///
/// This function is useful for filtering binary lines where large
/// continuous regions of the same color (white or black) are not desired.
///
fn contains_large_white_black_regions(binary_line: &[f32], max_width: usize) -> bool {
    let mut count = 0;
    let mut current_value = binary_line[0];

    for &value in binary_line {
        if value == current_value {
            count += 1;
        } else {
            if count > max_width {
                return true;
            }
            current_value = value;
            count = 1;
        }
    }

    count > max_width
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

/// Merges overlapping or adjacent barcode regions with the same vertical range.
///
/// This function takes a mutable vector of `BarcodeRegion` objects, groups regions
/// with identical `y_start` and `y_end` values, and merges their horizontal ranges.
/// The merged regions replace the original list.
///
/// # Arguments
///
/// * `barcode_regions` - A mutable reference to a vector of `BarcodeRegion` objects
///   that will be merged if their vertical ranges (`y_start` and `y_end`) match.
///
/// # Example
///
/// ```rust
/// let mut regions = vec![
///     BarcodeRegion { x_start: 10, x_end: 20, y_start: 50, y_end: 60 },
///     BarcodeRegion { x_start: 21, x_end: 30, y_start: 50, y_end: 60 },
///     BarcodeRegion { x_start: 5, x_end: 15, y_start: 70, y_end: 80 },
///     BarcodeRegion { x_start: 16, x_end: 25, y_start: 70, y_end: 80 },
/// ];
///
/// merge_barcode_regions(&mut regions);
///
/// assert_eq!(regions, vec![
///     BarcodeRegion { x_start: 10, x_end: 30, y_start: 50, y_end: 60 },
///     BarcodeRegion { x_start: 5, x_end: 25, y_start: 70, y_end: 80 },
/// ]);
/// ```
fn merge_barcode_regions(barcode_regions: &mut Vec<BarcodeRegion>) {
    // Sort regions by their vertical range (y_start, y_end)
    barcode_regions.sort_by(|a, b| (a.y_start, a.y_end).cmp(&(b.y_start, b.y_end)));

    let mut merged_regions = Vec::new();
    let mut current_group = Vec::new();

    for region in barcode_regions.drain(..) {
        if current_group.is_empty() {
            current_group.push(region);
        } else {
            let first_region = &current_group[0];
            if region.y_start == first_region.y_start && region.y_end == first_region.y_end {
                current_group.push(region);
            } else {
                // Merge the current group and start a new one
                merged_regions.push(merge_group(&current_group));
                current_group.clear();
                current_group.push(region);
            }
        }
    }

    // Merge the final group
    if !current_group.is_empty() {
        merged_regions.push(merge_group(&current_group));
    }

    // Replace the original vector with the merged results
    *barcode_regions = merged_regions;
}

/// Merges regions in a vector of `BarcodeRegion` if their `y_end` and `y_start` are consecutive.
/// This function modifies the original vector by replacing it with the merged regions.
///
/// # Arguments
///
/// * `regions` - A mutable reference to a vector of `BarcodeRegion` to be processed.
///
/// # Details
///
/// The function sorts the regions based on their `y_start` and `y_end`, ensuring that
/// regions with consecutive vertical positions (i.e., `y_end` of one region equals `y_start` of the next)
/// are merged into a single region. The horizontal range (`x_start` and `x_end`) is adjusted to cover
/// the full range of merged regions.
///
/// # Example
///
/// ```rust
/// let mut regions = vec![
///     BarcodeRegion { x_start: 10, x_end: 20, y_start: 0, y_end: 5 },
///     BarcodeRegion { x_start: 15, x_end: 25, y_start: 5, y_end: 10 },
///     BarcodeRegion { x_start: 30, x_end: 40, y_start: 20, y_end: 25 },
/// ];
///
/// merge_regions_if_y_matches(&mut regions);
///
/// assert_eq!(regions, vec![
///     BarcodeRegion { x_start: 10, x_end: 25, y_start: 0, y_end: 10 },
///     BarcodeRegion { x_start: 30, x_end: 40, y_start: 20, y_end: 25 },
/// ]);
/// ```
fn merge_regions_if_y_matches(regions: &mut Vec<BarcodeRegion>) {
    // Sort regions by their vertical position (`y_start`, then `y_end`) for consistent merging.
    regions.sort_by(|a, b| {
        a.y_start
            .cmp(&b.y_start)
            .then_with(|| a.y_end.cmp(&b.y_end))
    });

    let mut merged_regions = Vec::new();
    let mut current_group = Vec::new();

    // Iterate through all regions and group them based on vertical continuity.
    for region in regions.drain(..) {
        if current_group.is_empty() {
            // Start a new group with the current region.
            current_group.push(region);
        } else {
            let last_region = current_group.last().unwrap();
            if last_region.y_end == region.y_start {
                // If the current region's `y_start` matches the last region's `y_end`,
                // add it to the current group for merging.
                current_group.push(region);
            } else {
                // If the regions are not vertically continuous, merge the current group
                // and start a new group with the current region.
                merged_regions.push(merge_group(&current_group));
                current_group.clear();
                current_group.push(region);
            }
        }
    }

    // Merge the final group if there are any remaining regions.
    if !current_group.is_empty() {
        merged_regions.push(merge_group(&current_group));
    }

    // Replace the original regions with the merged results.
    *regions = merged_regions;
}

/// Merges a group of `BarcodeRegion` objects into a single region.
///
/// The function calculates the smallest `x_start` and the largest `x_end`
/// within the group. It assumes all regions in the group have the same
/// `y_start` and `y_end`.
///
/// # Arguments
///
/// * `group` - A slice of `BarcodeRegion` objects to be merged. All regions
///   must have the same `y_start` and `y_end`.
///
/// # Returns
///
/// A new `BarcodeRegion` that spans the entire horizontal range of the group.
///
/// # Panics
///
/// This function will panic if the input slice is empty.
///
/// # Example
///
/// ```rust
/// let group = vec![
///     BarcodeRegion { x_start: 10, x_end: 20, y_start: 50, y_end: 60 },
///     BarcodeRegion { x_start: 15, x_end: 25, y_start: 50, y_end: 60 },
/// ];
///
/// let merged = merge_group(&group);
///
/// assert_eq!(merged, BarcodeRegion { x_start: 10, x_end: 25, y_start: 50, y_end: 60 });
/// ```
fn merge_group(group: &[BarcodeRegion]) -> BarcodeRegion {
    if group.is_empty() {
        panic!("merge_group: Group is empty and cannot be merged.");
    }

    let x_start = group.iter().map(|r| r.x_start).min().unwrap();
    let x_end = group.iter().map(|r| r.x_end).max().unwrap();
    let y_start = group.first().unwrap().y_start;
    let y_end = group.last().unwrap().y_end;

    BarcodeRegion {
        x_start,
        x_end,
        y_start,
        y_end,
    }
}
/// Adjusts the dimensions of barcode regions by expanding or shrinking their coordinates.
///
/// This function modifies each region's coordinates to expand its size while ensuring
/// the new coordinates do not exceed the image boundaries. Specifically:
/// - `x_start` and `y_start` are reduced by 50 pixels if they are greater than or equal to 50.
/// - `x_end` and `y_end` are increased by 50 pixels but are capped at the image's width and height, respectively.
///
/// # Arguments
///
/// * `barcode_regions` - A mutable reference to a vector of `BarcodeRegion` objects to adjust.
/// * `width` - The width of the image. Used to cap `x_end`.
/// * `height` - The height of the image. Used to cap `y_end`.
///
/// # Example
///
/// ```rust
/// let mut regions = vec![
///     BarcodeRegion { x_start: 100, x_end: 200, y_start: 100, y_end: 150 }
/// ];
///
/// adjust_regions(&mut regions, 300, 200);
///
/// assert_eq!(regions, vec![
///     BarcodeRegion { x_start: 125, x_end: 175, y_start: 154, y_end: 200 }
/// ]);
/// ```
fn adjust_regions(barcode_regions: &mut [BarcodeRegion], _width: u32, height: u32) {
    // TODO: Optimize the process of removing * from both ends of the barcode
    for region in barcode_regions.iter_mut() {
        region.x_start += 25;
        region.x_end -= 25;
        region.y_start = region.y_end + 4;
        region.y_end = (region.y_end + 50).min(height);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn house_specific(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_character_regions, m)?)?;
    Ok(())
}
