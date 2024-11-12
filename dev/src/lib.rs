use image::{ImageBuffer, Luma};
use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::{num_complex::Complex, FftPlanner};

use image::{ImageBuffer, Luma};
use rustfft::{num_complex::Complex, FftPlanner};

/// 作成:
///   @Fujii Ryotaro
///   2024/11/12
///
///
/// 補足:
/// Represents a region in the image that is identified as a barcode.
struct BarcodeRegion {
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
}

// 定数の定義
const VERTICAL_SECTIONS: u32 = 60; // 縦画像のセクション数
const HORIZONTAL_SECTIONS: u32 = 100; // 横画像のセクション数
const SECTION_HEIGHT: u32 = 50; // セクションの高さ
const THRESHOLD: f32 = 50.0; // 振幅のしきい値
const CONSECUTIVE_THRESHOLD: usize = 5; // バーコード領域とみなす連続セクション数

fn detect_barcode_regions<T>(img: &ImageBuffer<Luma<u8>, Vec<T>>) {
    /// 画像全体の縦横比から、縦画像なのか横画像なのかを判定する
    /// is_ratio: true -> 縦画像, false -> 横画像
    let (width, height) = img.dimensions();
    let is_ratio = width <= height;

    /// 画像比の判定より、横方向のセクション数を決定
    /// 縦画像: 60, 横画像: 100 (TODO: ここは動的値を設定できるように修正)
    ///
    /// NOTE:
    ///　フーリエ変換時 y 方向で画像を圧縮するため、
    ///  section_height は数値が小さいほど認識精度が向上するが、処理時間が増加する
    ///
    let sections_per_width = if is_ratio {
        VERTICAL_SECTIONS
    } else {
        HORIZONTAL_SECTIONS
    };
    let section_width = width / sections_per_width;

    let sections_per_height = (height / SECTION_HEIGHT) as usize;

    for section_index_y in 0..sections_per_height {
        for section_index_y in 0..sections_per_height {
            for section_index_y in 0..num_sections_height {
                let section_y_start = section_index_y as u32 * SECTION_HEIGHT;
                let mut section_magnitudes = Vec::new();

                for section_index_x in 0..sections_per_width {
                    let section_x_start = section_index_x as u32 * section_width;

                    let section_line: Vec<u8> = (0..section_width)
                        .map(|x| {
                            img.get_pixel(section_x_start + x, section_y_start + SECTION_HEIGHT / 2)
                                [0]
                        })
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

                    if section_magnitude > THRESHOLD {
                        section_magnitudes.push(section_magnitude);
                    } else {
                        section_magnitudes.push(0.0);
                    }
                }

                /// 高周波成分の振幅が閾値を超えるセクションをバーコード領域として判定
                /// 閾値:
                ///  - 連続セクション数: CONSECUTIVE_THRESHOLD
                let mut consecutive_count = 0;
                let mut start_index = None;

                for (section_index, &magnitude) in section_magnitudes.iter().enumerate() {
                    if magnitude > 0.0 {
                        if consecutive_count == 0 {
                            start_index = Some(section_index); // 新しいバーコード領域の開始位置
                        }
                        consecutive_count += 1;

                        if consecutive_count >= CONSECUTIVE_THRESHOLD {
                            if let Some(start) = start_index {
                                let end = section_index;
                                let x_start = start as u32 * section_width;
                                let x_end = (end + 1) as u32 * section_width;

                                barcode_regions.push(BarcodeRegion {
                                    x_start,
                                    x_end,
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

            barcode_regions
        }
    }
}

#[test]
fn plot_section_magnitudes(
    magnitudes: &[f32],
    barcode_regions: &[(u32, u32, u32, u32)],
    section_num: usize,
    section_width: u32,
    section_height: u32,
) {
    use plotters::prelude::*;

    let filename = format!("assets/section_magnitudes_{}_height.png", section_num);
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_magnitude = magnitudes.iter().cloned().fold(f32::NAN, f32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Height Section {} - 周波数成分の強度", section_num),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..magnitudes.len(), 0f32..max_magnitude)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(magnitudes.iter().enumerate().map(|(i, &mag)| {
            let color = if barcode_regions
                .iter()
                .any(|&(x_start, x_end, y_start, y_end)| {
                    let x_pos = i as u32 * section_width;
                    let y_pos = section_num as u32 * section_height;

                    x_pos >= x_start && x_pos < x_end && y_pos >= y_start && y_pos < y_end
                }) {
                &RED
            } else {
                &BLUE
            };
            Rectangle::new([(i, 0.0), (i + 1, mag)], color.filled())
        }))
        .unwrap();
}
