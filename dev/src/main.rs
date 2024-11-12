use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::fs;
use std::path::Path;

fn main() {
    // 画像ファイルをバイトデータとして読み込む
    let img_bytes = fs::read("assets/yoko2.webp").expect("Failed to read image file");

    // バイトデータから画像を読み込む
    let img = image::load_from_memory(&img_bytes)
        .expect("Failed to load image from bytes")
        .to_luma8();

    let num_sections_width = 100; // 横方向のセクション数
    let section_width = img.width() / num_sections_width as u32;
    let section_height = 50; // 縦方向の分割高さ
    let num_sections_height = (img.height() / section_height) as usize;

    let threshold = 50.0; // 振幅の合計値のしきい値
    let consecutive_threshold = 5; // 横方向でバーコード領域とみなす連続セクション数

    let mut barcode_regions = Vec::new();

    // 保存先ディレクトリの作成
    let output_dir = Path::new("assets/sections");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    for j in 0..num_sections_height {
        let section_y_start = j as u32 * section_height;

        // 縦方向に切り取った画像を取得
        let section_image = img.view(0, section_y_start, img.width(), section_height);
        let section_image = GrayImage::from(section_image.to_image());
        let output_path = output_dir.join(format!("section_{}.png", j));
        section_image
            .save(&output_path)
            .expect("Failed to save section image");
        println!("Saved section image: {}", output_path.display());

        let mut section_magnitudes = Vec::new();

        for i in 0..num_sections_width {
            let section_x_start = i as u32 * section_width;

            // 縦方向の分割区画（y方向）内で中心の横ラインを取得
            let section_line: Vec<u8> = (0..section_width)
                .map(|x| {
                    img.get_pixel(section_x_start + x, section_y_start + section_height / 2)[0]
                })
                .collect();

            let binary_line: Vec<f32> = section_line
                .iter()
                .map(|&pixel| if pixel > 128 { 1.0 } else { 0.0 })
                .collect();

            let mut input: Vec<Complex<f32>> =
                binary_line.iter().map(|&x| Complex::new(x, 0.0)).collect();
            let mut output = vec![Complex::new(0.0, 0.0); input.len()];

            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(input.len());
            fft.process(&mut input);
            output.copy_from_slice(&input);

            let section_magnitude: f32 = output
                .iter()
                .skip(1)
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .sum();

            if section_magnitude > threshold {
                section_magnitudes.push(section_magnitude);
            } else {
                section_magnitudes.push(0.0);
            }
        }

        let mut consecutive_count = 0;
        let mut start_index = None;

        for (i, &magnitude) in section_magnitudes.iter().enumerate() {
            if magnitude > 0.0 {
                if consecutive_count == 0 {
                    start_index = Some(i); // 新しいバーコード領域の開始位置
                }
                consecutive_count += 1;

                if consecutive_count >= consecutive_threshold {
                    if let Some(start) = start_index {
                        let end = i;
                        let x_start = start as u32 * section_width;
                        let x_end = (end + 1) as u32 * section_width;

                        barcode_regions.push((
                            x_start,
                            x_end,
                            section_y_start,
                            section_y_start + section_height,
                        ));
                        println!(
                            "バーコード領域: x_start = {}, x_end = {}, y_start = {}, y_end = {}",
                            x_start,
                            x_end,
                            section_y_start,
                            section_y_start + section_height
                        );
                    }
                }
            } else {
                consecutive_count = 0;
                start_index = None;
            }
        }

        plot_section_magnitudes(
            &section_magnitudes,
            &barcode_regions,
            j,
            section_width,
            section_height,
        );
    }
}

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
