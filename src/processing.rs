use ndarray::{Array2, Axis, Slice};
use num_traits::Bounded;
use pyo3::exceptions::{PyFileExistsError, PyRuntimeError, PyValueError};
use pyo3::PyResult;
use std::path::Path;

/// the main function
pub fn get_px_deltas_from_lines(
    image_path: String,
    mask_path: Option<String>,
) -> PyResult<Vec<usize>> {
    let image_path = Path::new(&image_path);
    validate_image_path(image_path)?;
    let image = load_image(image_path)?;

    let mask = match mask_path {
        Some(pth) => {
            let mask_path = Path::new(&pth);
            validate_image_path(mask_path)?;
            load_image(mask_path)?
        }
        // no mask
        None => image.clone().mapv(|_| 0),
    };

    let result = all_pixel_deltas(image, mask)?;

    Ok(result)
}

fn validate_image_path(img_path: &Path) -> PyResult<()> {
    let img_path_str = img_path.display().to_string();
    if !img_path_str.ends_with(".png") {
        return Err(PyValueError::new_err(format!(
            "Non-PNG input: {img_path_str}"
        )));
    }
    if !img_path.exists() {
        return Err(PyFileExistsError::new_err(format!(
            "Image path does not exist: {img_path_str}"
        )));
    }
    Ok(())
}

fn load_image(img_path: &Path) -> PyResult<Array2<u8>> {
    let img_base = image::open(img_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .to_luma8();
    let img_vec = img_base.as_raw();
    let img_width = img_base.width() as usize;
    let img_height = img_base.height() as usize;

    Array2::<u8>::from_shape_vec((img_height, img_width), img_vec.to_owned())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Find indices of an intensity map where the value is maximum, i.e. the pixel is white.
fn white_pixel_indices<T: Bounded + PartialEq>(vec: &[&T]) -> Vec<usize> {
    let white = T::max_value();
    vec.iter()
        .enumerate()
        .filter(|(_, &val)| *val == white)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>()
}

/// Gets all distances between cell edges within a single image. A mask is required, but may be
/// all false (i.e. no masking).
fn all_pixel_deltas<T: Bounded + PartialEq>(
    image: Array2<T>,
    mask: Array2<T>,
) -> PyResult<Vec<usize>> {
    if image.shape() != mask.shape() {
        let msg = format!(
            "Shape mismatch: img={:?}, mask={:?}",
            image.shape(),
            mask.shape()
        );
        return Err(PyValueError::new_err(msg));
    }

    let axis = Axis(0);
    let mut diffs = Vec::new();
    let img_height = image.shape()[0] as isize;
    for row in 0..img_height {
        // take the whole row
        let indices = Slice::new(row, Some(row + 1), 1);
        let row_img = image
            .slice_axis(axis, indices)
            .into_iter()
            .collect::<Vec<&T>>();
        let row_mask = mask
            .slice_axis(axis, indices)
            .into_iter()
            .collect::<Vec<&T>>();
        let mut row_diffs = pixel_deltas_from_row(row_img, row_mask);
        diffs.append(&mut row_diffs);
    }

    Ok(diffs)
}

/// Get all pixel distances between cell boundaries for a single row in an image
fn pixel_deltas_from_row<T: Bounded + PartialEq>(row: Vec<&T>, row_mask: Vec<&T>) -> Vec<usize> {
    let white = T::max_value();

    // find indices to split row into sub-rows
    let mut mask_split_indices = white_pixel_indices(row_mask.as_slice());

    // make sure we go to the end of the image
    mask_split_indices.push(row.len());

    // identify and measure sub-rows
    let mut idx_start = 0;
    let mut row_diffs: Vec<usize> = Vec::new();
    for idx_end in mask_split_indices {
        // avoid negative usize overflow panic and skip adjacent pixels
        if (idx_end == 0) || !(*row_mask[idx_end - 1] == white) {
            let split = &row.as_slice()[idx_start..idx_end];
            let mut sub_diffs = pixel_deltas_from_masked_run(split);
            row_diffs.append(&mut sub_diffs);
        }
        // increment regardless of whether the current pixel was usable so we don't
        // accidentally keep masked pixels in the event of adjacent mask indices
        idx_start = idx_end + 1;
    }
    row_diffs
}

/// The actual diff-getter. Operates on a masked subsection of a single row. If two measurements
/// are in adjacent pixels, this method selects the leftmost and rightmost adjacent locations and
/// throws out the others (i.e. it only accepts measurements where the boundary location is >1 px
/// away from the previous boundary location). The distance between the leftmost and rightmost
/// adjacent locations is not counted.
fn pixel_deltas_from_masked_run<T: Bounded + PartialEq>(sub_row: &[&T]) -> Vec<usize> {
    let edges = white_pixel_indices(sub_row);
    let mut diffs = Vec::new();
    let mut last_edge_idx = 0;
    for (idx_no, idx) in edges.iter().enumerate() {
        // keep the first found edge from measuring a difference from the left edge of the image
        if (idx_no > 0) & (*idx != last_edge_idx) {
            let diff = idx - last_edge_idx;
            if diff > 1 {
                diffs.push(diff);
            }
        }
        // don't use adjacent edge pixels
        last_edge_idx = *idx;
    }

    diffs
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_validate_image_path {
        use super::super::*;

        #[test]
        fn not_png() {
            let bad_path = Path::new(r"test_data/does_not_exist.jpg");
            let validated = validate_image_path(bad_path);
            assert!(validated.is_err());
        }

        #[test]
        fn bad_path() {
            let bad_path = Path::new("test_data/does_not_exist.png");
            let validated = validate_image_path(bad_path);
            assert!(validated.is_err());
        }

        #[test]
        fn good_validation() {
            let good_path = Path::new("test_data/gray.png");
            let validated = validate_image_path(good_path);
            assert!(validated.is_ok());
        }
    }

    mod test_load_image {
        use super::super::*;

        const GOOD_IMG: &[[u8; 8]; 7] = &[
            [
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MAX,
            ],
            [
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MIN,
            ],
            [
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
            ],
            [
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
            ],
            [
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MAX,
            ],
            [
                u8::MIN,
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
            ],
            [
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MAX,
                u8::MIN,
                u8::MIN,
                u8::MIN,
                u8::MIN,
            ],
        ];

        #[test]
        fn grayscale() {
            let img_path = Path::new("test_data/gray.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path).unwrap();
            assert_eq!(correct, loaded);
        }

        #[test]
        fn rgb() {
            let img_path = Path::new("test_data/not_gray.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path).unwrap();
            assert_eq!(correct, loaded);
        }

        #[test]
        fn not_8_bit() {
            let img_path = Path::new("test_data/not_gray_16.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path).unwrap();
            assert_eq!(correct, loaded);
        }
    }

    #[test]
    fn test_get_diffs_from_sub_row() -> PyResult<()> {
        let sub_row = &[
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
            &u8::MIN,
            &u8::MAX,
            &u8::MAX,
            &u8::MAX,
            &u8::MIN,
            &u8::MAX,
        ];
        let good: [usize; 3] = [3, 2, 2];
        let result = pixel_deltas_from_masked_run(sub_row);
        assert_eq!(result, good);

        Ok(())
    }

    #[test]
    fn test_get_diffs_from_row() {
        let row = vec![
            &u8::MAX,
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
            &u8::MAX,
            &u8::MAX,
            &u8::MIN,
            &u8::MAX,
            &u8::MIN,
            &u8::MAX,
            &u8::MAX,
            &u8::MIN,
        ];
        let row_mask = vec![
            &u8::MIN,
            &u8::MIN,
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
            &u8::MIN,
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
            &u8::MIN,
            &u8::MIN,
            &u8::MAX,
        ];
        let good: [usize; 2] = [3, 2];
        let result = pixel_deltas_from_row(row, row_mask);
        assert_eq!(result, good);
    }

    mod test_get_all_diffs {
        use super::super::*;

        #[test]
        fn test_good_value_with_mask() {
            let img_height = 3;
            let img_width = 8;
            let image = Vec::from([
                [
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                ], // gaps: 2, 3, 2
                [
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                ], // gaps: 2, 3, 2
                [
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                ], // gaps: 2, 3, 2
            ])
            .concat();
            let image = Array2::<u8>::from_shape_vec((img_height, img_width), image).unwrap();
            let mask = Vec::from([
                [
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                ], // keep gaps
                [
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                ], // new gaps: 3 only (2s masked)
                [
                    u8::MIN,
                    u8::MAX,
                    u8::MAX,
                    u8::MAX,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                ], // new gaps: 1
            ])
            .concat();
            let mask = Array2::<u8>::from_shape_vec((img_height, img_width), mask).unwrap();
            let good = [2, 3, 2, 3, 2];
            let result = all_pixel_deltas(image, mask).unwrap();
            assert_eq!(result, good);
        }

        #[test]
        fn test_good_value_without_mask() {
            let img_height = 3;
            let img_width = 8;
            let image = Vec::from([
                [
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                ], // gaps: 2, 3, 2
                [
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MAX,
                ], // gaps: 2, 3, 2
                // this row will indicate if the diff-from-left-side-of-image bug pope back up
                [
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                    u8::MAX,
                    u8::MIN,
                    u8::MIN,
                ], // gaps: 3
            ])
            .concat();
            let image = Array2::<u8>::from_shape_vec((img_height, img_width), image).unwrap();
            let mask = Vec::from([
                [
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                ], // keep gaps
                [
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                ], // new gaps: 3 only (2s masked)
                [
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                    u8::MIN,
                ], // keep gaps
            ])
            .concat();
            let mask = Array2::<u8>::from_shape_vec((img_height, img_width), mask).unwrap();
            let good = [2, 3, 2, 2, 3, 2, 3];
            let result = all_pixel_deltas(image, mask).unwrap();
            assert_eq!(result, good);
        }

        #[test]
        fn test_shape_mismatch() {
            let image = Array2::<u8>::zeros((1, 4));
            let mask = Array2::<u8>::zeros((4, 59));
            let result = all_pixel_deltas(image, mask);
            assert!(result.is_err());
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("Shape mismatch: img="));
        }
    }
}
