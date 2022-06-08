use ndarray::{Array2, Axis, Slice};
use std::path::Path;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn main() {
    println!("Hello, world!");
}

fn validate_image_path(img_path: &Path) -> Result<()> {
    let img_path_str = img_path.display().to_string();
    if !img_path_str.ends_with(".png") {
        return Err(Error::from(format!("Non-PNG input: {img_path_str}")));
    }
    if !img_path.exists() {
        return Err(Error::from(format!(
            "Image path does not exist: {img_path_str}"
        )));
    }
    Ok(())
}

fn load_image(img_path: &Path) -> Result<Array2<bool>> {
    let img_base = image::open(img_path)?.to_luma8();
    let img_vec = img_base.as_raw();
    let img_width = img_base.width() as usize;
    let img_height = img_base.height() as usize;
    let image = Array2::<u8>::from_shape_vec((img_height, img_width), img_vec.to_owned())?
        .mapv(|a| a == u8::MAX);

    Ok(image)
}

fn find_true_indices(vec: &[&bool]) -> Vec<usize> {
    vec.iter()
        .enumerate()
        .filter(|(_, &val)| *val)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>()
}

/// the main one todo: wrap
pub fn get_px_deltas_from_lines(
    image_path: String,
    mask_path: Option<String>,
) -> Result<Vec<usize>> {
    let image_path = Path::new(&image_path);
    validate_image_path(image_path)?;
    let image = load_image(image_path)?;

    let mask = match mask_path {
        Some(pth) => {
            let mask_path = Path::new(&pth);
            validate_image_path(mask_path)?;
            load_image(image_path)?
        }
        // no mask
        None => image.clone().mapv(|_| false),
    };

    let result = get_all_diffs(image, mask)?;

    Ok(result)
}

/// Gets all distances between cell edges within a single image. A mask is required, but may be
/// all false (i.e. no masking).
fn get_all_diffs(image: Array2<bool>, mask: Array2<bool>) -> Result<Vec<usize>> {
    if image.shape() != mask.shape() {
        let msg = format!(
            "Shape mismatch: img={:?}, mask={:?}",
            image.shape(),
            mask.shape()
        );
        return Err(Error::from(msg));
    }

    let axis = Axis(0);
    let mut diffs = Vec::new();
    let img_height = image.shape()[0] as isize;
    for row in 0..img_height {
        // take the whole row
        let indices = Slice::new(row, Some(row + 1), 1);
        let row = image
            .slice_axis(axis, indices)
            .into_iter()
            .collect::<Vec<&bool>>();
        let row_mask = mask
            .slice_axis(axis, indices)
            .into_iter()
            .collect::<Vec<&bool>>();
        let mut row_diffs = get_diffs_from_row(row, row_mask)?;
        diffs.append(&mut row_diffs);
    }

    Ok(diffs)
}

/// Get all pixel distances between cell boundaries for a single row in an image
fn get_diffs_from_row(row: Vec<&bool>, row_mask: Vec<&bool>) -> Result<Vec<usize>> {
    // find indices to split row into sub-rows
    let mut mask_split_indices = find_true_indices(row_mask.as_slice());

    // make sure we go to the end of the image
    mask_split_indices.push(row.len());

    // identify and measure sub-rows
    let mut idx_start = 0;
    let mut row_diffs: Vec<usize> = Vec::new();
    for idx_end in mask_split_indices {
        // avoid negative usize overflow panic and skip adjacent pixels
        if (idx_end == 0) || !row_mask[idx_end - 1].to_owned() {
            let split = &row.as_slice()[idx_start..idx_end];
            let mut sub_diffs = get_diffs_from_sub_row(split)?;
            row_diffs.append(&mut sub_diffs);
        }
        // increment regardless of whether or not the current pixel was usable so we don't
        // accidentally keep masked pixels in the event of adjacent mask indices
        idx_start = idx_end + 1;
    }
    Ok(row_diffs)
}

/// The actual diff-getter. Operates on a masked subsection of a single row. If two measurements
/// are in adjacent pixels, this method selects the leftmost and rightmost adjacent locations and
/// throws out the others (i.e. it only accepts measurements where the boundary location is >1 px
/// away from the previous boundary location). The distance between the leftmost and rightmost
/// adjacent locations is not counted.
fn get_diffs_from_sub_row(sub_row: &[&bool]) -> Result<Vec<usize>> {
    let edges = find_true_indices(sub_row);
    let mut diffs = Vec::new();
    let mut last_edge_idx = 0;
    for idx in edges {
        if idx != last_edge_idx {
            let diff = idx - last_edge_idx;
            if diff > 1 {
                diffs.push(diff);
            }
        }
        // don't use adjacent edge pixels
        last_edge_idx = idx;
    }

    Ok(diffs)
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

        const GOOD_IMG: &[[bool; 8]; 7] = &[
            [true, false, false, false, false, false, false, true],
            [false, true, false, false, false, false, true, false],
            [false, false, true, false, false, true, false, false],
            [false, false, false, true, true, false, false, false],
            [true, false, false, true, false, false, false, true],
            [false, false, true, false, false, false, false, false],
            [false, true, false, true, false, false, false, false],
        ];

        #[test]
        fn grayscale() -> Result<()> {
            let img_path = Path::new("test_data/gray.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path)?;
            assert_eq!(correct, loaded);
            Ok(())
        }

        #[test]
        fn rgb() -> Result<()> {
            let img_path = Path::new("test_data/not_gray.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path)?;
            assert_eq!(correct, loaded);
            Ok(())
        }

        #[test]
        fn not_8_bit() -> Result<()> {
            let img_path = Path::new("test_data/not_gray_16.png");
            let correct = ndarray::arr2(GOOD_IMG);
            let loaded = load_image(img_path)?;
            assert_eq!(correct, loaded);
            Ok(())
        }
    }

    #[test]
    fn test_get_diffs_from_sub_row() -> Result<()> {
        let sub_row = &[
            &true, &false, &false, &true, &false, &true, &true, &true, &false, &true,
        ];
        let good: [usize; 3] = [3, 2, 2];
        let result = get_diffs_from_sub_row(sub_row)?;
        assert_eq!(result, good);

        Ok(())
    }

    #[test]
    fn test_get_diffs_from_row() -> Result<()> {
        let row = vec![
            &true, &false, &false, &true, &true, &true, &false, &true, &false, &true, &true, &false,
        ];
        let row_mask = vec![
            &false, &false, &false, &false, &true, &false, &false, &false, &true, &false, &false,
            &true,
        ];
        let good: [usize; 2] = [3, 2];
        let result = get_diffs_from_row(row, row_mask)?;
        assert_eq!(result, good);

        Ok(())
    }

    mod test_get_all_diffs {
        use super::super::*;

        #[test]
        fn test_good_value() -> Result<()> {
            let img_height = 2;
            let img_width = 8;
            let image = Vec::from([
                [true, false, true, false, false, true, false, true], // gaps: 2, 3, 2
                [true, false, true, false, false, true, false, true], // gaps: 2, 3, 2
            ])
            .concat();
            let image = Array2::<bool>::from_shape_vec((img_height, img_width), image)?;
            let mask = Vec::from([
                [false, false, false, false, false, false, false, false], // keep gaps
                [false, true, false, false, false, false, false, true], // new gaps: 3 only (2s masked)
            ])
            .concat();
            let mask = Array2::<bool>::from_shape_vec((img_height, img_width), mask)?;
            let good = [2, 3, 2, 3];
            let result = get_all_diffs(image, mask)?;
            assert_eq!(result, good);

            Ok(())
        }

        #[test]
        fn test_shape_mismatch() {
            let image = Array2::zeros((1, 4)).mapv(|a: i8| a != 0);
            let mask = Array2::zeros((4, 59)).mapv(|a: i8| a != 0);
            let result = get_all_diffs(image, mask);
            assert!(result.is_err());
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("Shape mismatch: img="));
        }
    }
}
