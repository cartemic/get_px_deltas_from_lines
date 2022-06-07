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

    // todo: this doesn't go here; this stuff all needs its own tests
    get_all_diffs(image.clone(), image.clone())?;

    Ok(image)
}

/// find indices where vec is True
fn find_true_indices(vec: &[&bool]) -> Vec<usize> {
    vec.iter()
        .enumerate()
        .filter(|(_, &val)| *val)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>()
}

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
        let mut row_diffs = get_diffs_from_row(row.clone(), row)?;
        diffs.append(&mut row_diffs);
    }

    Ok(diffs)
}

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

            // todo: real stuff here pls -- inverting the mask is bogus as hell and only done for testing
            let mut test_input_dont_use = Vec::new();
            for _ in split {
                test_input_dont_use.push(&true)
            }
            let mut sub_diffs = get_sub_diffs_from_row(&test_input_dont_use)?;
            row_diffs.append(&mut sub_diffs);
        }
        // increment regardless of whether or not the current pixel was usable so we don't
        // accidentally keep masked pixels in the event of adjacent mask indices
        idx_start = idx_end + 1;
    }
    Ok(row_diffs)
}

fn get_sub_diffs_from_row(sub_row: &[&bool]) -> Result<Vec<usize>> {
    let edges = find_true_indices(sub_row);
    let mut diffs = Vec::new();
    let mut last_edge_idx = 0;
    for idx in edges {
        if diffs.len() > 0 {
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
}
