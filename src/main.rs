use ndarray::Array2;
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
    let mut actual_image = Array2::<u8>::zeros((img_height, img_width)).mapv(|a| a == u8::MAX);
    for row_idx in 0..img_height {
        for col_idx in 0..img_width {
            let actual_position = row_idx * (img_height + 1) + col_idx;
            actual_image[[row_idx, col_idx]] = *img_vec.get(actual_position).unwrap() == u8::MAX;
        }
    }
    Ok(actual_image)
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
