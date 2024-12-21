mod processing;

use pyo3::prelude::*;

/// Gets pixel deltas from lines... only faster
#[pyfunction]
#[pyo3(signature = (image_path, mask_path=None))]
fn using_rust(image_path: String, mask_path: Option<String>) -> PyResult<Vec<usize>> {
    processing::get_px_deltas_from_lines(image_path, mask_path)
}

#[pymodule]
fn get_px_deltas_from_lines(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(using_rust, m)?)?;
    Ok(())
}
