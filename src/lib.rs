mod processing;

use cpython::{exc, py_fn, py_module_initializer, PyErr, PyResult, Python};

py_module_initializer!(get_px_deltas_from_lines, |py, m| {
    m.add(py, "__doc__", "Gets pixel deltas from lines... only faster")?;
    m.add(
        py,
        "get_px_deltas_from_lines",
        py_fn!(
            py,
            wrapped_get_px_deltas_from_lines(image_path: String, mask_path: Option<String>)
        ),
    )?;
    Ok(())
});

fn wrapped_get_px_deltas_from_lines(
    py: Python,
    image_path: String,
    mask_path: Option<String>,
) -> PyResult<Vec<usize>> {
    match processing::get_px_deltas_from_lines(image_path, mask_path) {
        Ok(val) => Ok(val),
        Err(e) => Err(PyErr::new::<exc::ValueError, _>(py, e.to_string())),
    }
}
