//! A writer for the one NumPy array shape this crate produces.
//!
//! The label field crosses to Python as a file rather than through a binding,
//! so the simulation side needs no Python at all and the analysis side needs no
//! Rust. Only C-order `uint32` two-dimensional arrays are written, which is
//! what a lattice is.

use std::io::{self, Write};
use std::path::Path;

/// Write a `depth` by `height` by `width` `uint32` array as a version 1.0
/// `.npy` file. A depth of one writes a two-dimensional array, since that is
/// what a plane is and what the analysis side expects to load.
///
/// # Errors
/// Any I/O error from creating or writing the file.
pub fn write_u32(
    path: &Path,
    data: &[u32],
    width: usize,
    height: usize,
    depth: usize,
) -> io::Result<()> {
    assert_eq!(
        data.len(),
        width * height * depth,
        "data does not match the shape"
    );

    let shape = if depth > 1 {
        format!("({depth}, {height}, {width})")
    } else {
        format!("({height}, {width})")
    };
    let header = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': {shape}, }}");
    // The header, its two length bytes and the magic must total a multiple of
    // 64 bytes, padded with spaces and closed by a newline.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header = format!("{header}{}\n", " ".repeat(padding));

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1u8, 0u8])?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for &value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()
}

/// Write a `depth` by `height` by `width` `float64` array as a version 1.0
/// `.npy` file, the same way [`write_u32`] writes labels.
///
/// # Errors
/// Any I/O error from creating or writing the file.
pub fn write_f64(
    path: &Path,
    data: &[f64],
    width: usize,
    height: usize,
    depth: usize,
) -> io::Result<()> {
    assert_eq!(
        data.len(),
        width * height * depth,
        "data does not match the shape"
    );

    let shape = if depth > 1 {
        format!("({depth}, {height}, {width})")
    } else {
        format!("({height}, {width})")
    };
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}");
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header = format!("{header}{}\n", " ".repeat(padding));

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1u8, 0u8])?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for &value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_is_padded_to_a_multiple_of_sixty_four() {
        let dir = std::env::temp_dir().join("glazier-npy-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.npy");
        write_u32(&path, &[1, 2, 3, 4, 5, 6], 3, 2, 1).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(bytes.len(), 10 + header_len + 6 * 4);
        assert_eq!(&bytes[0..6], b"\x93NUMPY");
    }

    #[test]
    fn a_float_array_writes_its_own_descriptor() {
        let dir = std::env::temp_dir().join("glazier-npy-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.npy");
        write_f64(&path, &[1.5, 2.5, 3.5, 4.5], 2, 2, 1).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = String::from_utf8_lossy(&bytes[10..10 + header_len]);
        assert!(header.contains("<f8"), "{header}");
        assert_eq!(bytes.len(), 10 + header_len + 4 * 8);
    }

    #[test]
    fn a_volume_writes_a_three_dimensional_shape() {
        let dir = std::env::temp_dir().join("glazier-npy-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.npy");
        write_u32(&path, &[0u32; 24], 2, 3, 4).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = String::from_utf8_lossy(&bytes[10..10 + header_len]);
        assert!(header.contains("(4, 3, 2)"), "{header}");
    }
}
