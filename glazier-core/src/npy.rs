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

/// The dtype, shape and data offset of a `.npy` file's header.
fn parse_header(bytes: &[u8]) -> io::Result<(String, Vec<usize>, usize)> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    let (len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(bad("truncated .npy header"));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => return Err(bad(&format!("unsupported .npy version {v}"))),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + len)
            .ok_or_else(|| bad("truncated .npy header"))?,
    )
    .map_err(|_| bad("the .npy header is not text"))?;
    if header.contains("'fortran_order': True") {
        return Err(bad("Fortran-order arrays are not read"));
    }
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|r| r.split('\'').nth(1))
        .ok_or_else(|| bad("the .npy header has no descr"))?
        .to_string();
    let shape_text = header
        .split("'shape':")
        .nth(1)
        .and_then(|r| r.split('(').nth(1))
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| bad("the .npy header has no shape"))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| bad("the .npy shape is not integers"))
        })
        .collect::<io::Result<Vec<usize>>>()?;
    Ok((descr, shape, start + len))
}

/// Read a C-order label field of `uint32` or non-negative `int32` or `int64`.
///
/// Returns the values in lattice order and the array's shape.
///
/// # Errors
/// An I/O error, a header this reader does not accept, or a negative label.
pub fn read_labels(path: &Path) -> io::Result<(Vec<u32>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    let (descr, shape, offset) = parse_header(&bytes)?;
    let count: usize = shape.iter().product();
    let data = &bytes[offset..];
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let values: Vec<u32> = match descr.as_str() {
        "<u4" => data
            .chunks_exact(4)
            .take(count)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "<i4" => data
            .chunks_exact(4)
            .take(count)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .map(|v| u32::try_from(v).map_err(|_| bad(format!("negative label {v}"))))
            .collect::<io::Result<Vec<u32>>>()?,
        "<i8" | "<u8" => data
            .chunks_exact(8)
            .take(count)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .map(|v| u32::try_from(v).map_err(|_| bad(format!("label {v} is outside u32"))))
            .collect::<io::Result<Vec<u32>>>()?,
        other => return Err(bad(format!("labels of dtype {other} are not read"))),
    };
    if values.len() != count {
        return Err(bad("the .npy file is shorter than its shape".into()));
    }
    Ok((values, shape))
}

/// Read a C-order `float64` or `float32` array whose last axis has length two,
/// as pairs in lattice order.
///
/// # Errors
/// An I/O error, a header this reader does not accept, or a last axis that is
/// not two.
pub fn read_pairs(path: &Path) -> io::Result<(Vec<[f64; 2]>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    let (descr, shape, offset) = parse_header(&bytes)?;
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    if shape.last() != Some(&2) {
        return Err(bad(format!("the last axis is {:?}, want 2", shape.last())));
    }
    let count: usize = shape.iter().product();
    let data = &bytes[offset..];
    let flat: Vec<f64> = match descr.as_str() {
        "<f8" => data
            .chunks_exact(8)
            .take(count)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        "<f4" => data
            .chunks_exact(4)
            .take(count)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        other => return Err(bad(format!("a field of dtype {other} is not read"))),
    };
    if flat.len() != count {
        return Err(bad("the .npy file is shorter than its shape".into()));
    }
    Ok((flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect(), shape))
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

    #[test]
    fn labels_and_pairs_read_back_what_was_written() {
        let dir = std::env::temp_dir().join(format!("glazier_npy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let labels: Vec<u32> = (0..12).collect();
        let path = dir.join("labels.npy");
        write_u32(&path, &labels, 4, 3, 1).unwrap();
        let (read, shape) = read_labels(&path).unwrap();
        assert_eq!(read, labels);
        assert_eq!(shape, vec![3, 4]);

        let values: Vec<f64> = (0..12).map(|v| f64::from(v) * 0.5).collect();
        let path = dir.join("pairs.npy");
        write_f64(&path, &values, 2, 6, 1).unwrap();
        // Written as (6, 2), which a pair reader takes as six sites of two.
        let (pairs, shape) = read_pairs(&path).unwrap();
        assert_eq!(shape, vec![6, 2]);
        assert_eq!(pairs[5], [5.0, 5.5]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
