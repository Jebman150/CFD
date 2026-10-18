use std::error::Error;
use std::fs::{File, read_dir, remove_file};
use std::io::Write;
use std::path::Path;

pub fn clear_log_directory<P: AsRef<Path>>(dir: P) -> std::io::Result<()> {
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            clear_log_directory(&path)?;
        } else if path.is_file() {
            remove_file(&path)?;
        }
    }

    Ok(())
}

pub fn start_log_entry(file: &str) -> Result<(), Box<dyn Error>> {
    let filename = file;
    let mut file = File::options()
        .append(true)
        .create(true)
        .open(filename)?;
    writeln!(&mut file, "")?;

    Ok(())
}

pub fn log_value(file: &str, value: f32) -> Result<(), Box<dyn Error>> {
    let filename = file;
    let mut file = File::options()
        .append(true)
        .create(true)
        .open(filename)?;
    write!(&mut file, "{value} ")?;

    Ok(())
}