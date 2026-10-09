use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn start_solver_log(file: &str) -> Result<(), Box<dyn Error>> {
    let filename = file;
    let mut file = File::options()
        .append(true)
        .create(true)
        .open(filename)?;
    writeln!(&mut file, "")?;

    Ok(())
}

pub fn log_solver_value(file: &str, value: f32) -> Result<(), Box<dyn Error>> {
    let filename = file;
    let mut file = File::options()
        .append(true)
        .create(true)
        .open(filename)?;
    write!(&mut file, "{value} ")?;

    Ok(())
}