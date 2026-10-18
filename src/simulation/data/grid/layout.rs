
pub trait GridLayout<const D: usize> {
    fn dimensions(&self) -> [usize; D];
    fn stride(&self) -> [usize; D];
    fn total_count(&self) -> usize {
        self.dimensions().iter().product()
    }

    fn linearize(&self, idx: &[usize; D]) -> usize;
}

#[derive(Debug, Clone)]
pub struct CellLayout<const D: usize> {
    dims: [usize; D],
    stride: [usize; D],
}

impl<const D: usize> CellLayout<D> {
    pub fn new(cell_dimensions: [usize; D]) -> Self {
        let mut stride = [1; D];
        for i in 1..D {
            stride[i] = stride[i-1] * cell_dimensions[i-1];
        }

        Self {
            dims: cell_dimensions,
            stride,
        }
    }
}

impl<const D: usize> GridLayout<D> for CellLayout<D> {
    fn dimensions(&self) -> [usize; D] {
        return self.dims
    }

    fn stride(&self) -> [usize; D] {
        self.stride
    }

    fn linearize(&self, idx: &[usize; D]) -> usize {
        let mut sum = 0;
        for (axis, i) in idx.iter().enumerate() {
            sum += i * self.stride[axis];
        }
        sum
    }
}

#[derive(Debug, Clone)]
pub struct FaceLayout<const D: usize> {
    dims: [usize; D],
    stride: [usize; D],
    axis: usize,
}

impl<const D: usize> FaceLayout<D> {
    pub fn new(mut cell_dimensions: [usize; D], axis: usize) -> Self {
        cell_dimensions[axis] += 1;
        let mut stride = [1; D];
        for i in 1..D {
            stride[i] = stride[i-1] * cell_dimensions[i-1];
        }

        Self {
            dims: cell_dimensions,
            stride,
            axis
        }
    }

    pub fn get_axis(&self) -> usize {
        self.axis
    }
}

impl<const D: usize> GridLayout<D> for FaceLayout<D> {
    fn dimensions(&self) -> [usize; D] {
        return self.dims
    }

    fn stride(&self) -> [usize; D] {
        self.stride
    }

    fn linearize(&self, idx: &[usize; D]) -> usize {
        let mut sum = 0;
        for (axis, i) in idx.iter().enumerate() {
            sum += i * self.stride[axis];
        }
        sum
    }
}