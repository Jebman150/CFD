
pub trait FromIndex<const D: usize> {
    fn from_index(idx: [usize; D]) -> Self;
}

pub trait Linearizable<const D: usize> {
    fn linearize(&self, stride: &[usize; D]) -> usize;
}

pub struct MultiIndex<const D: usize> {
    indices: [usize; D],
    lower: [usize; D],
    upper: [usize; D],
    done: bool,
}

impl<const D: usize> MultiIndex<D> {
    pub fn new(size: &[usize; D]) -> Self {
        Self {
            indices: [0; D],
            lower: [0; D],
            upper: *size,
            done: false,
        }
    }

    pub fn set_lower_bound(mut self, axis: usize, lower_bound: usize) -> Self {
        self.lower[axis] = lower_bound;
        self.indices[axis] = lower_bound;
        self
    }

    pub fn set_upper_bound(mut self, axis: usize, upper_bound: usize) -> Self {
        self.upper[axis] = upper_bound+1;
        self
    }

    pub fn restrict_axis(mut self, axis: usize, value: usize) -> Self {
        self.upper[axis] = value + 1;
        self.lower[axis] = value;
        self.indices[axis] = value;
        self
    }

    pub fn exclude_border(mut self) -> Self {
        for axis in 0..D {
            self.upper[axis] -= 1;
            self.lower[axis] += 1;
            self.indices[axis] = self.lower[axis];
        }
        self
    }
}

impl<const D: usize> Linearizable<D> for MultiIndex<D> {
    fn linearize(&self, stride: &[usize; D]) -> usize {
        let mut sum = 0;
        for (axis, idx) in self.indices.iter().enumerate() {
            sum += idx * stride[axis];
        }
        sum
    }
}

impl<const D: usize> Iterator for MultiIndex<D> {
    type Item = [usize; D];

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices;

        for axis in 0..self.indices.len() {
            self.indices[axis] += 1;
            if self.indices[axis] < self.upper[axis] {
                break;
            };
            self.indices[axis] = self.lower[axis];
            
            if axis == self.indices.len()-1 {
                self.done = true;
                break;
            }
        }
        Some(result)
    }
}

pub struct TypedMultiIndex<T, const D: usize> {
    inner: MultiIndex<D>,
    _marker: std::marker::PhantomData<T>,
}

impl<T, const D: usize> TypedMultiIndex<T, D>
where
    T: FromIndex<D>
{
    pub fn new(size: &[usize; D]) -> Self {
        Self {
            inner: MultiIndex::<D>::new(size),
            _marker: std::marker::PhantomData
        }
    }

    pub fn set_lower_bound(mut self, axis: usize, lower_bound: usize) -> Self {
        self.inner = self.inner.set_lower_bound(axis, lower_bound);
        self
    }

    pub fn set_upper_bound(mut self, axis: usize, upper_bound: usize) -> Self {
        self.inner = self.inner.set_upper_bound(axis, upper_bound);
        self
    }

    pub fn restrict_axis(mut self, axis: usize, value: usize) -> Self {
        self.inner = self.inner.restrict_axis(axis, value);
        self
    }

    pub fn exclude_border(mut self) -> Self {
        self.inner = self.inner.exclude_border();
        self
    }
}

impl<T, const D: usize> Linearizable<D> for TypedMultiIndex<T, D> {
    fn linearize(&self, stride: &[usize; D]) -> usize {
        self.inner.linearize(stride)
    }
} 

impl<T, const D: usize> Iterator
    for TypedMultiIndex<T, D>
where
    T: FromIndex<D>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(T::from_index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellIndex<const D: usize> {
    pub idx: [usize; D],
}

impl<const D: usize> Linearizable<D> for CellIndex<D> {
    fn linearize(&self, stride: &[usize; D]) -> usize {
        let mut sum = 0;
        for (axis, idx) in self.idx.iter().enumerate() {
            sum += idx * stride[axis];
        }
        sum
    }
}

impl<const D: usize> FromIndex<D> for CellIndex<D> {
    fn from_index(idx: [usize; D]) -> Self {
        Self { idx }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceIndex<const D: usize> {
    pub axis: usize,
    pub idx: [usize; D],
}

impl<const D: usize> Linearizable<D> for FaceIndex<D> {
    fn linearize(&self, stride: &[usize; D]) -> usize {
        let mut sum = 0;
        for (axis, idx) in self.idx.iter().enumerate() {
            sum += idx * stride[axis];
        }
        sum
    }
}

impl<const D: usize> FaceIndex<D> {

    pub fn new(axis: usize, idx: [usize; D]) -> Self {
        Self { axis, idx }
    }
}

impl<const D: usize> FromIndex<D> for FaceIndex<D> {
    fn from_index(idx: [usize; D]) -> Self {
        Self {
            axis: 0,
            idx: idx
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_multi_index() {
        let size: [usize; 2] = [5, 5];
        let stride: [usize; 2] = [1, 5];
        for (i, idx) in MultiIndex::<2>::new(&size).enumerate() {
            for axis in 0..stride.len() {
                let check_idx = (i / stride[axis]) % size[axis];
                assert_eq!(idx[axis], check_idx as usize);
            }
        }
    }

    #[test]
    fn test_multi_index_restricted() {
        let size: [usize; 2] = [5, 5];
        let stride: [usize; 2] = [1, 5];
        let iter = MultiIndex::<2>::new(&size).restrict_axis(1, 2).enumerate();
        for (i, idx) in iter {
            println!("{}: {:?}", i, idx);
            for axis in 0..stride.len() {
                if axis == 1 {
                    continue;
                }
                let check_idx = (i / stride[axis]) % size[axis];
                assert_eq!(idx[axis], check_idx as usize);
            }
            assert_eq!(idx[1], 2);
        }
    }

    #[test]
    fn test_typed_index() {
        let size: [usize; 2] = [5, 5];
        let stride: [usize; 2] = [1, 5];
        for (i, cell_idx) in TypedMultiIndex::<CellIndex<2>,2>::new(&size).enumerate() {
            for axis in 0..stride.len() {
                let check_idx = (i / stride[axis]) % size[axis];
                assert_eq!(cell_idx.idx[axis], check_idx as usize);
            }
        }
    }

    #[test]
    fn test_typed_index_restricted() {
        let size: [usize; 2] = [5, 5];
        let stride: [usize; 2] = [1, 5];
        let iter = TypedMultiIndex::<CellIndex<2>,2>::new(&size).set_lower_bound(1, 2).enumerate();
        for (i, cell_idx) in iter {
            println!("{}: {:?}", i, cell_idx.idx);
            for axis in 0..stride.len() {
                if axis == 1 {
                    continue;
                }
                let check_idx = (i / stride[axis]) % size[axis];
                assert_eq!(cell_idx.idx[axis], check_idx as usize);
            }
            assert!(cell_idx.idx[1] >= 2);
        }
    }
}