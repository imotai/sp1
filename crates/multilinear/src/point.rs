use serde::{Deserialize, Serialize};

/// A wrapper struct for a multivariate point.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Default)]
pub struct Point<K>(pub(crate) Vec<K>);

impl<K: Clone + Default> Point<K> {
    pub fn new(point: Vec<K>) -> Self {
        Point(point)
    }

    pub fn dimension(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    pub fn first_k_points(&self, k: usize) -> Self {
        Point(self.0[..k].to_vec())
    }

    pub fn last_k_points(&self, k: usize) -> Self {
        Point(self.0[self.0.len() - k..].to_vec())
    }

    pub fn last_k(&self, k: usize) -> Self {
        Point(self.0[self.0.len() - k..].to_vec())
    }

    pub fn split_at(&self, k: usize) -> (Self, Self) {
        (Point(self.0[..k].to_vec()), Point(self.0[k..].to_vec()))
    }

    pub fn split_at_first(&self) -> (K, Self) {
        (self.0[0].clone(), Point(self.0[1..].to_vec()))
    }

    pub fn split_at_last(&self) -> (Self, K) {
        (Point(self.0[..self.0.len() - 1].to_vec()), self.0[self.0.len() - 1].clone())
    }

    pub fn reversed_point(&self) -> Self {
        Point(self.0.iter().rev().cloned().collect())
    }

    pub fn add_dimension(&mut self, dim_val: K) {
        self.0.insert(0, dim_val);
    }

    pub fn remove_last_coordinate(&mut self) -> K {
        self.0.pop().unwrap()
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    pub fn ith_coordinate(&self, i: usize) -> K {
        self.0[i].clone()
    }

    pub fn ith_coordinate_or_default(&self, i: Option<usize>) -> K {
        i.and_then(|i| self.0.get(i).cloned()).unwrap_or_default()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, K> {
        self.0.iter()
    }

    pub fn to_vec(&self) -> Vec<K> {
        self.0.clone()
    }
}
