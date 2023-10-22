extern crate nalgebra as na;

use rv::prelude::*;
use rand;
use rand_distr::num_traits::Inv;
use rayon::prelude::*;
use rayon::iter::{ ParallelIterator, IndexedParallelIterator};
use rand_distr::{Normal, Distribution};
use na::{DMatrix, DVector, Dyn, SymmetricEigen};
use core::iter::Iterator;
use std::ops::Mul;
// use rayon::iter::IndexedParallelIterator;

// use itertools::Itertools;
// use rayon::prelude::*;

const LOG_2PI: f64 = 1.837877066409345444578941147617480054291973978803466899914960778699346285617763;

struct SuffStats where 
{
    n: usize,
    k: usize,
    xtx_eig: SymmetricEigen<f64, Dyn>,   // Eigendecomposition of x' * x
    xty: DVector<f64>,                   // x' * y
    yty: f64,                            // y' * y
}

impl SuffStats
    where 
        {
    fn new(x: &DMatrix<f64>, y: &DVector<f64>) -> SuffStats {
        let n = x.nrows();
        let k = x.ncols();
    
        assert_eq!(n, y.nrows());
    
        // We'll initialize xtx and xty to zero and then traverse the columns of x, updating xtx and xty as we go
        /**************************************************************************************************************
         * TODO: We're initializing these, only to write over them. Can we leave them uninitialized to start?
         * Maybe std::mem::MaybeUninit can help here?
         **************************************************************************************************************/
        let mut xtx = DMatrix::zeros(k, k);

        /**************************************************************************************************************
         * TODO: Can we do this in parallel? Seems tricky in Rust 
         * because we'd have to split ownership of xtx and xty over multiple threads.
         **************************************************************************************************************/
        xtx.par_column_iter_mut().enumerate().for_each(|(i, mut xtx_i)| {
            let x_i = &x.column(i);
            xtx_i[i] = x_i.dot(&x_i);
            for j in 0..i {
                let xtx_ij = x_i.dot(&x.column(j));
                xtx_i[j] = xtx_ij;
            }
        });

        for i in 0..k {
            for j in 0..i {
                xtx[(i, j)] = xtx[(j, i)];
            }
        }



        let xty = x.ad_mul(&y);

        

        let xtx_eig = na::SymmetricEigen::new(xtx);
        
        let yty = y.dot(y);
    
        SuffStats {
            n,
            k,
            xtx_eig,
            xty,
            yty
        }
    }

    fn ssr(&self, weights: &DVector<f64>) -> f64 {
        let xty = &self.xty;
        let yty = self.yty;

        // We need to compute wᵗ * (Xᵗ * X) * w = wᵗ * (Q * Λ * Qᵗ) * w
        //                                      = (Qᵗ * w)ᵗ * Λ * (Qᵗ * w)
        // So first we'll compute Qᵗ * w. This way we only have to compute the matrix-vector product once.
        let qt_w = &self.xtx_eig.eigenvectors.transpose() * weights;
        let wt_xtx_w = qt_w.zip_fold(&self.xtx_eig.eigenvalues, 0.0, |acc, qt_w_i, lam_i| {
            acc + qt_w_i * qt_w_i * lam_i
        });

        let result = yty - 2.0 * xty.dot(weights) + wt_xtx_w;
        result.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_suffstats() {
        let x = DMatrix::from_column_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let stats = SuffStats::new(&x, &y);
        assert_eq!(stats.n, 3);
        assert_eq!(stats.k, 2);
        assert_eq!(stats.xty, DVector::from_column_slice(&[14.0, 32.0]));
        assert_eq!(stats.yty, 14.0);
    }
}
struct Fit {
    suffstats: SuffStats,
    weights: DVector<f64>,
    prior_precision: f64,
    noise_precision: f64,
    
    iteration_count: usize,
    update_prior: bool,
    update_noise: bool,
}

fn symmetrize(x: &mut DMatrix<f64>) -> &DMatrix<f64> {
    // check that x is square
    let n = x.nrows();
    assert_eq!(n, x.ncols());
    for i in 0..n {
        for j in 0..i {
            x[(i, j)] += x[(j, i)];
            x[(i, j)] /= 2.0;
            x[(j, i)] = x[(i, j)];
        }
    }
    x
}

impl Fit {
    fn new(x: &DMatrix<f64>, y: &DVector<f64>) -> Fit {
        let suffstats = SuffStats::new(x, y);
        let prior_precision = 1.0;
        let noise_precision = 1.0;

        // Initialize the weights
        // This uses ridge regression in case xtx is singular
        let eigvals: &DVector<f64> = &suffstats.xtx_eig.eigenvalues;
        let eigvecs: &DMatrix<f64> = &suffstats.xtx_eig.eigenvectors;
        let diag = DMatrix::from_diagonal(
            &eigvals
                .map(|lam|  (1.0 + lam).inv())
        );
        let hinv = eigvecs * diag * eigvecs.transpose();
        symmetrize(&mut hinv.clone());
        let weights = hinv * &suffstats.xty;

        let fit = Fit {
            suffstats,
            weights,
            prior_precision,
            noise_precision,
            iteration_count: 0,
            update_prior: true,
            update_noise: true,
        };
        fit
    }

    fn ssr(&self) -> f64 {
        self.suffstats.ssr(&self.weights)
    }

    fn num_samples(&self) -> usize {
        self.suffstats.n
    }

    fn num_features(&self) -> usize {
        self.suffstats.k
    }


    fn xty(&self) -> &DVector<f64> {
        &self.suffstats.xty
    }



    fn effective_num_parameters(&self) -> f64 {
        let a = self.prior_precision;
        let b = self.noise_precision;

        // cache this to avoid recomputing
        let a_over_b = a / b;

        let eigvals =  &(self.suffstats.xtx_eig.eigenvalues);
        let result = eigvals
            .iter()
            .map(|lam| lam / (a_over_b + lam))
            .sum();
        debug_assert!(0.0 <= result);
        debug_assert!(result <= self.num_features() as f64);

        result
    }

    

    fn log_evidence(&self) -> f64 {
        let n = self.num_samples() as f64;
        let k = self.num_features() as f64;
        
        let a = self.prior_precision;
        let b = self.noise_precision;
        let rtr = self.ssr();

        0.5 * (
              k * a.ln()
            + n * b.ln()
            - (b * rtr + a * self.weights.dot(&self.weights))
            - self.logdet_hessian()
            - n * LOG_2PI
        )
    }


    fn logdet_hessian(&self) -> f64 {
        let a = self.prior_precision;
        let b = self.noise_precision;
        debug_assert!(a >= 0.0);
        debug_assert!(b >= 0.0);
        let eigvals = &(self.suffstats.xtx_eig.eigenvalues);

        eigvals
            .iter()
            .map(|lam| (a + b * lam).ln())
            .sum()
    }

    // TODO: store inverse_hessian in the struct and update in-place
    fn inverse_hessian(&self) -> DMatrix<f64> {
        let eigvals: &DVector<f64> = &(self.suffstats.xtx_eig.eigenvalues);
        let eigvecs: &DMatrix<f64> = &self.suffstats.xtx_eig.eigenvectors;
        let diag = DMatrix::from_diagonal(
            &eigvals
                .map(|lam| 1.0 / (self.prior_precision + self.noise_precision * lam))
        );
        let hinv = eigvecs * diag * eigvecs.transpose();
        symmetrize(&mut hinv.clone());
        hinv
    }

    fn update_weights(&mut self) {
        let b = self.noise_precision;

        let hinv = self.inverse_hessian();
        self.weights = hinv * self.xty();

        self.weights *= b;
    }

    // TODO: Use this to add an Iterator trait implementation for Fit
    fn update(&mut self) {
        let n = self.num_samples() as f64;
        
        let w = &self.weights;
        
        let wtw = w.dot(w);
        debug_assert!(wtw >= 0.0);
        
        if self.update_prior {
            self.prior_precision = self.effective_num_parameters() / wtw;
        }
        
        if self.update_noise {
            self.noise_precision = (n - self.effective_num_parameters()) / self.ssr();
            debug_assert!(self.noise_precision > 0.0);
        }

        self.update_weights();

        self.iteration_count += 1;
    }
}


fn fake_data(x: &DMatrix<f64>, prior_precision: f64, noise_precision: f64) -> (DVector<f64>, DVector<f64>) {
    let n = x.nrows();
    let k = x.ncols();


    let mut rng = rand::thread_rng();
    let weights_rv = Gaussian::new(0.0, prior_precision.sqrt().inv()).unwrap();
    let noise_rv = Gaussian::new(0.0, noise_precision.sqrt().inv()).unwrap();
    let w: DVector<f64> = DVector::from_vec(weights_rv.sample(k, &mut rng));
    let y = x * &w + DVector::from_vec(noise_rv.sample(n, &mut rng));
    (w, y)
}





fn main() {
    // Fill x with values from a normal distribution
    let n = 100000;
    let k = 100;
    let mut rng = rand::thread_rng();

    let x_rv = Gaussian::standard();
    let x: DMatrix<f64> = DMatrix::from_vec(n, k, x_rv.sample(n * k, &mut rng));
    let (w, y) = fake_data(&x, 7.0, 11.0);
    // let x = &DMatrix::from_fn(n, k, |_, _| rand::random::<f64>());
    let mut fit = Fit::new(&x, &y);
  
    println!();
    for _n in 0..10 {
        fit.update();
        // println!("-----------------------------------------");
        // println!("Iteration: {}", n);
        // println!("Weights: {}", fit.weights);
        // println!("Prior precision: {}\tNoise precision: {}", fit.prior_precision, fit.noise_precision);
        // println!("Noise precision: {}", fit.noise_precision);
        // println!("SSR: {}", fit.ssr());
        // println!();
        // println!("Effective number of parameters: {}", fit.effective_num_parameters());
        // println!("Log determinant of Hessian: {}", fit.logdet_hessian());
        // println!("Inverse Hessian: {}", fit.inverse_hessian());
        println!("Log evidence: {}", fit.log_evidence());
        // println!();
    }
}


// result is a vec in column-major order

// struct BayesianLinReg<'a, T, K: Dim, S> {
//     suffstats: SuffStats<T>,
//     n: isize,                          // number of samples (rows in X)
//     xtx: &'a mut Matrix<T, K, K, S>,   // X^T * X
//     xty: &'a mut Vector<T, K, S>,      // X^T * y
//     yty: &'a mut T,                    // y^T * y

//     hinv: &'a mut Matrix<T, K, K, S>,
//     weights: &'a mut Vector<T, K, S>,
//     sigma: &'a mut T,

//     prior_precision: &'a T,            // prior precision
//     noise_precision: &'a T,            // noise precision
// }



// impl<'a, T, K: Dim, S: RawStorage<T, N, K>> BayesianLinReg<'a, T, K, S> {
//     fn new<N: Dim>(
//         x: &'a Matrix<T, N, K, S>,
//         y: &'a Vector<T, N, S>,
//     ) -> BayesianLinReg<'a, T, K, S>
//     where
//         S: RawStorage<T, N, K>,
//      {
//         let n: isize = x.nrows() as isize;
//         // xtx = x' * x
//         let mut xtx = x.transpose() * x;
//         // xty = x' * y
//         let mut xty = x.transpose() * y;
//         // yty = y' * y
//         let mut yty = y.dot(y);
//         // hinv = xtx * prior_precision + I
//         let mut hinv = xtx.clone();
//         // weights = hinv^-1 * xty * prior_precision
//         let mut weights = xty.clone();
//         // sigma = (yty - weights' * xty) / (n - weights' * xtx - noise_precision)
//         let mut sigma = yty.clone();
//         BayesianLinReg {
//             n,
//             xtx: &mut xtx,
//             xty: &mut xty,
//             yty: &mut yty,
//             hinv: &mut hinv,
//             weights: &mut weights,
//             sigma: &mut sigma,
//             prior_precision: &T::zero(),
//             noise_precision: &T::zero(),
//         }
//     }

//     fn add_sample(&mut self, x: &Vector<T, K, S>, y: T) {
//         self.n += 1;
//         *self.xtx += x * x.transpose();
//         *self.xty += x * y;
//         *self.yty += y * y;
//     }

//     fn compute(&mut self) {
//         let n = self.n as T;
//         let prior_precision = *self.prior_precision;
//         let noise_precision = *self.noise_precision;

//         *self.hinv = *self.xtx * prior_precision + Matrix::identity();
//         *self.weights = self.hinv.try_inverse().unwrap() * *self.xty * prior_precision;
//         *self.sigma = (self.yty - self.weights.dot(&*self.xty)) / (n - self.weights.dot(&*self.xtx) - noise_precision);
//     }
// }

