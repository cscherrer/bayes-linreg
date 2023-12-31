extern crate nalgebra as na;

use core::iter::Iterator;
use na::{DMatrix, DVector, Dyn, SymmetricEigen};
use rand;
use rand_distr::num_traits::Inv;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rv::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};

extern crate streaming_iterator as streamit;

const LOG_2PI: f64 = 1.8378770664093456;

fn is_symmetric(x: &DMatrix<f64>) -> bool {
    let n = x.nrows();
    assert_eq!(n, x.ncols());
    for i in 0..n {
        for j in 0..i {
            if x[(i, j)] != x[(j, i)] {
                return false;
            }
        }
    }
    true
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
    debug_assert!(is_symmetric(x));
    x
}

#[derive(Clone)]
struct SuffStats {
    num_samples: usize,
    num_features: usize,
    xtx_eig: SymmetricEigen<f64, Dyn>, // Eigendecomposition of x' * x
    xty: DVector<f64>,                 // x' * y
    yty: f64,                          // y' * y
}

impl SuffStats {
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
        xtx.par_column_iter_mut()
            .enumerate()
            .for_each(|(i, mut xtx_i)| {
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
            num_samples: n,
            num_features: k,
            xtx_eig,
            xty,
            yty,
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
        assert_eq!(stats.num_samples, 3);
        assert_eq!(stats.num_features, 2);
        assert_eq!(stats.xty, DVector::from_column_slice(&[14.0, 32.0]));
        assert_eq!(stats.yty, 14.0);
    }
}

#[derive(Clone)]
struct Fit {
    weights: DVector<f64>,
    prior_precision: f64,
    noise_precision: f64,

    iteration_count: usize,
    update_prior: bool,
    update_noise: bool,
}

impl Fit {
    fn new(stats: &SuffStats) -> Fit {
        let prior_precision = 1.0;
        let noise_precision = 1.0;

        // Initialize the weights
        // This uses ridge regression in case xtx is singular
        let eigvals: &DVector<f64> = &stats.xtx_eig.eigenvalues;
        let eigvecs: &DMatrix<f64> = &stats.xtx_eig.eigenvectors;
        let diag = DMatrix::from_diagonal(&eigvals.map(|lam| (1.0 + lam).inv()));
        let mut hinv = eigvecs * diag * eigvecs.transpose();
        symmetrize(&mut hinv);
        let weights = hinv * &stats.xty;

        Fit {
            weights,
            prior_precision,
            noise_precision,
            iteration_count: 0,
            update_prior: true,
            update_noise: true,
        }
    }

    // TODO: store inverse_hessian in the struct and update in-place
    fn inverse_hessian(&self, stats: &SuffStats) -> DMatrix<f64> {
        let eigvals: &DVector<f64> = &(stats.xtx_eig.eigenvalues);
        let eigvecs: &DMatrix<f64> = &stats.xtx_eig.eigenvectors;
        let diag = DMatrix::from_diagonal(
            &eigvals.map(|lam| 1.0 / (self.prior_precision + self.noise_precision * lam)),
        );
        let mut hinv = eigvecs * diag * eigvecs.transpose();
        symmetrize(&mut hinv);
        debug_assert!(is_symmetric(&hinv));
        hinv
    }

    fn update_weights(&mut self, stats: &SuffStats) {
        let b = self.noise_precision;

        let hinv = self.inverse_hessian(stats);
        self.weights = hinv * &stats.xty;

        self.weights *= b;
    }

    fn effective_num_parameters(&self, stats: &SuffStats) -> f64 {
        let a = self.prior_precision;
        let b = self.noise_precision;

        // cache this to avoid recomputing
        let a_over_b = a / b;

        let eigvals = &stats.xtx_eig.eigenvalues;
        let result = eigvals.iter().map(|lam| lam / (a_over_b + lam)).sum();
        debug_assert!(0.0 <= result);
        debug_assert!(result <= stats.num_samples as f64);

        result
    }

    fn ssr(&self, stats: &SuffStats) -> f64 {
        let w = &self.weights;
        stats.ssr(&w)
    }

    // TODO: Use this to add an Iterator trait implementation for Fit
    fn update(&mut self, stats: &SuffStats) -> Option<&Self> {
        let n = stats.num_samples as f64;

        let w = &self.weights;

        let wtw = w.dot(&w);
        debug_assert!(wtw >= 0.0);

        if self.update_prior {
            self.prior_precision = self.effective_num_parameters(stats) / wtw;
        }

        if self.update_noise {
            self.noise_precision = (n - self.effective_num_parameters(stats)) / self.ssr(stats);
            debug_assert!(self.noise_precision > 0.0);
        }
        self.iteration_count += 1;

        self.update_weights(stats);

        Some(self)
    }

    fn logdet_hessian(&self, stats: &SuffStats) -> f64 {
        let a = self.prior_precision;
        let b = self.noise_precision;
        debug_assert!(a >= 0.0);
        debug_assert!(b >= 0.0);
        let eigvals = &(stats.xtx_eig.eigenvalues);

        eigvals.iter().map(|lam| (a + b * lam).ln()).sum()
    }

    fn log_evidence(&self, stats: &SuffStats) -> f64 {
        let n = stats.num_samples as f64;
        let k = stats.num_features as f64;

        let a = self.prior_precision;
        let b = self.noise_precision;
        let rtr = self.ssr(stats);

        0.5 * (k * a.ln() + n * b.ln()
            - (b * rtr + a * self.weights.dot(&self.weights))
            - self.logdet_hessian(stats)
            - n * LOG_2PI)
    }

    fn loglik_pvalue(&self, stats: &SuffStats, prior_precision: f64, noise_precision: f64) -> f64 {
        // self serves as the alternative hypothesis
        let loglik_alt = self.log_evidence(stats);

        let mut freedom = 0.0;
        if self.update_prior {
            freedom += 1.0;
        }
        if self.update_noise {
            freedom += 1.0;
        }

        // The null hypothesis is the same, but with the prior and noise precisions set to the values passed in
        let mut null_hyp = self.clone();
        null_hyp.prior_precision = prior_precision;
        null_hyp.noise_precision = noise_precision;
        null_hyp.update_prior = false;
        null_hyp.update_noise = false;

        null_hyp.update(stats);
        let loglik_null = null_hyp.log_evidence(stats);

        let test = ChiSquared::new(freedom).unwrap();
        test.cdf(2.0 * (loglik_alt - loglik_null))
    }
}

fn fake_data(
    x: &DMatrix<f64>,
    prior_precision: f64,
    noise_precision: f64,
) -> (DVector<f64>, DVector<f64>) {
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
    let n = 1000;
    let k = 10;
    let mut rng = rand::thread_rng();

    let x_rv = Gaussian::standard();
    let mut pval_sum = 0.0;
    let nsamples = 1000;
    for _ in 0..nsamples {
        let x: DMatrix<f64> = DMatrix::from_vec(n, k, x_rv.sample(n * k, &mut rng));
        let prior_precision = 7.0;
        let noise_precision = 11.0;
        let (_w, y) = fake_data(&x, prior_precision, noise_precision);
        let stats = SuffStats::new(&x, &y);
        let mut fit = Fit::new(&stats);

        // 10 iterations. This is very kludgy; we should use an Iterator trait implementation instead.
        for _n in 0..10 {
            fit.update(&stats);
        }
        let pval = fit.loglik_pvalue(&stats, prior_precision, noise_precision);
        pval_sum += pval;
    }

    let pval_mean = pval_sum / nsamples as f64;

    // Should be close to 0.5
    println!("mean pval: {:.3}", pval_mean);
}
