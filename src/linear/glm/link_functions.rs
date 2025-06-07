/// Defines the link functions for the GLM.
use faer_traits::RealField;
use num::Float;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkFunction {
    Identity, // Normal
    Log,      // Poisson
    Logit,    // Binomial
    Inverse,  // Gamma
}

impl LinkFunction {
    /// g(μ)
    pub fn link<T: RealField + Float>(&self, mu: T) -> T {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log => mu.ln(),
            LinkFunction::Logit => {
                // logit(p) = ln(p/(1-p))
                let one = T::one();
                (mu / (one - mu)).ln()
            }
            LinkFunction::Inverse => mu.recip(),
        }
    }

    /// g^(-1)(η)
    pub fn inv<T: RealField + Float>(&self, eta: T) -> T {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => {
                // inv_logit(x) = exp(x)/(1+exp(x))
                let one = T::one();
                let exp_eta = eta.exp();
                exp_eta / (one + exp_eta)
            }
            LinkFunction::Inverse => eta.recip(),
        }
    }

    /// Computes g'(μ)
    pub fn deriv<T: RealField + Float>(&self, mu: T) -> T {
        match self {
            LinkFunction::Identity => T::one(),
            LinkFunction::Log => mu.recip(),
            LinkFunction::Logit => {
                // d/dp logit(p) = 1/(p(1-p))
                let one = T::one();
                one / (mu * (one - mu))
            }
            LinkFunction::Inverse => -mu.recip().powi(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarianceFunction {
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
}

impl VarianceFunction {
    /// Variance(μ)
    pub fn variance<T: RealField + Float>(&self, mu: T) -> T {
        match self {
            VarianceFunction::Gaussian => T::one(),
            VarianceFunction::Poisson => mu,
            VarianceFunction::Binomial => {
                let one = T::one();
                mu * (one - mu)
            }
            VarianceFunction::Gamma => mu.powi(2),
        }
    }
}
