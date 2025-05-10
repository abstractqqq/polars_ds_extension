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

impl From<&str> for LinkFunction {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_ref() {
            "id" | "identity" => Self::Identity,
            "log" => Self::Log,
            "logit" => Self::Logit,
            "inv" | "inverse" => Self::Inverse,
            _ => Self::Identity,
        }
    }
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
            LinkFunction::Inverse => -mu.powi(2).recip(),
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

impl From<LinkFunction> for VarianceFunction {
    fn from(value: LinkFunction) -> Self {
        match value {
            LinkFunction::Identity => Self::Gaussian,
            LinkFunction::Log => Self::Poisson,
            LinkFunction::Logit => Self::Binomial,
            LinkFunction::Inverse => Self::Gamma,
        }
    }
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
