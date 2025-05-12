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
                let mu_clamped = mu.clamp(T::epsilon(), T::one() - T::epsilon());
                (mu_clamped / (T::one() - mu_clamped)).ln()
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
                let eta_exp = eta.exp();
                eta_exp / (T::one() + eta_exp)
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
                let mu_clamped = mu.clamp(T::epsilon(), T::one() - T::epsilon());
                (mu_clamped * (T::one() - mu_clamped)).recip()
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
                let mu_clamped = mu.clamp(T::epsilon(), T::one() - T::epsilon());
                mu_clamped * (T::one() - mu_clamped)
            }
            VarianceFunction::Gamma => mu.powi(2),
        }
    }
}
