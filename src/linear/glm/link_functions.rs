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
    pub fn compute<T: RealField + Float>(&self, x: T) -> T {
        match self {
            LinkFunction::Identity => x,
            LinkFunction::Log => x.ln(),
            LinkFunction::Logit => {
                // logit(p) = ln(p/(1-p))
                let x_clamped = x.clamp(T::epsilon(), T::one() - T::epsilon());
                (x_clamped / (T::one() - x_clamped)).ln()
            }
            LinkFunction::Inverse => x.recip(),
        }
    }

    pub fn deriv<T: RealField + Float>(&self, x: T) -> T {
        match self {
            LinkFunction::Identity => T::one(),
            LinkFunction::Log => x.recip(),
            LinkFunction::Logit => {
                // d/dp logit(p) = 1/(p(1-p))
                let x_clamped = x.clamp(T::epsilon(), T::one() - T::epsilon());
                (x_clamped * (T::one() - x_clamped)).recip()
            }
            LinkFunction::Inverse => -x.powi(2).recip(),
        }
    }

    pub fn inv<T: RealField + Float>(&self, x: T) -> T {
        match self {
            LinkFunction::Identity => x,
            LinkFunction::Log => x.exp(),
            LinkFunction::Logit => {
                // inv_logit(x) = exp(x)/(1+exp(x))
                let x_exp = x.exp();
                x_exp / (T::one() + x_exp)
            }
            LinkFunction::Inverse => x.recip(),
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
    pub fn compute<T: RealField + Float>(&self, x: T) -> T {
        match self {
            VarianceFunction::Gaussian => T::one(),
            VarianceFunction::Poisson => x,
            VarianceFunction::Binomial => {
                let x_clamped = x.clamp(T::epsilon(), T::one() - T::epsilon());
                x_clamped * (T::one() - x_clamped)
            }
            VarianceFunction::Gamma => x.powi(2),
        }
    }
}

