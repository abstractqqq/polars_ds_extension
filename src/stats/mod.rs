/// This submodule is mostly taken from the project statrs. See credit section in README.md
/// The reason I do not want to add it as a dependency is that it has a nalgebra dependency for
/// multi-variate distributions, which is something that I think will not be needed in this
/// package. Another reason is that if I want to do linear algebra, I would use Faer since Faer
/// performs better and nalgebra is too much of a dependency for this package right now.
pub mod beta;
pub mod gamma;
pub mod normal;

pub const PREC_ACC: f64 = 0.0000000000000011102230246251565;
pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;
//pub const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

#[inline]
pub fn is_zero(x: f64) -> bool {
    x.abs() < PREC_ACC
}
