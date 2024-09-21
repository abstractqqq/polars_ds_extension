/// This submodule is mostly taken from the project statrs. See credit section in README.md
/// I do not want to add it as a dependency because a lot of what it offers won't fit.
///
/// MIT License
/// Copyright (c) 2016 Michael Ma
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
/// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
/// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
/// and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
/// The above copyright notice and this permission notice shall be included in all copies or substantial portions
/// of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
/// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
/// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
/// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
/// THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pub mod beta;
pub mod gamma;
pub mod normal;

pub const PREC_ACC: f64 = 0.0000000000000011102230246251565;
pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;
//pub const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;
