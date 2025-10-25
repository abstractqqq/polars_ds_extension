LIN_REG_EXPR_F64 = True
"""
If true, all linear regression expression will use f64 as the default data type
in the underlying implementation. If fase, f32 will be used. This only controls
linear regression expressions.

The memory footprint will be smaller, but it is possible to have slower speed than f64 for 
multiple reasons:
1. If input data is already in f64, then using f32 will incur additional casts, slowing
down the process.
2. If input data is not big enough, there won't be any noticeable difference in runtime.
"""

def _which_lin_reg(x: str) -> str:
    return x if LIN_REG_EXPR_F64 else f"{x}_f32"







