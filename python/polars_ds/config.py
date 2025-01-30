# Configs used in transforms and pipelines
# STREAM_IN_TRANSFORM: bool = False
# Level of optimiztion and memory usage, etc.
# Is there a better way to do this?


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


def _lin_reg_expr_symbol(x: str) -> str:
    if LIN_REG_EXPR_F64:
        return x
    else:
        return x + "_f32"
