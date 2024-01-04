from typing import Literal
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9
    from typing_extensions import TypeAlias


DetrendMethod: TypeAlias = Literal["linear", "mean"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
Distance = Literal["l1", "l2", "inf", "h", "haversine"]
