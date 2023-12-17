from typing import Literal
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9
    from typing_extensions import TypeAlias


DetrendMethod: TypeAlias = Literal["linear", "mean"]
AhoCorasickMatchKind: TypeAlias = Literal["standard", "left_most_first", "left_most_longest"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
