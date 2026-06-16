"""mineru_layout — chart/layout detection (PP-DocLayoutV2).

Heavy imports (torch via ChartDetector / PPDocLayoutV2LayoutModel) are loaded
lazily so that pure utilities like ``bbox_utils`` can be imported without
pulling in the full model stack.
"""

__all__ = ["ChartDetector", "PPDocLayoutV2LayoutModel"]


def __getattr__(name):
    if name == "ChartDetector":
        from .chart_detector import ChartDetector
        return ChartDetector
    if name == "PPDocLayoutV2LayoutModel":
        from .pp_doclayoutv2 import PPDocLayoutV2LayoutModel
        return PPDocLayoutV2LayoutModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
