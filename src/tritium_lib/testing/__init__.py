"""Tritium visual testing and ESP32 device automation.

Visual checks (VisualCheck, FlickerAnalyzer) require opencv-python and numpy.
Device automation (DeviceAPI, UITestRunner) requires requests.
All imports are lazy to avoid hard dependency issues.
"""

_LAZY_IMPORTS = {
    "DeviceAPI": ".device",
    "UITestRunner": ".runner",
    "VisualCheck": ".visual",
    "LayoutIssue": ".visual",
    "FlickerAnalyzer": ".flicker",
    "FlickerResult": ".flicker",
    "ScreenshotAnalyzer": ".visual_testing",
    "AnalysisReport": ".visual_testing",
    "BlankScreenResult": ".visual_testing",
    "UIElementResult": ".visual_testing",
    "ColorDistributionResult": ".visual_testing",
    "TextReadabilityResult": ".visual_testing",
    "OverlapResult": ".visual_testing",
    "BaselineComparisonResult": ".visual_testing",
    "MapTileResult": ".visual_testing",
    "MarkerResult": ".visual_testing",
    "LLMVisualAnalyzer": ".llm_visual",
    "VisionResult": ".llm_visual",
    "ScreenshotAnalysis": ".llm_visual",
}


def __getattr__(name):
    """Lazy imports for all testing classes."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
