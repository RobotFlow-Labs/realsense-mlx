# realsense-mlx: MLX-accelerated processing for Intel RealSense on Apple Silicon
"""
Usage:
    import realsense_mlx as rsmlx

    pipeline = rsmlx.DepthPipeline()
    colorizer = rsmlx.DepthColorizer()
    pc_gen = rsmlx.PointCloudGenerator(intrinsics, depth_scale)
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading MLX until needed
def __getattr__(name):
    if name == "DepthPipeline":
        from realsense_mlx.filters import DepthPipeline
        return DepthPipeline
    if name == "DepthColorizer":
        from realsense_mlx.filters.colorizer import DepthColorizer
        return DepthColorizer
    if name == "PointCloudGenerator":
        from realsense_mlx.geometry.pointcloud import PointCloudGenerator
        return PointCloudGenerator
    if name == "Aligner":
        from realsense_mlx.geometry.align import Aligner
        return Aligner
    if name == "FormatConverter":
        from realsense_mlx.converters.format_converter import FormatConverter
        return FormatConverter
    if name == "CameraIntrinsics":
        from realsense_mlx.geometry.intrinsics import CameraIntrinsics
        return CameraIntrinsics
    raise AttributeError(f"module 'realsense_mlx' has no attribute {name!r}")
