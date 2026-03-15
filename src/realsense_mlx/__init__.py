# realsense-mlx: MLX-accelerated processing for Intel RealSense on Apple Silicon
"""
Usage:
    import realsense_mlx as rsmlx

    pipeline = rsmlx.DepthPipeline()
    colorizer = rsmlx.DepthColorizer()
    pc_gen = rsmlx.PointCloudGenerator(intrinsics, depth_scale)

    # Or use the all-in-one processor:
    proc = rsmlx.RealsenseProcessor(intrinsics, depth_scale=0.001)
    result = proc.process(depth_frame)
"""

__version__ = "0.1.0"

__all__ = [
    "DepthPipeline", "DepthColorizer", "PointCloudGenerator", "Aligner",
    "FormatConverter", "CameraIntrinsics", "RealsenseProcessor", "ProcessingResult",
    "OccupancyGridGenerator", "OccupancyGrid", "ObstacleDetector", "ObstacleResult",
    "StereoDepthEstimator", "StereoDepthConfig", "StereoCamera", "StereoCameraError",
]

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
    if name == "RealsenseProcessor":
        from realsense_mlx.processor import RealsenseProcessor
        return RealsenseProcessor
    if name == "ProcessingResult":
        from realsense_mlx.processor import ProcessingResult
        return ProcessingResult
    # Robotics modules
    if name == "OccupancyGridGenerator":
        from realsense_mlx.robotics.occupancy import OccupancyGridGenerator
        return OccupancyGridGenerator
    if name == "OccupancyGrid":
        from realsense_mlx.robotics.occupancy import OccupancyGrid
        return OccupancyGrid
    if name == "ObstacleDetector":
        from realsense_mlx.robotics.obstacles import ObstacleDetector
        return ObstacleDetector
    if name == "ObstacleResult":
        from realsense_mlx.robotics.obstacles import ObstacleResult
        return ObstacleResult
    # Stereo pipeline — vendor-SDK-free, works with any USB stereo camera.
    if name == "StereoDepthEstimator":
        from realsense_mlx.stereo.depth import StereoDepthEstimator
        return StereoDepthEstimator
    if name == "StereoDepthConfig":
        from realsense_mlx.stereo.depth import StereoDepthConfig
        return StereoDepthConfig
    if name == "StereoCamera":
        from realsense_mlx.stereo.camera import StereoCamera
        return StereoCamera
    if name == "StereoCameraError":
        from realsense_mlx.stereo.camera import StereoCameraError
        return StereoCameraError
    raise AttributeError(f"module 'realsense_mlx' has no attribute {name!r}")
