# PRD-06: Metal/SDL2 Display Viewer

## Overview
Create a performant display viewer for depth/color/pointcloud visualization on macOS. Replaces OpenCV `cv2.imshow` (which has latency and threading issues) with SDL2 or native Metal rendering.

## Blocked By
- PRD-00 (Project Setup)
- PRD-05 (Colorizer — for depth visualization)

## Design

### Viewer Architecture
```
MLX Pipeline ──→ SDL2 Window ──→ Metal Layer
                     │
                ┌────┴────┐
                │ 2D View  │  Depth colorized + color side-by-side
                │ 3D View  │  Point cloud (stretch goal)
                └──────────┘
```

### Why SDL2 over cv2.imshow
- SDL2 uses Metal/CAMetalLayer natively on macOS
- No GIL contention (cv2.waitKey blocks)
- Better framerate (direct blit vs OpenCV overhead)
- Can run in separate process for true async display

### Why Not Native Metal View (yet)
- SDL2 is simpler (pip-installable, cross-platform)
- Metal view is Phase 2 optimization (PRD-06b)

## Implementation

```python
# realsense_mlx/display/viewer.py

class RealsenseViewer:
    """SDL2-based viewer for RealSense streams."""

    def __init__(self, width: int = 1280, height: int = 480, title: str = "RealSense MLX"):
        """Initialize SDL2 window. Width = depth_w + color_w for side-by-side."""
        ...

    def show(self, depth_color: mx.array, color: mx.array = None):
        """Display frames. depth_color: (H, W, 3) uint8 (colorized depth)."""
        ...

    def show_depth(self, depth: mx.array, colorizer: DepthColorizer = None):
        """Convenience: colorize + display depth frame."""
        ...

    def is_open(self) -> bool: ...
    def close(self): ...

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
```

### Minimal SDL2 Display Loop
```python
import ctypes
import sdl2
import sdl2.ext

class SDLDisplay:
    def __init__(self, width, height, title):
        sdl2.ext.init()
        self.window = sdl2.ext.Window(title, size=(width, height))
        self.window.show()
        self.surface = self.window.get_surface()

    def update(self, frame_rgb: np.ndarray):
        """Blit RGB frame to window surface."""
        # frame_rgb: (H, W, 3) uint8
        pixels = sdl2.ext.pixels3d(self.surface)
        pixels[:frame_rgb.shape[1], :frame_rgb.shape[0], :3] = frame_rgb.transpose(1, 0, 2)
        self.window.refresh()

    def pump_events(self) -> bool:
        """Process SDL events. Returns False if window closed."""
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return False
        return True

    def close(self):
        sdl2.ext.quit()
```

## Demo Script

```python
#!/usr/bin/env python3
"""rs-mlx-viewer: Live RealSense depth viewer with MLX processing."""

import pyrealsense2 as rs
import realsense_mlx as rsmlx

def main():
    # Setup camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Setup MLX processing
    depth_pipeline = rsmlx.DepthPipeline()
    colorizer = rsmlx.DepthColorizer(colormap="jet")
    viewer = rsmlx.RealsenseViewer(width=1280, height=480)

    try:
        while viewer.is_open():
            frames = pipeline.wait_for_frames()
            depth = mx.array(np.asanyarray(frames.get_depth_frame().get_data()))
            color = mx.array(np.asanyarray(frames.get_color_frame().get_data()))

            # Process
            filtered_depth = depth_pipeline.process(depth)
            depth_vis = colorizer.colorize(filtered_depth)

            # Display side-by-side
            viewer.show(depth_vis, color)
    finally:
        pipeline.stop()
        viewer.close()
```

## Acceptance Criteria

- [ ] SDL2 window opens and displays frames
- [ ] Side-by-side depth (colorized) + color display
- [ ] Handles window close/quit events
- [ ] ≥30 FPS display refresh at 640×480
- [ ] Context manager support (with statement)
- [ ] Falls back to cv2.imshow if SDL2 not installed

## Estimated Effort
8-12 hours

## Port Difficulty
**MEDIUM** — SDL2 integration is well-documented, main challenge is async display
