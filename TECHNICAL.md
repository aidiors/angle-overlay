# Technical documentation

[Русская версия](TECHNICAL.ru.md) · [User guide](USER-GUIDE.md) · [Home](README.md)

## From screen image to overlay

Angle Overlay estimates the angle of a line in a 256 × 256 pixel region around the mouse
pointer. It draws a straight line through the pointer, extending to the selected monitor's
edges. Capture and inference run locally. Screen images stay in memory.

The application has three parts: screen capture, angle prediction, and a transparent drawing
window. A worker handles capture and prediction. The UI thread draws the result and handles
settings, tray actions, and shortcuts.

| Component | Implementation |
|---|---|
| Settings window | CustomTkinter, Python 3.14 |
| Drawing window | Tk canvas with native Win32 window styles |
| Screen capture | DXcam with Windows Graphics Capture, MSS and Pillow fallbacks |
| NVIDIA inference | TensorRT-RTX and CUDA Python bindings |
| AMD, Intel, and NVIDIA inference | ONNX Runtime with DirectML |
| CPU inference | ONNX Runtime |
| Image conversion | NumPy |
| Tray icon | pystray |
| Model reconstruction and export | PyTorch, timm, and ONNX |
| Development tools | uv, Ruff, and ty |

## Model input and polar coordinates

The model takes RGB values from physical screen pixels. The inference interface accepts a
`float32` tensor of shape `[1, 3, 256, 256]`, in NCHW order, with values between 0 and 255.
Preprocessing is part of the model graph, so every inference backend uses the same geometry
and normalization.

The graph divides RGB values by 255 and samples a signed polar image. Its dimensions are
384 radial samples by 360 angular samples. The sampling grid is fixed:

```text
centre = (127.5, 127.5)
r ∈ [-127.5, 127.5]
θ[j] = j × π / 360,  j = 0, …, 359
x = 127.5 + r × cos(θ)
y = 127.5 + r × sin(θ)
```

Bilinear interpolation uses `align_corners=True` and border padding. Each RGB channel is then
normalized with mean 0.5 and standard deviation 0.5.

The radius includes both sides of the pointer. A line through the centre therefore appears
along the radial direction at its corresponding angle. This lets the network compare angle
columns while retaining image detail along the line.

A line has the same orientation at θ and θ + 180°. At the polar seam, that change also
reverses the signed radius: `P(r, θ + π) = P(-r, θ)`. Fine-crop sampling handles the seam by
extending the polar image with copies flipped along the radial axis. A crop near 0° or 180°
can cross the seam without an abrupt change of geometry.

## Coarse prediction

ConvNeXt Tiny extracts features from the full polar image. The final backbone stage has
768 channels. The angle head reduces the radial dimension in three ways: mean pooling,
maximum pooling, and learned attention. It concatenates those profiles with two positional
channels, `sin(2θ)` and `cos(2θ)`.

Three one-dimensional convolutions operate along the angle axis. They use 256 channels,
kernel widths 5, 5, and 3, GELU activations, and circular padding. A learned score produces
attention weights over the angle columns. The head uses those weights to pool both features
and positional vectors.

An MLP predicts a residual vector, scaled by 0.25 and added to the pooled positional vector.
The result is normalized and represents the coarse estimate as `[sin(2θ), cos(2θ)]`.
Double-angle encoding avoids a discontinuity between equivalent line directions at 0° and 180°.

## Fine correction

The coarse estimate determines the centre of a second polar crop. This crop keeps all
384 radial samples and resamples 65 angles across a ±4° window. Adjacent angular samples
are 0.125° apart. The spacing is a sampling resolution, not a guarantee of prediction accuracy.

The fine network receives five channels: RGB, normalized signed radius `r`, and `|r|`.
The radius channels tell it where a feature lies relative to the pointer.

| Fine stage | Output shape, excluding batch | Operation |
|---|---|---|
| Input | `5 × 384 × 65` | Polar RGB plus radius channels |
| Block 1 | `16 × 192 × 65` | Radial filter, convolution, GroupNorm, SiLU |
| Block 2 | `32 × 96 × 65` | Radial filter, convolution, GroupNorm, SiLU |
| Block 3 | `128 × 48 × 65` | Radial filter, convolution, GroupNorm, SiLU |
| Radial pooling | `256 × 65` | Concatenated mean and maximum |
| Correction head | One scalar | MLP with 128 hidden units |

Each convolution uses a 5 × 23 kernel, with stride 2 along the radius and stride 1 along the
angle axis. Before each convolution, the radial filter applies weights `[1, 2, 1] / 4` with
replicated border samples. It smooths the features before radial downsampling. Angular
resolution is retained through all three blocks.

The pooled profile is flattened and concatenated with the two coarse-vector components.
The MLP maps these 16,642 values to a scalar. A `tanh` limits the correction to ±4°.
The final vector is obtained by rotating the coarse vector through twice that correction.
At the start of fine training, the last layer has zero weights and bias, so the initial
fine prediction equals the coarse prediction.

The model has 34,033,768 parameters. Retaining radial samples preserves detail for refinement,
while wide angular kernels let the fine network compare nearby directions. Mean and maximum
pooling retain both distributed responses and local peaks. The correction remains local:
it cannot reliably recover a line whose coarse estimate falls outside the crop window.

## Training the selected weights

The deployed model was trained sequentially:

1. Train ConvNeXt Tiny and the coarse angle head for 5,200 updates, starting from ImageNet weights.
2. Freeze the coarse network and train a new fine branch for 5,200 updates.
3. Continue training the fine branch for another 5,200 updates, with an optimizer restart.

The total is 15,600 task updates. During both fine stages, the coarse network stays in
inference mode and its weights remain unchanged. Fine training therefore sees a stable
reference angle and learns the remaining correction.

Training used batches of 32, BF16 precision, MuSGD, and a vector Charbonnier loss. The fine
stages used a 200-update warmup followed by cosine learning-rate decay. The packaged checkpoint
is from seed 42. Training images contained synthetic lines and backgrounds.

The coarse backbone uses timm's `convnext_tiny.in12k_ft_in1k` pretrained configuration.
The deployment manifests identify the source checkpoint and record the hashes of the exported
models. Training images are not included in the application. The application source is licensed
under [Apache License 2.0](LICENSE), and dependencies retain their own licenses.

## Inference backends

All three backends accept the same RGB tensor and return two angle-vector components.
The application decodes them as:

```text
θ = (atan2(vector[0], vector[1]) / 2) mod π
```

Non-finite values and vectors with near-zero length are rejected.

### TensorRT-RTX

The NVIDIA backend loads a portable engine. Convolutional branches use BF16, while sampling
geometry and angle-head calculations use FP32. The input and output interfaces remain FP32.
TensorRT-RTX specializes execution for the local GPU when creating the execution context.
The runtime cache is stored per user, keyed by the engine hash and runtime version.

The predictor allocates pinned host buffers, device buffers, and one CUDA stream at startup.
For each prediction it copies the input to the pinned buffer, schedules a host-to-device
transfer, executes the engine, and schedules the two output values back to the host. It
synchronizes the stream before decoding the vector. Buffer addresses are reused across frames.

CUDA libraries are loaded only when this backend starts. The engine manifest is checked
against the installed TensorRT-RTX version. This backend has been tested on an RTX 5070.

### DirectML

The DirectML backend uses a separate FP32 ONNX graph at opset 17. The exporter decomposes GELU
into its exact expression and preserves bilinear image sampling in a form supported by this
backend.

The session uses sequential execution with memory-pattern optimization disabled. CPU execution
of unsupported graph nodes is disabled, and calls to the session are serialized with a lock.
Explicit DirectML selection therefore reports a startup error if the graph cannot run on the
chosen GPU.

The adapter is selected through DXGI enumeration independently of the capture monitor. DirectML
has been tested on NVIDIA hardware. AMD and Intel hardware have not been tested.

### CPU and automatic selection

The CPU backend runs the FP32 ONNX graph with four intra-operation threads and one
inter-operation thread. Automatic selection tries TensorRT-RTX on NVIDIA hardware, then
DirectML, then CPU. Each candidate loads the model and runs a warmup prediction before it is
accepted. The interface shows the backend actually in use. An explicitly selected backend
reports its error without switching to another backend.

Every predictor verifies the deployed model's SHA256 hash against its manifest before loading.

## Screen capture and Windows coordinates

The default capture path uses DXcam's WinRT backend, which wraps Windows Graphics Capture.
The camera is matched to the selected monitor by its Windows device name. Each prediction
requests the current pointer region, converts BGRA pixels to RGB, and copies the crop into an
owned NumPy array. Keeping an owned copy prevents a capture buffer from changing during inference.

A request may return no new frame on an unchanged desktop. In that case, the application
captures the current region with MSS rather than reusing a crop from an earlier pointer position.
If camera initialization fails, MSS handles capture. The hide-during-capture path uses Pillow.

Monitor rectangles come from Win32 display enumeration, and pointer positions come from
`GetPhysicalCursorPos`. The application uses PerMonitorV2 DPI awareness. Capture threads also
set their DPI context explicitly. All capture rectangles and overlay coordinates are expressed
in physical pixels, including negative origins in a multi-monitor desktop.

DXcam takes monitor-local coordinates. MSS and Pillow take desktop coordinates after adding
the monitor origin. The crop always measures 256 × 256 physical pixels, regardless of Windows
UI scaling. At a screen edge, repeated border pixels fill the missing part of the crop. When
the pointer leaves the selected monitor, capture is skipped and the line is hidden.

## Drawing and frame lifetime

For a decoded angle θ, the drawing direction is `(cos(θ), sin(θ))` in screen coordinates,
where positive Y points down. The drawing code intersects the infinite line through the
pointer with the monitor rectangle and uses the two extreme intersections as endpoints.
Horizontal and vertical directions are handled without division by a near-zero component.

The drawing window uses layered, transparent, topmost, tool-window, and no-activation Win32
styles. Mouse input passes through to the windows underneath. A Tk canvas draws the line;
its color and width can be changed without restarting inference. Width is measured in physical
pixels, from 1 to 12. The window's transparency key is `#010203`, so that color is reserved.

`WDA_EXCLUDEFROMCAPTURE` keeps the overlay out of supported screen-capture paths, preventing the
model from reading its own line. If Windows does not accept capture exclusion, the UI hides
the drawing window while Pillow captures the crop, then restores it.

Capture and inference are sequential in one worker. Image and result queues each hold at most
one item, replacing older entries. Results include a pointer position, timestamp, and generation
identifier. A generation change invalidates results from before a pause or restart.

The line is hidden if the pointer has moved more than 8 pixels from the captured position.
It is also hidden when the result is older than the greater of 2 seconds or three inference
durations. Drawing updates run on the UI thread, scheduled every 5 milliseconds. These intervals
control scheduling and expiry, rather than guaranteeing a frame rate.

## Desktop controls and settings

Global shortcuts use Win32 `RegisterHotKey` with `MOD_NOREPEAT` on a dedicated message thread.
Tray callbacks and shortcut handlers put actions into a queue consumed by the UI thread.
They do not call Tk directly from their own threads.

Closing or minimizing the settings window leaves the overlay running. Exit stops the worker,
releases capture and inference resources, unregisters shortcuts, and removes the tray icon.
A named mutex prevents a second application instance. When changing monitors, the application
stops the previous worker before starting another.

Preferences are stored in `%LOCALAPPDATA%\AngleOverlay`. Settings writes use a temporary JSON
file followed by an atomic replacement. The same directory holds logs and the RTX runtime
cache. Backend, adapter, monitor, color, and width are saved. Overlay activation is not restored
automatically on the next launch.

## Code layout

| Module | Responsibility |
|---|---|
| `launcher.py` | Settings UI and application lifecycle |
| `capture.py` | Crop extraction, image conversion, and capture fallbacks |
| `monitors.py`, `devices.py` | Display geometry and GPU adapter enumeration |
| `runtime.py`, `rtx_runtime.py` | Model loading and backend execution |
| `overlay.py` | Drawing window, worker, and result lifetime |
| `desktop.py`, `settings.py` | Tray, shortcuts, single-instance control, and preferences |
| `torch_model/` | Architecture reconstruction for export |

These modules are under `src/angle_overlay/`. Exported models and their manifests are in
`models/`. `tools/export_model.py` exports the complete preprocessing and prediction graph;
`tools/build_rtx.py` compiles the BF16 export into the portable NVIDIA engine. PyTorch is used
for reconstruction and export, and is not loaded by the inference application.
