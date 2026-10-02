# User guide

[Русская версия](USER-GUIDE.ru.md) · [Home](README.md)

Angle Overlay extends the line beneath your mouse pointer to the edges of your chosen screen.
It stays above other windows, lets clicks pass through, and processes screen images on your PC.

## Install and start

1. Open **Releases** on the project's GitHub page.
2. Download `AngleOverlay-<version>-windows-x64.zip` from **Assets**.
   The automatic “Source code” downloads are for developers, not the application.
3. Extract the entire ZIP to a folder. Keep `_internal`, `models`, and `assets` beside the EXE.
4. Open **AngleOverlay.exe**. Python and a CUDA Toolkit are not required.
5. Select your screen under **DISPLAY & LINE**, then click **Start overlay**.
6. Move your pointer onto the line you want to extend. The overlay follows the pointer.

The app is for Windows x64. Update your graphics driver before using GPU inference.
DirectML requires a DirectX 12 GPU and Windows 10 version 1903 or newer.
The packaged RTX engine targets NVIDIA RTX Ampere, Ada, and Blackwell GPUs.
Unsupported NVIDIA GPUs can use DirectML or CPU instead.

## Choose a GPU

Leave **INFERENCE** on **Automatic** for normal use:

- Supported NVIDIA RTX: the app tries the optimized RTX runtime.
- AMD, Intel, or other compatible cards: it tries **DirectML** GPU inference.
- If a GPU runtime cannot start: Automatic mode uses CPU and displays this in the status.

You can select **NVIDIA RTX**, **DirectML (AMD / Intel)**, or **CPU** explicitly.
On a laptop with two GPUs, select the desired card under **GPU for DirectML**.
An explicit GPU selection shows an error if it cannot start; it does not silently switch to CPU.
The first RTX startup can take longer while the model is prepared for your card. Let it finish.

## Controls

| Shortcut | Action |
|---|---|
| **Ctrl+Alt+O** | Turn the overlay on or off |
| **Ctrl+Alt+P** | Pause or resume |
| **Ctrl+Alt+Q** | Stop the overlay |

Shortcuts work while settings are hidden. If another app uses one of them, the settings window
shows a conflict; use its buttons or the tray menu instead.

Minimize or close the settings window, or click **To tray**, to hide it in the notification area.
The overlay keeps working. Click its tray icon to reopen settings. Right-click the icon for
overlay controls and **Exit**, which closes the app completely.

## Screen and line appearance

Choose a screen in **DISPLAY & LINE**. The app supports different resolutions, mixed Windows
scales, portrait screens, and displays positioned to the left or above the primary screen.
The line appears only on the selected screen and hides when your pointer leaves it.
Use **Refresh displays** after changing your setup if needed.

Click the color button to choose a line color. Adjust the slider for a thickness of **1–12 screen
pixels**. Changes apply immediately. The color `#010203` is reserved for transparency.
Screen, runtime, GPU, color and thickness are saved automatically. The overlay starts disabled
when you launch the app again.

## Troubleshooting

| Problem | What to try |
|---|---|
| No line appears | Start/resume the overlay, choose the correct screen, and hold the pointer on a visible line briefly. |
| GPU startup fails | Update the driver. Try DirectML, choose another listed adapter, or use CPU. |
| Startup seems slow | Wait for the first RTX preparation to finish; later starts reuse its cache. |
| The app seems to have disappeared | Find the Angle Overlay icon in the tray or its hidden-icons menu. |
| A shortcut does nothing | Check the conflict message in settings; use buttons or the tray menu. |
| Windows warns about the EXE | The app is unsigned. Download only from the project's own release page. |

The model estimates line orientation; it can make mistakes on faint, ambiguous, or unfamiliar
images. It does not guarantee perfect alignment on every desktop or in every application.

Screenshots are processed in memory and are not saved or uploaded. Settings, logs and the RTX
cache are stored in `%LOCALAPPDATA%\AngleOverlay`. Windows capture exclusion keeps the overlay
out of supported capture APIs; other recording programs may behave differently.

For technical details, see [Technical documentation](TECHNICAL.md).
