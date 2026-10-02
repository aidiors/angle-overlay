from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
import tkinter as tk
from contextlib import suppress
from tkinter import colorchooser, messagebox
from types import SimpleNamespace

import customtkinter as ctk

from .desktop import Hotkeys, Tray
from .devices import Adapter, hardware_adapters, preferred_adapter
from .monitors import list_monitors
from .overlay import Overlay, Windows
from .paths import app_directory, data_directory
from .runtime import load_predictor
from .settings import Settings, line_color, load_settings, save_settings

BACKENDS = {
    "Automatic (RTX → DirectML → CPU)": "auto",
    "NVIDIA RTX": "rtx",
    "DirectML (AMD / Intel)": "directml",
    "CPU": "cpu",
}


class Launcher:
    def __init__(self):
        self.win = Windows()
        self.monitors = list_monitors()
        self.settings_path = data_directory() / "settings.json"
        saved = load_settings(self.settings_path)
        ctk.set_appearance_mode("dark")
        ctk.deactivate_automatic_dpi_awareness()
        self.root = ctk.CTk(fg_color="#101218")
        self.root.title("Angle Overlay")
        self.root.iconbitmap(str(app_directory() / "assets/angle-overlay.ico"))
        primary = next(m for m in self.monitors if m.primary)
        self.ui_scale = primary.dpi_x / 96
        ctk.set_widget_scaling(self.ui_scale)
        ctk.set_window_scaling(self.ui_scale)
        height = max(460, min(860, round(primary.height * 96 / primary.dpi_y) - 100))
        self.root.geometry(f"620x{height}")
        self.root.minsize(480, 460)
        self.overlay = None
        self.loading = False
        self.closing = False
        self.destroyed = False
        self.desired_active = False
        self.pending_pause = False
        self.blocked_worker = None
        self.result = queue.Queue()
        self.actions = queue.SimpleQueue()
        self.next_display_check = time.monotonic() + 2
        self.backend = tk.StringVar(
            value=next(k for k, v in BACKENDS.items() if v == saved.backend)
        )
        try:
            adapters = hardware_adapters()
            preferred = preferred_adapter(adapters)
        except Exception:
            logging.exception("Could not enumerate DXGI adapters")
            adapters = [Adapter(0, "Default Windows adapter", 0)]
            preferred = adapters[0]
        self.adapters = {a.label: a.index for a in adapters}
        self.adapter = tk.StringVar(
            value=next(
                (a.label for a in adapters if a.index == saved.directml_device), preferred.label
            )
        )
        selected = next((m for m in self.monitors if m.device == saved.monitor_device), primary)
        self.monitor_choice = tk.StringVar(value=selected.label)
        self.color = saved.color
        self.thickness = tk.IntVar(value=saved.thickness)
        self.width_label = tk.StringVar(value=f"{saved.thickness} px")
        self.status = tk.StringVar(value="Move your cursor onto a line on the selected screen.")
        self.state = "Ready"

        outer = ctk.CTkFrame(self.root, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=24, pady=20)
        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            header, text="Angle Overlay", font=("Segoe UI", 27, "bold"), text_color="#f1f3f9"
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="To tray",
            width=80,
            height=32,
            command=self.hide,
            fg_color="#252b3a",
            hover_color="#343e54",
        ).pack(side="right")
        body = ctk.CTkScrollableFrame(outer, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True)

        def card(title):
            frame = ctk.CTkFrame(body, fg_color="#191d27", corner_radius=16)
            frame.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(
                frame, text=title, font=("Segoe UI", 12, "bold"), text_color="#929aaa"
            ).pack(anchor="w", padx=18, pady=(15, 10))
            return frame

        def menu(parent, variable, values, command):
            control = ctk.CTkOptionMenu(
                parent,
                variable=variable,
                values=values,
                command=command,
                height=40,
                corner_radius=9,
                fg_color="#252b3a",
                button_color="#30384b",
                button_hover_color="#404d68",
                dropdown_fg_color="#202633",
                dropdown_hover_color="#35415c",
                text_color="#e8edf7",
                text_color_disabled="#788398",
                font=("Segoe UI", 13),
                dropdown_font=("Segoe UI", 13),
                dynamic_resizing=False,
            )
            control.pack(fill="x", padx=18, pady=(0, 14))
            return control

        inference = card("INFERENCE")
        self.select = menu(inference, self.backend, list(BACKENDS), self.save)
        ctk.CTkLabel(inference, text="GPU for DirectML", text_color="#929aaa").pack(
            anchor="w", padx=18, pady=(0, 4)
        )
        self.gpu_select = menu(inference, self.adapter, list(self.adapters), self.save)
        display = card("DISPLAY & LINE")
        self.monitor_select = menu(
            display, self.monitor_choice, [m.label for m in self.monitors], self.change_monitor
        )
        ctk.CTkButton(
            display,
            text="Refresh displays",
            height=30,
            fg_color="#252b3a",
            hover_color="#343e54",
            command=self.refresh_displays,
        ).pack(anchor="w", padx=18, pady=(0, 12))
        style = ctk.CTkFrame(display, fg_color="transparent")
        style.pack(fill="x", padx=18, pady=(0, 12))
        self.color_button = ctk.CTkButton(
            style,
            text=self.color.upper(),
            width=125,
            height=36,
            command=self.choose_color,
            fg_color=self.color,
            hover_color=self.color,
        )
        self.color_button.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(style, textvariable=self.width_label, text_color="#d6ddea").pack(side="right")
        self.slider = ctk.CTkSlider(
            style,
            from_=1,
            to=12,
            number_of_steps=11,
            variable=self.thickness,
            command=self.change_style,
            progress_color="#7189ef",
            button_color="#a2b2ff",
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ctk.CTkLabel(
            display,
            text="256 × 256 physical-pixel crop · Thickness in screen pixels",
            text_color="#929aaa",
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=18, pady=(0, 14))

        status_card = ctk.CTkFrame(outer, fg_color="#191d27", corner_radius=16)
        status_card.pack(fill="x", pady=(12, 12))
        self.state_badge = ctk.CTkLabel(
            status_card, text="●  Ready", font=("Segoe UI", 13, "bold"), text_color="#70dda9"
        )
        self.state_badge.pack(anchor="w", padx=18, pady=(10, 0))
        self.status_label = ctk.CTkLabel(
            status_card,
            textvariable=self.status,
            font=("Segoe UI", 12),
            text_color="#adb6c6",
            wraplength=480,
            anchor="w",
            justify="left",
        )
        self.status_label.pack(fill="x", padx=18, pady=(3, 10))
        status_card.bind(
            "<Configure>",
            lambda event: self.status_label.configure(
                wraplength=max(200, int(event.width / self.root._get_widget_scaling()) - 40)
            ),
        )
        self.progress = ctk.CTkProgressBar(
            status_card, mode="indeterminate", height=3, progress_color="#829cff"
        )
        buttons = ctk.CTkFrame(outer, fg_color="transparent")
        buttons.pack(fill="x", pady=(0, 12))
        buttons.grid_columnconfigure((0, 1, 2), weight=1, uniform="buttons")
        self.start_button = ctk.CTkButton(
            buttons,
            text="Start overlay",
            command=self.toggle,
            height=42,
            corner_radius=10,
            fg_color="#7189ef",
            hover_color="#839af8",
            text_color="#0e1429",
            font=("Segoe UI", 13, "bold"),
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.pause_button = ctk.CTkButton(
            buttons,
            text="Pause",
            command=self.pause,
            height=42,
            fg_color="#252b3a",
            hover_color="#343e54",
            state="disabled",
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stop_button = ctk.CTkButton(
            buttons,
            text="Stop",
            command=self.stop,
            height=42,
            fg_color="#252b3a",
            hover_color="#343e54",
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        self.shortcuts_label = ctk.CTkLabel(
            outer,
            text="Ctrl+Alt+O  Toggle   ·   Ctrl+Alt+P  Pause   ·   Ctrl+Alt+Q  Stop",
            font=("Segoe UI", 11),
            text_color="#929aaa",
        )
        self.shortcuts_label.pack(anchor="w")
        ctk.CTkLabel(
            outer,
            text="Close / minimize → tray · Tray menu → Exit",
            font=("Segoe UI", 11),
            text_color="#687284",
        ).pack(anchor="w", pady=(3, 0))
        self.tray = Tray(app_directory() / "assets/angle-overlay.ico", self.actions.put)
        self.hotkeys = Hotkeys(self.actions.put)
        if self.hotkeys.errors:
            self.shortcuts_label.configure(
                text="; ".join(self.hotkeys.errors), text_color="#e9bc73"
            )
            logging.warning("%s", self.hotkeys.errors)
        self.root.protocol("WM_DELETE_WINDOW", self.window_close)
        self.root.bind("<Unmap>", self.minimized)
        self.root.after(50, self.poll)

    def selected_monitor(self):
        return next(m for m in self.monitors if m.label == self.monitor_choice.get())

    def save(self, _value=None):
        try:
            save_settings(
                self.settings_path,
                Settings(
                    BACKENDS[self.backend.get()],
                    self.adapters[self.adapter.get()],
                    self.selected_monitor().device,
                    self.color,
                    self.thickness.get(),
                ),
            )
        except OSError:
            logging.exception("Could not save preferences")
            self.status.set("Preferences could not be saved; this session still works.")

    def choose_color(self):
        _, color = colorchooser.askcolor(self.color, title="Line color", parent=self.root)
        if color is not None:
            try:
                self.color = line_color(color)
            except ValueError as error:
                messagebox.showerror("Line color", str(error), parent=self.root)
                return
            self.change_style()

    def change_style(self, _value=None):
        width = self.thickness.get()
        self.width_label.set(f"{width} px")
        rgb = tuple(int(self.color[i : i + 2], 16) for i in (1, 3, 5))
        text = "#101218" if sum(rgb) > 400 else "#ffffff"
        self.color_button.configure(
            text=self.color.upper(), fg_color=self.color, hover_color=self.color, text_color=text
        )
        if self.overlay is not None:
            self.overlay.set_line_style(self.color, width)
        self.save()

    def refresh_displays(self):
        try:
            current = list_monitors()
        except Exception:
            logging.exception("Display enumeration failed")
            self.status.set("Could not refresh displays. Check the application log.")
            return
        if current == self.monitors:
            return
        old = self.selected_monitor()
        self.monitors = current
        selected = next((m for m in current if m.device == old.device), current[0])
        self.monitor_select.configure(values=[m.label for m in current])
        self.monitor_choice.set(selected.label)
        if old != selected:
            self.change_monitor()

    def change_monitor(self, _value=None):
        self.save()
        if self.overlay is not None:
            paused = self.overlay.paused
            self.stop()
            self.pending_pause = paused
            self.start()

    def set_state(self, state):
        self.state = state
        color = {
            "Starting": "#e9bc73",
            "Paused": "#e9bc73",
            "Error": "#ff7c8b",
            "Stopped": "#929aaa",
        }.get(state, "#70dda9")
        self.state_badge.configure(text=f"●  {state}", text_color=color)

    def controls(self, busy):
        for control in (self.select, self.gpu_select):
            control.configure(state="disabled" if busy else "normal")
        self.pause_button.configure(state="normal" if self.overlay is not None else "disabled")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.start_button.configure(
            text="Cancel start"
            if self.loading
            else ("Disable overlay" if self.overlay is not None else "Start overlay")
        )

    def start(self):
        if self.closing or self.overlay is not None or self.loading:
            return
        if self.blocked_worker is not None:
            self.status.set("Previous inference did not stop. Exit the app before restarting.")
            self.set_state("Error")
            return
        self.desired_active = True
        self.loading = True
        self.save()
        self.controls(True)
        self.status.set("Preparing the model… First GPU launch may take longer.")
        self.set_state("Starting")
        self.progress.pack(fill="x", padx=18, pady=(0, 10))
        self.progress.start()
        backend, device_id = BACKENDS[self.backend.get()], self.adapters[self.adapter.get()]

        def load():
            predictor = None
            try:
                predictor = load_predictor(app_directory() / "models", backend, device_id)
                self.result.put((predictor, None))
            except Exception as error:
                if predictor is not None:
                    predictor.close()
                self.result.put((None, str(error)))

        threading.Thread(target=load, name="model-loader", daemon=True).start()

    def toggle(self):
        if self.overlay is not None or (self.loading and self.desired_active):
            self.stop()
        elif self.loading:
            self.desired_active = True
            self.status.set("Preparing the model…")
        else:
            self.start()

    def poll(self):
        if self.destroyed:
            return
        while not self.actions.empty():
            action = self.actions.get()
            if action == "exit":
                self.close()
                if self.destroyed:
                    return
            elif not self.closing:
                {
                    "show": self.show,
                    "toggle": self.toggle,
                    "pause": self.pause,
                    "stop": self.stop,
                    "tray_error": self.show,
                    "hotkey_error": lambda: self.status.set("Global shortcut service failed."),
                }[action]()
        if not self.closing and time.monotonic() >= self.next_display_check:
            self.next_display_check = time.monotonic() + 2
            self.refresh_displays()
            dpi = self.win.user.GetDpiForWindow(self.win.user.GetAncestor(self.root.winfo_id(), 2))
            if dpi and dpi / 96 != self.ui_scale:
                self.ui_scale = dpi / 96
                ctk.set_widget_scaling(self.ui_scale)
                ctk.set_window_scaling(self.ui_scale)
        try:
            predictor, error = self.result.get_nowait()
        except queue.Empty:
            pass
        else:
            self.loading = False
            self.progress.stop()
            self.progress.pack_forget()
            if self.closing or not self.desired_active:
                if predictor is not None:
                    predictor.close()
                if self.closing:
                    self.destroy()
                    return
                self.stopped(None)
            elif error:
                self.stopped(error)
            else:
                args = SimpleNamespace(
                    line_width=self.thickness.get(),
                    line_color=self.color,
                    monitor=self.selected_monitor(),
                    managed_hotkeys=True,
                    capture="winrt",
                    fps=120,
                    smoke_seconds=0,
                    report=None,
                )
                try:
                    self.overlay = Overlay(
                        predictor,
                        (256, 256),
                        self.win,
                        args,
                        parent=self.root,
                        on_close=self.stopped,
                    )
                except Exception as error:
                    logging.exception("Overlay initialization failed")
                    predictor.close()
                    for child in self.root.winfo_children():
                        if isinstance(child, tk.Toplevel):
                            child.destroy()
                    self.stopped(str(error))
                else:
                    self.overlay.paused = self.pending_pause
                    self.pending_pause = False
                    self.controls(True)
                    self.status.set(predictor.device_label)
                    self.set_state("Paused" if self.overlay.paused else "Running")
        if self.overlay is not None:
            current = "Paused" if self.overlay.paused else "Running"
            self.set_state(current)
            self.pause_button.configure(text="Resume" if self.overlay.paused else "Pause")
        if not self.destroyed:
            self.root.after(50, self.poll)

    def pause(self):
        if self.overlay is None:
            return
        self.overlay.paused = not self.overlay.paused
        self.overlay.generation += 1
        self.overlay.latest = None
        self.pause_button.configure(text="Resume" if self.overlay.paused else "Pause")
        self.set_state("Paused" if self.overlay.paused else "Running")

    def stop(self):
        self.desired_active = False
        self.pending_pause = False
        if self.overlay is not None:
            self.overlay.close()
        elif self.loading:
            self.status.set("Start cancelled. Waiting for preparation to finish…")

    def stopped(self, error):
        if self.overlay is not None and self.overlay.worker.is_alive():
            self.blocked_worker = self.overlay.worker
        self.overlay = None
        self.desired_active = False
        self.controls(False)
        self.pause_button.configure(text="Pause")
        self.status.set("Stopped." if error is None else "Overlay stopped because of an error.")
        self.set_state("Stopped" if error is None else "Error")
        if error is not None and not self.closing:
            self.show()
            messagebox.showerror("Angle Overlay", error, parent=self.root)

    def hide(self):
        if self.tray.available:
            self.root.withdraw()
        else:
            self.status.set("Tray unavailable. Use the window to control the overlay.")
            self.root.iconify()

    def window_close(self):
        if self.tray.available:
            self.hide()
        else:
            self.close()

    def minimized(self, event):
        if event.widget == self.root and self.root.state() == "iconic" and self.tray.available:
            self.root.after_idle(self.hide)

    def show(self):
        if not self.closing:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

    def destroy(self):
        if self.destroyed:
            return
        self.destroyed = True
        self.hotkeys.close()
        self.tray.close()
        self.win.close()
        for job in self.root.tk.call("after", "info"):
            with suppress(tk.TclError):
                self.root.after_cancel(job)
        self.root.destroy()

    def close(self):
        self.closing = True
        self.stop()
        if self.loading:
            self.status.set("Waiting for model preparation to finish…")
        else:
            self.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if not self.destroyed:
                self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, default=0)
    parser.add_argument("--runtime", choices=("auto", "cpu", "rtx", "directml"), default="auto")
    parser.add_argument("--directml-device", type=int, default=0, help="DXGI adapter index")
    parser.add_argument("--report")
    parser.add_argument("--gui-smoke-seconds", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--monitor", default="", help="Windows display device name")
    parser.add_argument("--line-color", type=line_color, default="#ff0000")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()
    if args.directml_device < 0 or not 1 <= args.line_width <= 12:
        parser.error("Invalid GPU adapter index or line width (1..12)")
    logging.basicConfig(
        filename=data_directory() / "app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.self_check or args.smoke_seconds:
        predictor = load_predictor(app_directory() / "models", args.runtime, args.directml_device)
        if args.self_check:
            import numpy as np

            angle, vector = predictor.predict(np.zeros((256, 256, 3), dtype=np.uint8))
            predictor.close()
            if args.report:
                with open(args.report, "w", encoding="utf-8") as report:
                    json.dump(
                        {
                            "ok": True,
                            "runtime": args.runtime,
                            "device": predictor.device_label,
                            "angle": angle,
                            "vector": vector,
                        },
                        report,
                    )
            return
        win = Windows()
        monitor = next((m for m in list_monitors() if m.device == args.monitor), None)
        if args.monitor and monitor is None:
            predictor.close()
            win.close()
            parser.error("The requested display is not connected")
        options = SimpleNamespace(
            line_width=args.line_width,
            line_color=args.line_color,
            monitor=monitor,
            capture="winrt",
            fps=120,
            smoke_seconds=args.smoke_seconds,
            report=args.report,
        )
        try:
            Overlay(predictor, (256, 256), win, options).run()
        finally:
            win.close()
        return
    try:
        app = Launcher()
        if args.gui_smoke_seconds:

            def finish_smoke():
                if args.report:
                    from pathlib import Path

                    Path(args.report).write_text(
                        json.dumps(
                            {
                                "tray_available": app.tray.available,
                                "hotkey_errors": app.hotkeys.errors,
                                "monitors": [m.label for m in app.monitors],
                                "ui_scale": app.ui_scale,
                                "window_state": app.root.state(),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                app.close()

            app.root.after(round(args.gui_smoke_seconds * 1000), finish_smoke)
        app.run()
    except Exception as error:
        logging.exception("Application failed")
        messagebox.showerror("Angle Overlay", str(error))


if __name__ == "__main__":
    main()
