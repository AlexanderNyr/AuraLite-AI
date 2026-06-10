import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from model_engine import AuraLiteEngine
import threading
import multiprocessing
import os


class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraLite AI v2.0 — Modern Transformer Edition")
        self.root.geometry("800x960")
        self.root.minsize(750, 900)
        self.root.configure(bg="#f5f6f7")

        self.engine = AuraLiteEngine()
        self.is_trained = False
        self.selected_file_path = None
        self.stop_event = threading.Event()

        # ---- Styles ----------------------------------------------------
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10), background="#f5f6f7")
        style.configure("Header.TLabel",
                         font=("Segoe UI", 16, "bold"), background="#f5f6f7")
        style.configure("Sub.TLabel",
                         font=("Segoe UI", 9, "italic"), background="#f5f6f7")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        # ---- Main frame ------------------------------------------------
        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ================================================================
        #  Header
        # ================================================================
        header = ttk.Label(main_frame,
                           text="🌟 AuraLite AI v2.0 — Modern Edition",
                           style="Header.TLabel")
        header.pack(pady=(0, 4))

        if self.engine.device.type == "cuda":
            dev = "GPU: CUDA 🟢"
        else:
            dev = f"CPU: {self.engine.num_threads} threads"
        self.device_label = ttk.Label(main_frame,
                                      text=f"Hardware: {dev}",
                                      style="Sub.TLabel")
        self.device_label.pack(pady=(0, 2))

        self.param_label = ttk.Label(main_frame,
                                     text="Parameters: —",
                                     style="Sub.TLabel")
        self.param_label.pack(pady=(0, 10))

        # ================================================================
        #  ⚙️  Architecture & Hyperparameters
        # ================================================================
        hp_frame = ttk.LabelFrame(main_frame,
                                  text="  ⚙️  Architecture & Hyperparameters  ",
                                  padding="12")
        hp_frame.pack(fill=tk.X, pady=(0, 8))

        grid = ttk.Frame(hp_frame)
        grid.pack(fill=tk.X)

        self.params = {
            "lr":         tk.StringVar(value="0.0003"),
            "epochs":     tk.StringVar(value="100"),
            "d_model":    tk.StringVar(value="128"),
            "d_ff":       tk.StringVar(value="256"),
            "n_heads":    tk.StringVar(value="4"),
            "n_layers":   tk.StringVar(value="4"),
            "seq_length": tk.StringVar(value="64"),
            "batch_size": tk.StringVar(value="32"),
            "dropout":    tk.StringVar(value="0.1"),
            "grad_clip":  tk.StringVar(value="1.0"),
        }

        labels = [
            ("Learning Rate:",       "lr"),
            ("Model Dim (D_Model):", "d_model"),
            ("Epochs:",              "epochs"),
            ("FF Dim (D_FF):",       "d_ff"),
            ("Heads (N_Heads):",     "n_heads"),
            ("Layers (N_Layers):",   "n_layers"),
            ("Context Window (Seq):","seq_length"),
            ("Batch Size:",          "batch_size"),
            ("Dropout:",             "dropout"),
            ("Grad Clip:",           "grad_clip"),
        ]

        for i, (text, key) in enumerate(labels):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(grid, text=text).grid(row=row, column=col,
                                            sticky=tk.W, padx=5, pady=3)
            ttk.Entry(grid, textvariable=self.params[key],
                      width=10).grid(row=row, column=col + 1,
                                      sticky=tk.W, padx=5, pady=3)

        # ================================================================
        #  🏋️  Training + 💾 Model Management
        # ================================================================
        train_frame = ttk.LabelFrame(main_frame,
                                     text="  🏋️  Training  /  💾 Model  ",
                                     padding="12")
        train_frame.pack(fill=tk.X, pady=8)

        # Row 1: file selection + model buttons
        top_row = ttk.Frame(train_frame)
        top_row.pack(fill=tk.X, pady=4)

        self.file_btn = ttk.Button(top_row, text="📂 Select .txt File",
                                   command=self.select_file)
        self.file_btn.pack(side=tk.LEFT, padx=4)

        self.save_btn = ttk.Button(top_row, text="💾 Save Model",
                                   command=self.save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=4)

        self.load_btn = ttk.Button(top_row, text="📂 Load Model",
                                   command=self.load_model)
        self.load_btn.pack(side=tk.LEFT, padx=4)

        self.file_label = ttk.Label(train_frame,
                                    text="No file selected",
                                    foreground="gray")
        self.file_label.pack(pady=2)

        # Row 2: start / stop
        btn_group = ttk.Frame(train_frame)
        btn_group.pack(pady=6)

        self.train_btn = ttk.Button(btn_group, text="🚀 Start Training",
                                    command=self.start_training,
                                    state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(btn_group, text="🛑 Stop",
                                   command=self.stop_training,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame,
                                            variable=self.progress_var,
                                            maximum=100, length=600)
        self.progress_bar.pack(pady=4)

        self.status_label = ttk.Label(train_frame,
                                      text="Status: Waiting for file…")
        self.status_label.pack(pady=(0, 6))

        # ================================================================
        #  ✨  Generation
        # ================================================================
        gen_frame = ttk.LabelFrame(main_frame,
                                   text="  ✨  Generation  ",
                                   padding="12")
        gen_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        # --- Generation settings row ---
        gen_settings = ttk.Frame(gen_frame)
        gen_settings.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(gen_settings, text="🌡️ Temperature:").grid(
            row=0, column=0, sticky=tk.W, padx=4)
        self.temp_var = tk.DoubleVar(value=0.8)
        self.temp_scale = ttk.Scale(gen_settings, from_=0.1, to=2.0,
                                    variable=self.temp_var, orient=tk.HORIZONTAL,
                                    length=160)
        self.temp_scale.grid(row=0, column=1, padx=4)
        self.temp_display = ttk.Label(gen_settings, text="0.80")
        self.temp_display.grid(row=0, column=2, padx=2)
        self.temp_var.trace_add("write", self._update_temp_display)

        ttk.Label(gen_settings, text="Top-K:").grid(
            row=0, column=3, sticky=tk.W, padx=(12, 4))
        self.topk_var = tk.StringVar(value="50")
        ttk.Entry(gen_settings, textvariable=self.topk_var,
                  width=6).grid(row=0, column=4, padx=4)

        ttk.Label(gen_settings, text="Top-P:").grid(
            row=0, column=5, sticky=tk.W, padx=(12, 4))
        self.topp_var = tk.StringVar(value="0.9")
        ttk.Entry(gen_settings, textvariable=self.topp_var,
                  width=6).grid(row=0, column=6, padx=4)

        # --- Seed + length ---
        ttk.Label(gen_frame, text="Seed phrase:").pack(anchor=tk.W)
        self.seed_entry = ttk.Entry(gen_frame, font=("Segoe UI", 11))
        self.seed_entry.pack(fill=tk.X, pady=4)
        self.seed_entry.insert(0, "The quick")

        len_row = ttk.Frame(gen_frame)
        len_row.pack(fill=tk.X, pady=2)
        ttk.Label(len_row, text="Length:").pack(side=tk.LEFT, padx=4)
        self.len_scale = ttk.Scale(len_row, from_=10, to=1000,
                                   orient=tk.HORIZONTAL)
        self.len_scale.set(100)
        self.len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.len_display = ttk.Label(len_row, text="100")
        self.len_display.pack(side=tk.LEFT, padx=4)
        self.len_scale.configure(command=self._update_len_display)

        self.gen_btn = ttk.Button(gen_frame, text="📝 Generate Text",
                                  command=self.generate_text,
                                  state=tk.DISABLED)
        self.gen_btn.pack(pady=8)

        self.result_text = tk.Text(gen_frame, height=10,
                                   font=("Consolas", 11), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    #  Callbacks
    # ==================================================================

    def _update_temp_display(self, *_):
        try:
            self.temp_display.config(text=f"{self.temp_var.get():.2f}")
        except tk.TclError:
            pass

    def _update_len_display(self, val):
        try:
            self.len_display.config(text=str(int(float(val))))
        except (ValueError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Training File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(
                text=f"Selected: {os.path.basename(file_path)}",
                foreground="black",
            )
            self.train_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Ready to train")

    # ------------------------------------------------------------------
    def update_progress(self, current, total, loss):
        percent = (current / total) * 100
        self.progress_var.set(percent)
        lr = self.engine.scheduler.get_lr() if self.engine.scheduler else 0
        self.status_label.config(
            text=f"Epoch {current}/{total}  |  Loss: {loss:.4f}  |  LR: {lr:.6f}"
        )
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    def stop_training(self):
        self.stop_event.set()
        self.status_label.config(text="Status: Stopping… 🛑")

    # ------------------------------------------------------------------
    def start_training(self):
        if not self.selected_file_path:
            return

        try:
            params = {
                "lr":          float(self.params["lr"].get()),
                "epochs":      int(self.params["epochs"].get()),
                "d_model":     int(self.params["d_model"].get()),
                "d_ff":        int(self.params["d_ff"].get()),
                "n_heads":     int(self.params["n_heads"].get()),
                "n_layers":    int(self.params["n_layers"].get()),
                "seq_length":  int(self.params["seq_length"].get()),
                "batch_size":  int(self.params["batch_size"].get()),
                "dropout":     float(self.params["dropout"].get()),
                "grad_clip":   float(self.params["grad_clip"].get()),
            }
        except ValueError:
            messagebox.showerror("Params Error",
                                 "Please enter valid numbers in all fields!")
            return

        try:
            with open(self.selected_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")
            return

        self.stop_event.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def run():
            try:
                self.engine.train(
                    text, params,
                    progress_callback=self.update_progress,
                    stop_event=self.stop_event,
                )
                if self.stop_event.is_set():
                    self.root.after(0, lambda: self.status_label.config(
                        text="Status: Stopped. 🛑"))
                else:
                    self.is_trained = True
                    self.root.after(0, lambda: self.status_label.config(
                        text="Status: Training complete! ✅"))
                    self.root.after(0, lambda: self.gen_btn.config(
                        state=tk.NORMAL))
                    self.root.after(0, lambda: self.save_btn.config(
                        state=tk.NORMAL))
                # Show parameter count
                if self.engine.model is not None:
                    n = self.engine.model.count_parameters()
                    self.root.after(0, lambda c=n: self.param_label.config(
                        text=f"Parameters: {c:,}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Train Error", f"Error during training:\n{e}"))
            finally:
                self.root.after(0, self._reset_train_buttons)

        threading.Thread(target=run, daemon=True).start()

    def _reset_train_buttons(self):
        self.train_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def generate_text(self):
        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
        except ValueError:
            messagebox.showwarning("Warning",
                                   "Invalid generation settings!")
            return

        if not seed:
            messagebox.showwarning("Warning",
                                   "Please enter a seed phrase.")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Generating… please wait…\n")

        def run():
            try:
                res = self.engine.generate(seed, length,
                                           temperature, top_k, top_p)
                self.root.after(0, self._display_result, res)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{e}"))

        threading.Thread(target=run, daemon=True).start()

    def _display_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    # ------------------------------------------------------------------
    def save_model(self):
        path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.save_model(path)
            messagebox.showinfo("Saved",
                                f"Model saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.load_model(path)
            self.is_trained = True
            self.gen_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            n = self.engine.model.count_parameters()
            self.param_label.config(text=f"Parameters: {n:,}")
            self.status_label.config(
                text=f"Status: Model loaded ✅  ({os.path.basename(path)})")
            # Fill GUI fields from stored params
            p = self.engine.params_used
            for key in ("lr", "epochs", "d_model", "d_ff", "n_heads",
                        "n_layers", "seq_length", "batch_size", "dropout",
                        "grad_clip"):
                if key in p:
                    self.params[key].set(str(p[key]))
        except Exception as e:
            messagebox.showerror("Load Error", str(e))


# ======================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()
