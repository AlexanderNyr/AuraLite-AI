import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from model_engine import AuraLiteEngine
import threading
import os

class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraLite AI - CUDA Powered")
        self.root.geometry("700x750")
        self.root.configure(bg="#f5f6f7")

        self.engine = AuraLiteEngine()
        self.is_trained = False
        self.selected_file_path = None
        self.stop_event = threading.Event()

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10), background="#f5f6f7")
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), background="#f5f6f7")

        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_frame, text="🌟 AuraLite AI - CUDA Edition", style="Header.TLabel")
        header.pack(pady=(0, 10))
        
        device_text = "GPU: CUDA" if self.engine.device.type == 'cuda' else "CPU: Only"
        device_label = ttk.Label(main_frame, text=f"Hardware Acceleration: {device_text}", font=("Segoe UI", 9, "italic"))
        device_label.pack(pady=(0, 20))

        settings_frame = ttk.LabelFrame(main_frame, text=" ⚙️ Hyperparameters ", padding="15")
        settings_frame.pack(fill=tk.X, pady=5)

        params_grid = ttk.Frame(settings_frame)
        params_grid.pack(fill=tk.X)

        self.params = {
            'lr': tk.StringVar(value="0.001"),
            'epochs': tk.StringVar(value="300"),
            'd_model': tk.StringVar(value="64"),
            'd_ff': tk.StringVar(value="128"),
            'seq_length': tk.StringVar(value="16"),
        }

        labels = [
            ("Learning Rate:", 'lr'),
            ("Epochs:", 'epochs'),
            ("Model Dim (D_Model):", 'd_model'),
            ("FF Dim (D_FF):", 'd_ff'),
            ("Context Window (Seq):", 'seq_length'),
        ]

        for i, (text, key) in enumerate(labels):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(params_grid, text=text).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_grid, textvariable=self.params[key], width=10).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)

        train_frame = ttk.LabelFrame(main_frame, text=" 🏋️ Training ", padding="15")
        train_frame.pack(fill=tk.X, pady=10)

        self.file_btn = ttk.Button(train_frame, text="📂 Select .txt File", command=self.select_file)
        self.file_btn.pack(pady=5)

        self.file_label = ttk.Label(train_frame, text="No file selected", foreground="gray")
        self.file_label.pack(pady=2)

        btn_group = ttk.Frame(train_frame)
        btn_group.pack(pady=10)

        self.train_btn = ttk.Button(btn_group, text="🚀 Start Training", command=self.start_training, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_group, text="🛑 Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var, maximum=100, length=500)
        self.progress_bar.pack(pady=5)

        self.status_label = ttk.Label(train_frame, text="Status: Waiting for file...")
        self.status_label.pack(pady=(0, 10))

        gen_frame = ttk.LabelFrame(main_frame, text=" ✨ Generation ", padding="15")
        gen_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(gen_frame, text="Seed phrase:").pack(anchor=tk.W)
        self.seed_entry = ttk.Entry(gen_frame, font=("Segoe UI", 11))
        self.seed_entry.pack(fill=tk.X, pady=5)
        self.seed_entry.insert(0, "The quick")

        ttk.Label(gen_frame, text="Length:").pack(anchor=tk.W)
        self.len_scale = ttk.Scale(gen_frame, from_=10, to=1000, orient=tk.HORIZONTAL)
        self.len_scale.set(50)
        self.len_scale.pack(fill=tk.X, pady=5)

        self.gen_btn = ttk.Button(gen_frame, text="📝 Generate Text", command=self.generate_text, state=tk.DISABLED)
        self.gen_btn.pack(pady=10)

        self.result_text = tk.Text(gen_frame, height=12, font=("Consolas", 11), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Training File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(text=f"Selected: {os.path.basename(file_path)}", foreground="black")
            self.train_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Ready to train")

    def update_progress(self, current, total, loss):
        percent = (current / total) * 100
        self.progress_var.set(percent)
        self.status_label.config(text=f"Epoch {current}/{total} | Loss: {loss:.4f}")
        self.root.update_idletasks()

    def stop_training(self):
        self.stop_event.set()
        self.status_label.config(text="Status: Stopping... 🛑")

    def start_training(self):
        if not self.selected_file_path:
            return

        try:
            params = {
                'lr': float(self.params['lr'].get()),
                'epochs': int(self.params['epochs'].get()),
                'd_model': int(self.params['d_model'].get()),
                'd_ff': int(self.params['d_ff'].get()),
                'seq_length': int(self.params['seq_length'].get()),
            }
        except ValueError:
            messagebox.showerror("Params Error", "Please enter valid numbers in settings!")
            return

        try:
            with open(self.selected_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file: {e}")
            return

        self.stop_event.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        def run():
            try:
                self.engine.train(text, params, progress_callback=self.update_progress, stop_event=self.stop_event)
                if self.stop_event.is_set():
                    self.root.after(0, lambda: self.status_label.config(text="Status: Stopped. 🛑"))
                else:
                    self.is_trained = True
                    self.root.after(0, lambda: self.status_label.config(text="Status: Training complete! ✅"))
                    self.root.after(0, lambda: self.gen_btn.config(state=tk.NORMAL))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Train Error", f"Error during training: {e}"))
            finally:
                self.root.after(0, self.reset_train_buttons)

        threading.Thread(target=run, daemon=True).start()

    def reset_train_buttons(self):
        self.train_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def generate_text(self):
        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        
        if not seed:
            messagebox.showwarning("Warning", "Please enter a seed phrase")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Generating... please wait...\n")
        
        def run():
            try:
                res = self.engine.generate(seed, length)
                self.root.after(0, self.display_result, res)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Gen Error", f"Error during generation: {e}"))

        threading.Thread(target=run, daemon=True).start()

    def display_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()
