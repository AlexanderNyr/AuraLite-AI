"""gui/tabs/console.py — ConsoleTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from ..common import (filedialog, messagebox, tk, ttk)

class ConsoleTabMixin:
    def _build_console_tab(self):
        tab = self.tab_console

        # Toolbar: clear + copy + autoscroll toggle
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(toolbar, text="🖥️ Live stdout / stderr from the engine",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="🗑 Clear",
                   command=self._clear_console).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="📋 Copy All",
                   command=self._copy_console).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="💾 Save to file…",
                   command=self._save_console).pack(side=tk.RIGHT, padx=2)

        # Output area with scrollbar
        out_frame = ttk.LabelFrame(tab, text="  📜  Output  ", padding="4")
        out_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(out_frame, orient=tk.VERTICAL)
        self.console_text = tk.Text(
            out_frame,
            font=("Consolas", 10),
            bg="#1e1e1e", fg="#dcdcdc",
            insertbackground="#dcdcdc",
            wrap=tk.NONE,
            state=tk.DISABLED,
            yscrollcommand=ysb.set,
        )
        ysb.config(command=self.console_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Colour tags ----------------------------------------------
        # VS-Code-ish palette on a dark background.
        self.console_text.tag_configure("error",  foreground="#f48771",
                                        font=("Consolas", 10, "bold"))
        self.console_text.tag_configure("warn",   foreground="#dcdcaa")
        self.console_text.tag_configure("info",   foreground="#9cdcfe")
        self.console_text.tag_configure("ok",     foreground="#6a9955",
                                        font=("Consolas", 10, "bold"))
        self.console_text.tag_configure("engine", foreground="#c586c0")
        self.console_text.tag_configure("epoch",  foreground="#4ec9b0")

        # Legend strip
        legend = ttk.Frame(tab)
        legend.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(legend, text="Legend:", style="Sub.TLabel").pack(side=tk.LEFT, padx=4)
        for txt, color in [
            ("[AuraLite]", "#c586c0"),
            ("epoch",      "#4ec9b0"),
            ("INFO",       "#9cdcfe"),
            ("WARNING",    "#dcdcaa"),
            ("ERROR",      "#f48771"),
            ("✅ done",     "#6a9955"),
        ]:
            tk.Label(legend, text=txt, fg=color, bg="#f5f6f7",
                     font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=6)


    def _clear_console(self):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.config(state=tk.DISABLED)


    def _copy_console(self):
        try:
            text = self.console_text.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Status: Console copied to clipboard ✅")
        except tk.TclError:
            pass


    def _save_console(self):
        path = filedialog.asksaveasfilename(
            title="Save console log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.console_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"Console log saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

