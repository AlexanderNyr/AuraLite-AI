"""gui/tabs/generation.py — GenerationTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from ..common import (build_web_context, messagebox, threading, tk, ttk)

class GenerationTabMixin:
    def _build_generation_tab(self):
        tab = self.tab_gen

        # --- Sampling settings ---
        gen_settings = ttk.LabelFrame(tab, text="  🎛️  Sampling  ", padding="10")
        gen_settings.pack(fill=tk.X, pady=(0, 8))

        srow = ttk.Frame(gen_settings)
        srow.pack(fill=tk.X)

        ttk.Label(srow, text="🌡️ Temperature:").grid(row=0, column=0,
                                                     sticky=tk.W, padx=4)
        self.temp_var = tk.DoubleVar(value=0.8)
        self.temp_scale = ttk.Scale(srow, from_=0.1, to=2.0,
                                    variable=self.temp_var,
                                    orient=tk.HORIZONTAL, length=150)
        self.temp_scale.grid(row=0, column=1, padx=4)
        self.temp_display = ttk.Label(srow, text="0.80")
        self.temp_display.grid(row=0, column=2, padx=2)
        self.temp_var.trace_add("write", self._update_temp_display)

        ttk.Label(srow, text="Top-K:").grid(row=0, column=3,
                                            sticky=tk.W, padx=(14, 4))
        self.topk_var = tk.StringVar(value="50")
        ttk.Entry(srow, textvariable=self.topk_var,
                  width=6).grid(row=0, column=4, padx=4)

        ttk.Label(srow, text="Top-P:").grid(row=0, column=5,
                                            sticky=tk.W, padx=(14, 4))
        self.topp_var = tk.StringVar(value="0.9")
        ttk.Entry(srow, textvariable=self.topp_var,
                  width=6).grid(row=0, column=6, padx=4)

        ttk.Label(srow, text="Rep. Penalty:").grid(row=1, column=0,
                                                   sticky=tk.W, padx=4,
                                                   pady=(6, 0))
        self.rep_var = tk.StringVar(value="1.1")
        ttk.Entry(srow, textvariable=self.rep_var,
                  width=6).grid(row=1, column=1, sticky=tk.W, padx=4,
                                pady=(6, 0))
        ttk.Label(srow, text="Min-P:").grid(row=1, column=2,
                                            sticky=tk.W, padx=(14, 4),
                                            pady=(6, 0))
        self.minp_var = tk.StringVar(value="0.0")
        ttk.Entry(srow, textvariable=self.minp_var,
                  width=6).grid(row=1, column=3, sticky=tk.W, padx=4,
                                pady=(6, 0))
        ttk.Label(srow, text="(1.0 = off, 1.1–1.3 fights loops)",
                  style="Sub.TLabel").grid(row=1, column=4, columnspan=3,
                                           sticky=tk.W, pady=(6, 0))

        # 2.11: generation window (applies to the Chat tab as well)
        ctx_row = ttk.Frame(gen_settings)
        ctx_row.pack(fill=tk.X, pady=(6, 0))
        self.full_context_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctx_row, text="Full context (no window)",
                        variable=self.full_context_var,
                        command=self._update_context_hint).pack(side=tk.LEFT, padx=4)
        self.context_hint = ttk.Label(ctx_row, text="", style="Sub.TLabel")
        self.context_hint.pack(side=tk.LEFT, padx=8)
        self._update_context_hint()

        # --- Seed + length ---
        seed_frame = ttk.LabelFrame(tab, text="  🌱  Prompt  ", padding="10")
        seed_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(seed_frame, text="Seed phrase:").pack(anchor=tk.W)
        self.seed_entry = ttk.Entry(seed_frame, font=("Segoe UI", 11))
        self.seed_entry.pack(fill=tk.X, pady=4)
        self.seed_entry.insert(0, "The quick")

        len_row = ttk.Frame(seed_frame)
        len_row.pack(fill=tk.X, pady=2)
        ttk.Label(len_row, text="Length (tokens):").pack(side=tk.LEFT, padx=4)
        self.len_scale = ttk.Scale(len_row, from_=10, to=1000,
                                   orient=tk.HORIZONTAL)
        self.len_scale.set(100)
        self.len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.len_display = ttk.Label(len_row, text="100", width=5)
        self.len_display.pack(side=tk.LEFT, padx=4)
        self.len_scale.configure(command=self._update_len_display)

        ttk.Button(len_row, text="🎯 Auto",
                   command=self._auto_gen_length, width=8).pack(side=tk.LEFT, padx=4)

        # NEW: Batch generation
        self.batch_row = ttk.Frame(seed_frame)
        self.batch_row.pack(fill=tk.X, pady=2)
        self.batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.batch_row, text="Batch mode (multiple prompts)",
                        variable=self.batch_var).pack(side=tk.LEFT, padx=4)
        self.batch_entry = ttk.Entry(self.batch_row, font=("Segoe UI", 10))
        self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.batch_entry.insert(0, "Prompt 1 | Prompt 2 | Prompt 3")
        self.batch_entry.pack_forget()  # Hidden by default

        self.gen_btn = ttk.Button(seed_frame, text="📝 Generate Text",
                                  command=self.generate_text,
                                  state=tk.DISABLED)
        self.gen_btn.pack(pady=6)

        self.stream_var = tk.BooleanVar(value=False)
        self.stream_cb = ttk.Checkbutton(seed_frame,
            text="Streaming output (token-by-token)",
            variable=self.stream_var)
        self.stream_cb.pack(pady=2)

        # Chat streaming toggle (in generation tab for reference)
        self.chat_stream_var = tk.BooleanVar(value=True)

        # NEW: Thinking mode + Web search
        smart_row = ttk.Frame(seed_frame)
        smart_row.pack(fill=tk.X, pady=2)

        self.thinking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(smart_row,
            text="🧠 Thinking mode (two-pass draft → answer)",
            variable=self.thinking_var).pack(side=tk.LEFT, padx=4)

        self.websearch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(smart_row,
            text="🌐 Web search (DuckDuckGo)",
            variable=self.websearch_var,
            command=self._toggle_web_query).pack(side=tk.LEFT, padx=12)

        self.web_query_row = ttk.Frame(seed_frame)
        ttk.Label(self.web_query_row,
                  text="Search query (empty = use seed):").pack(
            side=tk.LEFT, padx=4)
        self.web_query_entry = ttk.Entry(self.web_query_row,
                                         font=("Segoe UI", 10))
        self.web_query_entry.pack(side=tk.LEFT, fill=tk.X,
                                  expand=True, padx=4)

        # --- Output ---
        out_frame = ttk.LabelFrame(tab, text="  📄  Output  ", padding="6")
        out_frame.pack(fill=tk.BOTH, expand=True)

        # Toolbar above the result widget
        out_toolbar = ttk.Frame(out_frame)
        out_toolbar.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(out_toolbar, text="Generated text",
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=2)

        ttk.Button(out_toolbar, text="🗑 Clear",
                   command=self._clear_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="💾 Save…",
                   command=self._save_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="➕ Append to file…",
                   command=self._append_result).pack(side=tk.RIGHT, padx=2)
        ttk.Button(out_toolbar, text="📋 Copy",
                   command=self._copy_result).pack(side=tk.RIGHT, padx=2)

        # Text + scrollbar
        text_wrap = ttk.Frame(out_frame)
        text_wrap.pack(fill=tk.BOTH, expand=True)

        res_ysb = ttk.Scrollbar(text_wrap, orient=tk.VERTICAL)
        self.result_text = tk.Text(text_wrap, height=10,
                                   font=("Consolas", 11), wrap=tk.WORD,
                                   yscrollcommand=res_ysb.set,
                                   undo=True)
        res_ysb.config(command=self.result_text.yview)
        res_ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right-click context menu on the result widget
        self._result_menu = tk.Menu(self.root, tearoff=0)
        self._result_menu.add_command(label="Copy selection",
                                      command=self._copy_selection_result)
        self._result_menu.add_command(label="Copy all",
                                      command=self._copy_result)
        self._result_menu.add_separator()
        self._result_menu.add_command(label="Save to file…",
                                      command=self._save_result)
        self._result_menu.add_command(label="Append to file…",
                                      command=self._append_result)
        self._result_menu.add_separator()
        self._result_menu.add_command(label="Clear",
                                      command=self._clear_result)
        # bind right click (Button-3 on Linux/Win, Button-2 on macOS)
        self.result_text.bind("<Button-3>", self._show_result_menu)
        self.result_text.bind("<Button-2>", self._show_result_menu)
        # Ctrl/Cmd+S to save
        self.result_text.bind("<Control-s>", lambda e: (self._save_result(), "break"))
        self.result_text.bind("<Command-s>", lambda e: (self._save_result(), "break"))


    def _toggle_batch_entry(self, *_):
        """Show / hide the batch entry based on the batch mode checkbox."""
        if self.batch_var.get():
            self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        else:
            self.batch_entry.pack_forget()


    def _toggle_web_query(self, *_):
        """Show / hide the web-search query row."""
        if self.websearch_var.get():
            self.web_query_row.pack(fill=tk.X, pady=2)
        else:
            self.web_query_row.pack_forget()


    def _fetch_web_context(self, seed: str) -> str:
        """Run a web search (blocking — call from a worker thread).

        Returns formatted snippet context, or '' on failure / no results.
        """
        query = self.web_query_entry.get().strip() or seed
        try:
            ctx = build_web_context(query, max_results=4)
            if ctx:
                print(f"🌐 Web search OK: {len(ctx)} chars of context "
                      f"for query '{query}'")
            else:
                print(f"🌐 Web search: no results for '{query}'")
            return ctx
        except Exception as e:
            print(f"⚠️ Web search failed ({e}) — generating without it.")
            return ""


    def _update_context_hint(self, *_):
        """Explain what the window checkbox does for the loaded model."""
        model = getattr(self.engine, "model", None)
        trained = getattr(model, "max_seq_len", None)
        if self.full_context_var.get():
            text = ("whole history: past the trained length "
                    f"({trained} tokens) the output degrades" if trained else
                    "whole history: past the trained length the output degrades")
        else:
            text = (f"each token sees the newest {trained} tokens (the trained length)"
                    if trained else "each token sees as many tokens as the model was trained on")
        self.context_hint.config(text=text)

    def _apply_context_choice(self):
        """Hand the checkbox to the engine before a generation starts (2.11).

        Applied here, between generations, rather than on click: changing the
        window drops the KV cache, which must not happen under a running
        generation.
        """
        var = getattr(self, "full_context_var", None)
        if var is None:
            return
        try:
            self.engine.set_context_window("full" if var.get() else "auto")
        except Exception as err:  # an engine without the 2.11 API
            print(f"[AuraLite] context window not applied: {err}")
        self._update_context_hint()

    def generate_text(self):
        self._apply_context_choice()
        # Handle batch mode
        if self.batch_var.get():
            self._generate_batch()
            return

        # Thinking / web-search mode has its own pipeline
        if self.thinking_var.get() or self.websearch_var.get():
            self._generate_thinking()
            return

        # Handle streaming mode
        if self.stream_var.get():
            self._generate_streaming()
            return

        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
            min_p = float(self.minp_var.get())
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
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                res = self.engine.generate(seed, length,
                                           temperature, top_k, top_p,
                                           repetition_penalty=rep_pen,
                                           min_p=min_p)
                self.root.after(0, self._display_result, res)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{err}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _generate_thinking(self):
        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
            min_p = float(self.minp_var.get())
        except ValueError:
            messagebox.showwarning("Warning",
                                   "Invalid generation settings!")
            return

        if not seed:
            messagebox.showwarning("Warning",
                                   "Please enter a seed phrase.")
            return

        use_web = self.websearch_var.get()
        use_thinking = self.thinking_var.get()

        self.result_text.delete("1.0", tk.END)
        steps = []
        if use_web:
            steps.append("🌐 searching the web")
        if use_thinking:
            steps.append("🧠 thinking")
        steps.append("✍️ generating")
        self.result_text.insert(tk.END, " → ".join(steps) + " …\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                web_ctx = self._fetch_web_context(seed) if use_web else ""

                if use_thinking:
                    thoughts, final = self.engine.generate_with_thinking(
                        seed, length, temperature, top_k, top_p,
                        repetition_penalty=rep_pen,
                        web_context=web_ctx or None, min_p=min_p)
                else:
                    # Web search only: prepend context, generate once
                    prompt = f"{web_ctx}\n{seed}" if web_ctx else seed
                    full = self.engine.generate(
                        prompt, length, temperature, top_k, top_p,
                        repetition_penalty=rep_pen, min_p=min_p)
                    thoughts = ""
                    final = seed + full[len(prompt):]

                parts = []
                if web_ctx:
                    parts.append("🌐 WEB CONTEXT\n──────────────\n"
                                 + web_ctx)
                if thoughts:
                    parts.append("🧠 THINKING (draft pass)\n"
                                 "──────────────\n" + thoughts)
                parts.append("✅ ANSWER\n──────────────\n" + final)
                self.root.after(0, self._display_result,
                                "\n\n".join(parts))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{err}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _generate_streaming(self):
        seed = self.seed_entry.get()
        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
            min_p = float(self.minp_var.get())
        except ValueError:
            messagebox.showwarning("Warning",
                                   "Invalid generation settings!")
            return

        if not seed:
            messagebox.showwarning("Warning",
                                   "Please enter a seed phrase.")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Generating…\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                for token_text in self.engine.generate_streaming(
                    seed, length, temperature, top_k, top_p,
                    repetition_penalty=rep_pen, min_p=min_p
                ):
                    # tkinter is NOT thread-safe: marshal the widget update
                    # onto the main loop via root.after instead of calling
                    # root.update() directly from this worker thread.
                    self.root.after(0, lambda t=token_text:
                        (self.result_text.insert(tk.END, t),
                         self.result_text.see(tk.END)))
                self.root.after(0, lambda: self.result_text.insert(
                    tk.END, "\n\n✅ Generation complete."))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Gen Error", f"Error during generation:\n{err}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()


    def _generate_batch(self):
        batch_text = self.batch_entry.get()
        if not batch_text.strip():
            messagebox.showwarning("Warning", "Please enter prompts separated by '|'")
            return

        prompts = [p.strip() for p in batch_text.split("|") if p.strip()]
        if not prompts:
            messagebox.showwarning("Warning", "No valid prompts found")
            return

        length = int(self.len_scale.get())
        try:
            temperature = float(self.temp_var.get())
            top_k = int(self.topk_var.get())
            top_p = float(self.topp_var.get())
            rep_pen = float(self.rep_var.get())
            min_p = float(self.minp_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Invalid generation settings!")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Generating {len(prompts)} prompts in batch…\n")
        self.gen_btn.config(state=tk.DISABLED)

        def run():
            try:
                results = self.engine.generate_batch(
                    prompts, length, temperature, top_k, top_p,
                    repetition_penalty=rep_pen, min_p=min_p
                )
                for i, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
                    self.root.after(0, lambda p=prompt, r=result, idx=i:
                        self.result_text.insert(tk.END,
                            f"\n{'='*40}\nPrompt {idx+1}: {p}\n{'='*40}\n{r}\n"))
                self.root.after(0, lambda: self.result_text.insert(
                    tk.END, f"\n✅ Batch generation complete ({len(prompts)} prompts)"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror(
                    "Gen Error", f"Error during batch generation:\n{err}"))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(
                    state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

