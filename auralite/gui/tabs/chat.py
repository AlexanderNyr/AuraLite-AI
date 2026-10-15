"""gui/tabs/chat.py — ChatTabMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from .. import common as gui_common
from .. import visualizations
from ..i18n import tr
from ..common import (AuraLiteAgent, CHAT_TEMPLATES, Figure, FigureCanvasTkAgg, HAS_CHAT_SUPPORT, Sandbox, TOOL_REGISTRY, messagebox, threading, tk, ttk)

class ChatTabMixin:
    def _build_chat_tab(self):
        tab = self.tab_chat

        # Declare agent-mode state BEFORE it is used by any widget below.
        self.agent_mode_var = tk.BooleanVar(value=False)

        if not HAS_CHAT_SUPPORT:
            ttk.Label(tab,
                      text="⚠ Chat interface not available (chat_interface.py missing).",
                      font=("Segoe UI", 12, "bold")).pack(pady=40)
            return

        # ---- Chat Template Selection ----
        template_frame = ttk.LabelFrame(tab, text="  💬  Chat Template  ", padding="8")
        template_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(template_frame, text="Template:").pack(side=tk.LEFT, padx=4)
        self.chat_template_var = tk.StringVar(value="chatml")
        templates = list(CHAT_TEMPLATES.keys()) if 'CHAT_TEMPLATES' in globals() else ["chatml", "llama2", "mistral", "simple"]
        self.chat_template_combo = ttk.Combobox(
            template_frame, textvariable=self.chat_template_var,
            values=templates, state="readonly", width=18
        )
        self.chat_template_combo.pack(side=tk.LEFT, padx=4)

        self.chat_system_var = tk.StringVar(value="You are a helpful assistant.")
        ttk.Label(template_frame, text="System prompt:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(template_frame, textvariable=self.chat_system_var,
                  width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # ---- Chat History ----
        history_frame = ttk.LabelFrame(tab, text="  📜  Conversation  ", padding="8")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Chat display
        chat_display_frame = ttk.Frame(history_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True)

        ysb = ttk.Scrollbar(chat_display_frame, orient=tk.VERTICAL)
        self.chat_text = tk.Text(
            chat_display_frame,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            yscrollcommand=ysb.set,
            state=tk.DISABLED,
            bg="#f8f9fa",
        )
        ysb.config(command=self.chat_text.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Color tags
        self.chat_text.tag_configure("user", foreground="#0066cc", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("assistant", foreground="#228B22")
        self.chat_text.tag_configure("system", foreground="#666666", font=("Segoe UI", 10, "italic"))

        # Input area
        input_frame = ttk.Frame(history_frame)
        input_frame.pack(fill=tk.X, pady=(8, 0))

        self.chat_input = ttk.Entry(input_frame, font=("Segoe UI", 11))
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.chat_input.bind("<Return>", lambda e: self._send_chat_message())

        self.chat_send_btn = ttk.Button(input_frame, text="Send", command=self._send_chat_message)
        self.chat_send_btn.pack(side=tk.LEFT, padx=2)

        self.chat_stream_var = tk.BooleanVar(value=True)
        self.chat_stream_cb = ttk.Checkbutton(input_frame, text="Stream",
                                              variable=self.chat_stream_var)
        self.chat_stream_cb.pack(side=tk.LEFT, padx=4)

        self.chat_clear_btn = ttk.Button(input_frame, text="Clear History", command=self._clear_chat_history)
        self.chat_clear_btn.pack(side=tk.LEFT, padx=2)

        # ── Agent Mode ──────────────────────────────────────────────────────
        agent_frame = ttk.LabelFrame(tab, text="  🤖  Agent Mode (Tool-Calling Sandbox)  ", padding="6")
        agent_frame.pack(fill=tk.X, pady=(4, 0))

        # Left column: enable toggle + tool list
        agent_left = ttk.Frame(agent_frame)
        agent_left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.agent_mode_cb = ttk.Checkbutton(
            agent_left, text="Enable Agent Mode",
            variable=self.agent_mode_var,
            command=self._on_agent_mode_toggle,
        )
        self.agent_mode_cb.pack(anchor=tk.W)

        if gui_common.HAS_AGENT:
            tool_names = list(TOOL_REGISTRY.keys())
            ttk.Label(agent_left, text="Available tools:", font=("Segoe UI", 9, "italic")).pack(anchor=tk.W)
            for tn in tool_names:
                ttk.Label(agent_left, text=f"  • {tn}", font=("Segoe UI", 9)).pack(anchor=tk.W)
        else:
            ttk.Label(agent_left, text="⚠ agent/ package not found.",
                      foreground="red", font=("Segoe UI", 9)).pack(anchor=tk.W)

        # Right column: sandbox log
        agent_right = ttk.Frame(agent_frame)
        agent_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(agent_right, text="Sandbox log:", font=("Segoe UI", 9, "italic")).pack(anchor=tk.W)
        sb_scroll = ttk.Scrollbar(agent_right, orient=tk.VERTICAL)
        self.agent_sandbox_log = tk.Text(
            agent_right, height=5, font=("Consolas", 9),
            bg="#1e1e1e", fg="#d4d4d4",
            state=tk.DISABLED, wrap=tk.WORD,
            yscrollcommand=sb_scroll.set,
        )
        sb_scroll.config(command=self.agent_sandbox_log.yview)
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.agent_sandbox_log.pack(fill=tk.BOTH, expand=True)
        self.agent_sandbox_log.tag_configure("ok",    foreground="#6a9955")
        self.agent_sandbox_log.tag_configure("error", foreground="#f44747")
        self.agent_sandbox_log.tag_configure("info",  foreground="#9cdcfe")
        self.agent_sandbox_log.tag_configure("tool",  foreground="#dcdcaa")

        # Stop agent button
        self.agent_stop_btn = ttk.Button(
            agent_right, text="⏹ Stop Agent",
            command=self._stop_agent, state=tk.DISABLED,
        )
        self.agent_stop_btn.pack(anchor=tk.E, pady=(2, 0))

        # ── Insights: token distribution + attention map (v2.6.5) ───────────
        self._build_chat_insights(tab)

        # Status
        self.chat_status = ttk.Label(tab, text="Chat ready. Load a model to start.", style="Sub.TLabel")
        self.chat_status.pack(anchor=tk.W, pady=4)

        # Initialize chat history
        self.chat_history = []

        # Agent state
        self._agent: "AuraLiteAgent | None" = None
        self._agent_sandbox: "Sandbox | None" = None
        self._agent_stop_event = threading.Event()



    # ── Insights panel (token distribution / attention map) ─────────────
    def _build_chat_insights(self, tab):
        """Bar chart of token frequencies + attention heatmap for the last turn.

        Both plots need `matplotlib` (optional dependency); without it the panel
        explains how to enable it instead of failing at first click.
        """
        frame = ttk.LabelFrame(tab, text="  📊  Insights  ", padding="6")
        frame.pack(fill=tk.X, pady=(4, 0))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)
        self.chat_insights_btn = ttk.Button(controls, text=tr("btn.insights"),
                                            command=self._plot_token_distribution)
        self.chat_insights_btn.pack(side=tk.LEFT, padx=2)
        self.chat_attention_btn = ttk.Button(controls, text=tr("btn.attention"),
                                             command=self._plot_attention_map)
        self.chat_attention_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(controls, text=tr("lbl.attention_layer")).pack(side=tk.LEFT, padx=(12, 2))
        self.chat_attention_layer_var = tk.StringVar(
            value=str(getattr(self.gui_config, "attention_layer", 0)))
        ttk.Entry(controls, textvariable=self.chat_attention_layer_var,
                  width=4).pack(side=tk.LEFT)

        self.chat_plot_ax = None
        if gui_common.HAS_MATPLOTLIB:
            self.chat_fig = Figure(figsize=(6, 2.6), dpi=80)
            self.chat_plot_ax = self.chat_fig.add_subplot(111)
            self.chat_plot_canvas = FigureCanvasTkAgg(self.chat_fig, master=frame)
            self.chat_plot_canvas.get_tk_widget().pack(fill=tk.X, expand=True)
            self.chat_plot_canvas.draw()
        else:
            ttk.Label(frame, text=tr("msg.plots_unavailable"),
                      style="Sub.TLabel").pack(anchor=tk.W)

    def _last_response_text(self) -> str:
        for message in reversed(getattr(self, "chat_history", []) or []):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message["content"])
        return ""

    def _last_exchange_text(self) -> str:
        parts = [m.get("content", "") for m in getattr(self, "chat_history", []) or []
                 if m.get("role") in ("user", "assistant") and m.get("content")]
        return "\n".join(parts[-2:])

    def _plot_token_distribution(self):
        """Bar chart of the token frequencies of the last assistant reply."""
        if not self._ensure_native_model(need_tokens=True):
            return
        text = self._last_response_text()
        if not text.strip():
            messagebox.showinfo(tr("btn.insights"), tr("msg.empty_output"))
            return
        try:
            pairs = self.engine.token_frequencies(text)
        except Exception as exc:
            messagebox.showwarning(tr("btn.insights"), str(exc))
            return
        visualizations.plot_token_distribution(
            self.chat_plot_ax, pairs,
            top=int(getattr(self.gui_config, "plot_top_tokens", 20)),
            title=tr("msg.tokens_title"))
        self._refresh_chat_plot()
        self.chat_status.config(text=f"{tr('lbl.tokens')}: {len(pairs)} unique / {len(text)} chars")

    def _plot_attention_map(self):
        """Heatmap of this layer's attention over the last user+assistant turn."""
        if not self._ensure_native_model():
            return
        text = self._last_exchange_text()
        if not text.strip():
            messagebox.showinfo(tr("btn.attention"), tr("msg.empty_output"))
            return
        try:
            layer = int(self.chat_attention_layer_var.get())
        except (TypeError, ValueError):
            layer = 0
        try:
            matrix = self.engine.attention_maps(text, layer=layer)
            ids = self.engine.encode(text)[-matrix.shape[-1]:]
            tokens = [self.engine.decode([i]) for i in ids]
        except Exception as exc:
            messagebox.showwarning(tr("btn.attention"), str(exc))
            return
        visualizations.plot_attention_heatmap(
            self.chat_plot_ax, matrix, tokens=tokens,
            title=tr("msg.attention_title", layer=layer))
        self._refresh_chat_plot()
        self.gui_config = self.gui_config.update(attention_layer=layer)

    def _ensure_native_model(self, need_tokens: bool = False) -> bool:
        """Shared guard for the insight plots."""
        if not getattr(self.engine, "model", None):
            messagebox.showwarning(tr("lbl.status"), tr("msg.no_model"))
            return False
        if self.chat_plot_ax is None:
            messagebox.showinfo(tr("btn.insights"), tr("msg.plots_unavailable"))
            return False
        return True

    def _refresh_chat_plot(self):
        canvas = getattr(self, "chat_plot_canvas", None)
        if canvas is None:
            return
        try:
            self.chat_fig.tight_layout()
        except Exception:
            pass
        canvas.draw()

    def _send_chat_message(self):
        if not self.engine.model:
            messagebox.showwarning("No Model", "Please load or train a model first.")
            return

        user_msg = self.chat_input.get().strip()
        if not user_msg:
            return

        self.chat_input.delete(0, tk.END)

        # Route to agent if agent mode is on
        if self.agent_mode_var.get() and gui_common.HAS_AGENT:
            self._send_agent_message(user_msg)
            return

        # 2.11: the generation tab's window checkbox applies to chat too
        apply_context = getattr(self, "_apply_context_choice", None)
        if apply_context is not None:
            apply_context()

        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_msg})

        # Update display
        self._append_chat_message("user", user_msg)

        # Generate response
        self.chat_send_btn.config(state=tk.DISABLED)
        self.chat_status.config(text="Generating response...")

        use_stream = bool(getattr(self, 'chat_stream_var', None) and self.chat_stream_var.get())

        def run():
            try:
                template = self.chat_template_var.get()
                system_prompt = self.chat_system_var.get().strip() or None

                if use_stream:
                    # Streaming mode
                    response_parts = []
                    self.root.after(0, lambda: self._begin_chat_stream())
                    for token in self.engine.generate_chat_streaming(
                        self.chat_history,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        chat_template=template,
                        system_prompt=system_prompt,
                    ):
                        response_parts.append(token)
                        # Update UI live
                        self.root.after(0, lambda t=token: self._append_chat_token(t))

                    full_response = "".join(response_parts).strip()
                    if full_response:
                        # Replace the streaming tokens with final message
                        self.root.after(0, lambda: self._finalize_chat_response(full_response))
                    else:
                        self.root.after(0, lambda: self._append_chat_message("assistant", "[No response]"))
                else:
                    # Non-streaming
                    response = self.engine.generate_chat(
                        self.chat_history,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        chat_template=template,
                        system_prompt=system_prompt,
                    )

                    if response:
                        self.chat_history.append({"role": "assistant", "content": response})
                        self.root.after(0, lambda: self._append_chat_message("assistant", response))
                    else:
                        self.root.after(0, lambda: self._append_chat_message("assistant", "[No response generated]"))

            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: messagebox.showerror("Chat Error", err))
            finally:
                self.root.after(0, lambda: (
                    self.chat_send_btn.config(state=tk.NORMAL),
                    self.chat_status.config(text="Ready")
                ))

        threading.Thread(target=run, daemon=True).start()


    def _append_chat_token(self, token: str):
        """Append a single token during streaming."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, token)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)


    def _begin_chat_stream(self):
        """Insert the assistant prefix before streaming tokens arrive."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "🤖 Assistant: ", "assistant")
        self.chat_text.config(state=tk.DISABLED)


    def _finalize_chat_response(self, full_response: str):
        """Clean up after streaming and store the final response."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "\n\n")
        self.chat_text.config(state=tk.DISABLED)

        # Make sure we have the response in history
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = full_response
        else:
            self.chat_history.append({"role": "assistant", "content": full_response})


    def _append_chat_message(self, role: str, content: str):
        self.chat_text.config(state=tk.NORMAL)
        if role == "user":
            self.chat_text.insert(tk.END, "👤 You: ", "user")
            self.chat_text.insert(tk.END, content + "\n\n")
        elif role == "assistant":
            self.chat_text.insert(tk.END, "🤖 Assistant: ", "assistant")
            self.chat_text.insert(tk.END, content + "\n\n")
        else:
            self.chat_text.insert(tk.END, f"[{role}] {content}\n\n", role)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)


    def _on_agent_mode_toggle(self):
        """Called when the Agent Mode checkbox is toggled."""
        if self.agent_mode_var.get():
            if not gui_common.HAS_AGENT:
                messagebox.showwarning("Agent Not Available",
                    "agent/ package not found. Make sure the agent/ directory is present.")
                self.agent_mode_var.set(False)
                return
            self._log_sandbox("Agent mode enabled. Sandbox initialised.", "info")
            self.chat_status.config(text="🤖 Agent Mode ON — I have access to terminal, files, web search.")
        else:
            if self._agent is not None:
                self._agent.stop()
                self._agent = None
            self.chat_status.config(text="Agent mode disabled.")
            self._log_sandbox("Agent mode disabled.", "info")


    def _send_agent_message(self, user_msg: str):
        """Handle a message in agent mode."""
        self._append_chat_message("user", user_msg)
        self.chat_send_btn.config(state=tk.DISABLED)
        self.agent_stop_btn.config(state=tk.NORMAL)
        self.chat_status.config(text="🤖 Agent thinking...")
        self._agent_stop_event.clear()

        def run():
            try:
                if not gui_common.HAS_AGENT:
                    raise RuntimeError("Agent package not available")

                # Create/reuse sandbox
                if self._agent_sandbox is None:
                    self._agent_sandbox = Sandbox(timeout=15.0)

                # Create fresh agent
                agent = AuraLiteAgent(
                    self.engine,
                    sandbox=self._agent_sandbox,
                    max_iterations=8,
                    max_tokens_per_step=512,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    chat_template=getattr(self, 'chat_template_var', None) and self.chat_template_var.get() or "chatml",
                )
                self._agent = agent

                # Build history (exclude last user msg — agent gets it fresh)
                history = list(self.chat_history[:-1]) if self.chat_history else []

                # Stream agent output to chat display
                self.root.after(0, self._begin_chat_stream)
                full_output_parts = []

                def step_callback(step):
                    # Log each tool call to sandbox log
                    for tc in step.tool_calls:
                        name = tc.get("name", "?")
                        self.root.after(0, lambda n=name: self._log_sandbox(f"→ Tool: {n}", "tool"))
                    for res in step.tool_results:
                        tag = "ok" if res.success else "error"
                        out = res.to_text()[:300]
                        self.root.after(0, lambda o=out, t=tag: self._log_sandbox(o, t))

                agent.on_step = step_callback

                for chunk in agent.run_streaming(user_msg, history=history):
                    if self._agent_stop_event.is_set():
                        break
                    full_output_parts.append(chunk)
                    self.root.after(0, lambda t=chunk: self._append_chat_token(t))

                full_response = "".join(full_output_parts).strip()

                # Store in history
                self.chat_history.append({"role": "user", "content": user_msg})
                if full_response:
                    self.chat_history.append({"role": "assistant", "content": full_response})
                self.root.after(0, lambda: self._finalize_chat_response(full_response or "[Agent produced no output]"))

            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: (
                    self._append_chat_message("assistant", f"[Agent error]: {err}"),
                    self._log_sandbox(f"ERROR: {err}", "error"),
                ))
            finally:
                self._agent = None
                self.root.after(0, lambda: (
                    self.chat_send_btn.config(state=tk.NORMAL),
                    self.agent_stop_btn.config(state=tk.DISABLED),
                    self.chat_status.config(text="🤖 Agent done."),
                ))

        threading.Thread(target=run, daemon=True).start()


    def _stop_agent(self):
        """Stop the running agent."""
        self._agent_stop_event.set()
        if self._agent is not None:
            self._agent.stop()
        self.agent_stop_btn.config(state=tk.DISABLED)
        self.chat_status.config(text="Agent stopped by user.")
        self._log_sandbox("Agent stopped.", "info")


    def _log_sandbox(self, text: str, tag: str = ""):
        """Append a line to the sandbox log widget."""
        try:
            self.agent_sandbox_log.config(state=tk.NORMAL)
            self.agent_sandbox_log.insert(tk.END, text.rstrip() + "\n", tag or None)
            lines = int(self.agent_sandbox_log.index("end-1c").split(".")[0])
            if lines > 500:
                self.agent_sandbox_log.delete("1.0", f"{lines-500}.0")
            self.agent_sandbox_log.see(tk.END)
            self.agent_sandbox_log.config(state=tk.DISABLED)
        except tk.TclError:
            pass


    def _clear_chat_history(self):
        self.chat_history.clear()
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)
        self.chat_status.config(text="Chat history cleared.")

