"""gui/tabs/result_view.py — ResultViewMixin for the AuraLite GUI (v2.6.5).

Extracted verbatim from `gui_app.py`; behaviour is unchanged.
"""
from ..common import (filedialog, messagebox, os, tk)
from ..i18n import tr

class ResultViewMixin:
    def _display_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)


    def _get_result_text(self) -> str:
        # strip the trailing newline tkinter always appends
        return self.result_text.get("1.0", "end-1c")


    def _copy_result(self):
        text = self._get_result_text()
        if not text.strip():
            self.status_label.config(text=tr("Status: Output is empty — nothing to copy."))
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()  # ensure clipboard persists after focus change
            n = len(text)
            self.status_label.config(
                text=tr("Status: Copied {n:,} characters to clipboard ✅", n=n))
        except tk.TclError as e:
            self.report_error(e, tr("copying"), title=tr("Copy Error"))


    def _copy_selection_result(self):
        try:
            sel = self.result_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return self._copy_result()  # nothing selected → copy all
        if not sel:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(sel)
        self.root.update()
        self.status_label.config(
            text=tr("Status: Copied selection ({n:,} chars) ✅", n=len(sel)))


    def _clear_result(self):
        self.result_text.delete("1.0", tk.END)


    def _save_result(self):
        text = self._get_result_text()
        if not text.strip():
            messagebox.showinfo(tr("Empty"), tr("Nothing to save — output is empty."))
            return
        path = filedialog.asksaveasfilename(
            title=tr("Save generated text"),
            defaultextension=".txt",
            filetypes=[(tr("Text files"), "*.txt"), (tr("Markdown"), "*.md"),
                       (tr("All files"), "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_label.config(
                text=tr("Status: Saved output ✅ ({name})", name=os.path.basename(path)))
        except Exception as e:
            self.report_error(e, tr("saving"), title=tr("Save Error"))


    def _append_result(self):
        text = self._get_result_text()
        if not text.strip():
            messagebox.showinfo(tr("Empty"), tr("Nothing to append — output is empty."))
            return
        path = filedialog.asksaveasfilename(
            title=tr("Append generated text to file (existing file will be appended)"),
            defaultextension=".txt",
            filetypes=[(tr("Text files"), "*.txt"), (tr("Markdown"), "*.md"),
                       (tr("All files"), "*.*")],
        )
        if not path:
            return
        try:
            sep = "\n\n" + "=" * 50 + "\n"
            with open(path, "a", encoding="utf-8") as f:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    f.write(sep)
                f.write(text)
            self.status_label.config(
                text=tr("Status: Appended output ✅ ({name})", name=os.path.basename(path)))
        except Exception as e:
            self.report_error(e, tr("appending to the file"), title=tr("Append Error"))


    def _show_result_menu(self, event):
        try:
            self._result_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._result_menu.grab_release()

