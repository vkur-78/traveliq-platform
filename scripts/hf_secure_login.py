"""
Secure local Hugging Face authentication helper for TRAVELIQ.

Reads the Hugging Face Write token via hidden terminal input (`getpass`) or
a masked local GUI dialog (`--gui`) and saves it exclusively to the official
Hugging Face local cache (`~/.cache/huggingface/token`) via `huggingface_hub.login`.

NEVER prints, logs, or echoes the token.
"""

import argparse
import getpass
import sys

from huggingface_hub import HfApi, login


def save_token_securely(token: str) -> str:
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Token cannot be empty.")
    api = HfApi(token=cleaned)
    who = api.whoami()
    username = who.get("name") or "unknown"
    login(token=cleaned, add_to_git_credential=False)
    return username


def run_gui() -> int:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("TRAVELIQ — Secure Hugging Face Local Login")
    root.geometry("500x210")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frame = tk.Frame(root, padx=20, pady=18)
    frame.pack(fill="both", expand=True)

    tk.Label(
        frame,
        text="Paste your Hugging Face WRITE Access Token below.\n"
             "Input is masked and saved ONLY to ~/.cache/huggingface/token.",
        justify="left",
        font=("Segoe UI", 10),
    ).pack(anchor="w", pady=(0, 10))

    token_var = tk.StringVar()
    entry = tk.Entry(frame, textvariable=token_var, show="*", width=56, font=("Consolas", 10))
    entry.pack(fill="x", pady=(0, 14))
    entry.focus_set()

    result = {"ok": False}

    def on_submit(_event=None):
        try:
            username = save_token_securely(token_var.get())
            token_var.set("")
            result["ok"] = True
            messagebox.showinfo(
                "TRAVELIQ — Authenticated",
                f"Authenticated securely as Hugging Face user: {username}\n\n"
                "You can close this window and return to Antigravity.",
            )
            root.destroy()
        except Exception as exc:
            messagebox.showerror(
                "Authentication Failed",
                f"Could not verify token ({type(exc).__name__}).\n"
                "Ensure you created a valid Hugging Face WRITE token and try again.",
            )

    btn_row = tk.Frame(frame)
    btn_row.pack(fill="x")
    tk.Button(
        btn_row,
        text="Verify & Save Securely",
        command=on_submit,
        bg="#1D63ED",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        padx=14,
        pady=5,
    ).pack(side="left")
    tk.Button(
        btn_row,
        text="Cancel",
        command=root.destroy,
        font=("Segoe UI", 10),
        padx=12,
        pady=5,
    ).pack(side="right")

    root.bind("<Return>", on_submit)
    root.mainloop()
    return 0 if result["ok"] else 1


def run_cli() -> int:
    print("TRAVELIQ — Secure Local Hugging Face Login")
    print("Input is hidden and saved ONLY to ~/.cache/huggingface/token (never printed).")
    token = getpass.getpass("Paste Hugging Face WRITE token (hidden): ")
    try:
        username = save_token_securely(token)
    except Exception as exc:
        print(f"Authentication failed: {type(exc).__name__}")
        return 1
    print(f"Authenticated securely as Hugging Face user: {username}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Secure local Hugging Face login (hidden input)")
    parser.add_argument("--gui", action="store_true", help="Open masked local desktop dialog")
    args = parser.parse_args()
    return run_gui() if args.gui else run_cli()


if __name__ == "__main__":
    sys.exit(main())
