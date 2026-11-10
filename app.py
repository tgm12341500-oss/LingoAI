
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from datetime import datetime
import json
import csv
import joblib
import customtkinter as ctk


# ==========================================
# 1. PROJECT PATHS AND AI MODELS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"

try:
    model = joblib.load(BASE_DIR / "language_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

except Exception as error:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Model Loading Error",
        f"Could not load the AI model files.\n\n{error}"
    )
    raise SystemExit


# ==========================================
# 2. LOAD SAVED HISTORY
# ==========================================

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

                if isinstance(data, list):
                    return data

        except (json.JSONDecodeError, OSError):
            pass

    return []


history = load_history()


def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=4)

    except OSError as error:
        print("History saving error:", error)


# ==========================================
# 3. APPLICATION CONFIGURATION
# ==========================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("LingoAI | Language Detection")
app.geometry("1150x760")
app.minsize(950, 650)

BG = "#0B1120"
SIDEBAR = "#111B2E"
CARD = "#17243A"
CARD_LIGHT = "#1D2D47"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"
GREEN = "#22C55E"
RED = "#EF4444"


# ==========================================
# 4. MAIN LAYOUT
# ==========================================

sidebar = ctk.CTkFrame(
    app,
    width=230,
    corner_radius=0,
    fg_color=SIDEBAR
)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

content = ctk.CTkFrame(
    app,
    corner_radius=0,
    fg_color=BG
)
content.pack(side="right", fill="both", expand=True)


# ==========================================
# 5. SIDEBAR
# ==========================================

logo = ctk.CTkLabel(
    sidebar,
    text="✦  LingoAI",
    font=("Arial", 27, "bold"),
    text_color=WHITE
)
logo.pack(anchor="w", padx=25, pady=(35, 5))

logo_subtitle = ctk.CTkLabel(
    sidebar,
    text="INTELLIGENT LANGUAGE TOOLS",
    font=("Arial", 9, "bold"),
    text_color=MUTED
)
logo_subtitle.pack(anchor="w", padx=27, pady=(0, 40))


def sidebar_button(text, command):
    return ctk.CTkButton(
        sidebar,
        text=text,
        command=command,
        height=45,
        corner_radius=10,
        fg_color="transparent",
        hover_color=CARD_LIGHT,
        text_color=WHITE,
        anchor="w",
        font=("Arial", 14)
    )


# ==========================================
# 6. CONTENT PAGES
# ==========================================

dashboard_page = ctk.CTkFrame(
    content,
    fg_color=BG,
    corner_radius=0
)

history_page = ctk.CTkFrame(
    content,
    fg_color=BG,
    corner_radius=0
)

about_page = ctk.CTkFrame(
    content,
    fg_color=BG,
    corner_radius=0
)

for page in (dashboard_page, history_page, about_page):
    page.place(relx=0, rely=0, relwidth=1, relheight=1)


def show_page(page):
    page.tkraise()


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

sidebar_button(
    "⌂   Dashboard",
    lambda: show_page(dashboard_page)
).pack(fill="x", padx=15, pady=5)

sidebar_button(
    "◷   Detection History",
    lambda: (
        refresh_history_page(),
        show_page(history_page)
    )
).pack(fill="x", padx=15, pady=5)

sidebar_button(
    "ⓘ   About Project",
    lambda: show_page(about_page)
).pack(fill="x", padx=15, pady=5)


sidebar_footer = ctk.CTkFrame(
    sidebar,
    fg_color="transparent"
)
sidebar_footer.pack(side="bottom", fill="x", padx=20, pady=25)

ctk.CTkLabel(
    sidebar_footer,
    text="POWERED BY",
    font=("Arial", 9, "bold"),
    text_color=MUTED
).pack(anchor="w")

ctk.CTkLabel(
    sidebar_footer,
    text="Python  •  Machine Learning",
    font=("Arial", 11),
    text_color=WHITE
).pack(anchor="w", pady=5)


# ==========================================
# 8. DASHBOARD HEADER
# ==========================================

header = ctk.CTkFrame(
    dashboard_page,
    fg_color="transparent"
)
header.pack(fill="x", padx=35, pady=(30, 15))

ctk.CTkLabel(
    header,
    text="AI Language Dashboard",
    font=("Arial", 27, "bold"),
    text_color=WHITE
).pack(anchor="w")

ctk.CTkLabel(
    header,
    text="Detect the language of your text using machine learning.",
    font=("Arial", 13),
    text_color=MUTED
).pack(anchor="w", pady=(5, 0))


# ==========================================
# 9. STATISTICS
# ==========================================

stats_frame = ctk.CTkFrame(
    dashboard_page,
    fg_color="transparent"
)
stats_frame.pack(fill="x", padx=25, pady=10)

total_value = ctk.StringVar(value="0")
languages_value = ctk.StringVar(value="0")
last_value = ctk.StringVar(value="None")


def create_stat_card(parent, title, variable, color):
    card = ctk.CTkFrame(
        parent,
        fg_color=CARD,
        corner_radius=14,
        height=100
    )
    card.pack(
        side="left",
        fill="both",
        expand=True,
        padx=10
    )
    card.pack_propagate(False)

    ctk.CTkLabel(
        card,
        text=title,
        font=("Arial", 12),
        text_color=MUTED
    ).pack(anchor="w", padx=18, pady=(15, 3))

    ctk.CTkLabel(
        card,
        textvariable=variable,
        font=("Arial", 23, "bold"),
        text_color=color
    ).pack(anchor="w", padx=18)


create_stat_card(
    stats_frame,
    "TOTAL DETECTIONS",
    total_value,
    BLUE
)

create_stat_card(
    stats_frame,
    "UNIQUE LANGUAGES",
    languages_value,
    PURPLE
)

create_stat_card(
    stats_frame,
    "LAST DETECTION",
    last_value,
    GREEN
)


# ==========================================
# 10. TEXT ANALYZER CARD
# ==========================================

analyzer = ctk.CTkFrame(
    dashboard_page,
    fg_color=CARD,
    corner_radius=16
)
analyzer.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=(15, 25)
)

ctk.CTkLabel(
    analyzer,
    text="Text Analysis",
    font=("Arial", 20, "bold"),
    text_color=WHITE
).pack(anchor="w", padx=25, pady=(22, 5))

ctk.CTkLabel(
    analyzer,
    text="Enter your text below to identify its language.",
    font=("Arial", 12),
    text_color=MUTED
).pack(anchor="w", padx=25, pady=(0, 15))


text_input = ctk.CTkTextbox(
    analyzer,
    height=125,
    corner_radius=12,
    fg_color="#0F1A2D",
    border_width=1,
    border_color="#334155",
    text_color=WHITE,
    font=("Arial", 15),
    wrap="word"
)
text_input.pack(fill="x", padx=25)


# ==========================================
# 11. TEXT COUNTER
# ==========================================

counter = ctk.CTkLabel(
    analyzer,
    text="0 words  |  0 characters",
    font=("Arial", 11),
    text_color=MUTED
)
counter.pack(anchor="e", padx=28, pady=(5, 10))


def update_counter(event=None):
    text = text_input.get("1.0", "end-1c")
    words = len(text.split())
    characters = len(text)

    counter.configure(
        text=f"{words} words  |  {characters} characters"
    )


text_input.bind("<KeyRelease>", update_counter)


# ==========================================
# 12. RESULT CARD
# ==========================================

result_card = ctk.CTkFrame(
    analyzer,
    fg_color="#102D3D",
    corner_radius=12,
    height=75
)
result_card.pack(fill="x", padx=25, pady=12)
result_card.pack_propagate(False)

result_label = ctk.CTkLabel(
    result_card,
    text="Your detected language will appear here",
    font=("Arial", 15, "bold"),
    text_color=MUTED
)
result_label.pack(expand=True)


# ==========================================
# 13. STATISTICS UPDATE
# ==========================================

def update_statistics():
    total_value.set(str(len(history)))

    languages = {
        item.get("language")
        for item in history
        if item.get("language")
    }

    languages_value.set(str(len(languages)))

    if history:
        last_value.set(history[0].get("language", "None"))
    else:
        last_value.set("None")


# ==========================================
# 14. DETECT LANGUAGE
# ==========================================

def detect_language():
    text = text_input.get("1.0", "end-1c").strip().lower()

    if not text:
        result_label.configure(
            text="Please enter some text first.",
            text_color=RED
        )
        return

    if not any(character.isalpha() for character in text):
        result_label.configure(
            text="Please enter text containing letters.",
            text_color=RED
        )
        return

    try:
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        language = str(label_encoder.inverse_transform(prediction)[0])

        result_label.configure(
            text=f"Detected Language: {language}",
            text_color=GREEN
        )

        history.insert(0, {
            "text": text[:250],
            "language": language,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        del history[100:]
        save_history()
        update_statistics()

    except Exception as error:
        result_label.configure(
            text="Detection failed. Please try again.",
            text_color=RED
        )
        print("Detection error:", error)


# ==========================================
# 15. CLEAR AND COPY
# ==========================================

def clear_text():
    text_input.delete("1.0", "end")

    result_label.configure(
        text="Your detected language will appear here",
        text_color=MUTED
    )

    update_counter()


def copy_result():
    current_result = result_label.cget("text")

    if not current_result.startswith("Detected Language:"):
        messagebox.showinfo(
            "Copy Result",
            "Please detect a language first."
        )
        return

    app.clipboard_clear()
    app.clipboard_append(current_result)
    app.update()

    messagebox.showinfo(
        "Copied",
        "Result copied successfully."
    )


# ==========================================
# 16. ACTION BUTTONS
# ==========================================

actions = ctk.CTkFrame(
    analyzer,
    fg_color="transparent"
)
actions.pack(pady=(5, 20))

ctk.CTkButton(
    actions,
    text="✦  Detect Language",
    width=190,
    height=43,
    corner_radius=10,
    fg_color=BLUE,
    hover_color="#2563EB",
    font=("Arial", 13, "bold"),
    command=detect_language
).pack(side="left", padx=7)

ctk.CTkButton(
    actions,
    text="Copy Result",
    width=125,
    height=43,
    corner_radius=10,
    fg_color=PURPLE,
    hover_color="#7C3AED",
    font=("Arial", 12, "bold"),
    command=copy_result
).pack(side="left", padx=7)

ctk.CTkButton(
    actions,
    text="Clear",
    width=100,
    height=43,
    corner_radius=10,
    fg_color=CARD_LIGHT,
    hover_color="#334155",
    font=("Arial", 12),
    command=clear_text
).pack(side="left", padx=7)


# ==========================================
# 17. EXPORT HISTORY TO CSV
# ==========================================

def export_history():
    if not history:
        messagebox.showinfo(
            "No History",
            "There is no detection history to export."
        )
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="lingoai_history.csv",
        title="Export Detection History"
    )

    if not file_path:
        return

    try:
        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=["text", "language", "time"]
            )

            writer.writeheader()
            writer.writerows(history)

        messagebox.showinfo(
            "Export Successful",
            "Your detection history was exported successfully!"
        )

    except Exception as error:
        messagebox.showerror(
            "Export Error",
            f"Could not export history:\n{error}"
        )


# ==========================================
# 18. HISTORY PAGE
# ==========================================

history_header = ctk.CTkLabel(
    history_page,
    text="Detection History",
    font=("Arial", 27, "bold"),
    text_color=WHITE
)
history_header.pack(anchor="w", padx=35, pady=(35, 5))

ctk.CTkLabel(
    history_page,
    text="Your latest language detection activities.",
    font=("Arial", 13),
    text_color=MUTED
).pack(anchor="w", padx=35, pady=(0, 20))


# EXPORT BUTTON

ctk.CTkButton(
    history_page,
    text="↓  Export History to CSV",
    command=export_history,
    width=210,
    height=40,
    corner_radius=10,
    fg_color=BLUE,
    hover_color="#2563EB",
    font=("Arial", 12, "bold")
).pack(anchor="w", padx=35, pady=(0, 15))


history_list = ctk.CTkScrollableFrame(
    history_page,
    fg_color=CARD,
    corner_radius=14
)
history_list.pack(fill="both", expand=True, padx=35, pady=(0, 30))


def refresh_history_page():
    for widget in history_list.winfo_children():
        widget.destroy()

    if not history:
        ctk.CTkLabel(
            history_list,
            text="No detection history yet.",
            font=("Arial", 15),
            text_color=MUTED
        ).pack(pady=35)
        return

    for item in history:
        row = ctk.CTkFrame(
            history_list,
            fg_color=CARD_LIGHT,
            corner_radius=10
        )
        row.pack(fill="x", padx=10, pady=7)

        ctk.CTkLabel(
            row,
            text=item.get("language", "Unknown"),
            font=("Arial", 15, "bold"),
            text_color=GREEN
        ).pack(anchor="w", padx=15, pady=(10, 2))

        ctk.CTkLabel(
            row,
            text=item.get("text", ""),
            font=("Arial", 12),
            text_color=WHITE,
            wraplength=650,
            justify="left"
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkLabel(
            row,
            text=item.get("time", ""),
            font=("Arial", 10),
            text_color=MUTED
        ).pack(anchor="w", padx=15, pady=(2, 10))


# ==========================================
# 19. ABOUT PAGE
# ==========================================

ctk.CTkLabel(
    about_page,
    text="About LingoAI",
    font=("Arial", 28, "bold"),
    text_color=WHITE
).pack(anchor="w", padx=35, pady=(40, 20))

about_card = ctk.CTkFrame(
    about_page,
    fg_color=CARD,
    corner_radius=16
)
about_card.pack(fill="x", padx=35, pady=10)

about_text = (
    "LingoAI is a machine learning application "
    "designed to identify the language of written text.\n\n"
    "TECHNOLOGIES\n"
    "• Python\n"
    "• CustomTkinter\n"
    "• Scikit-learn\n"
    "• TF-IDF Feature Extraction\n"
    "• Support Vector Machine (SVM)\n\n"
    "The application uses a trained machine learning model "
    "to predict the language of the submitted text."
)

ctk.CTkLabel(
    about_card,
    text=about_text,
    font=("Arial", 14),
    text_color=WHITE,
    justify="left"
).pack(anchor="w", padx=25, pady=25)


# ==========================================
# 20. KEYBOARD SHORTCUTS
# ==========================================

app.bind("<Control-Return>", lambda event: detect_language())
app.bind("<Control-l>", lambda event: clear_text())


# ==========================================
# 21. START APPLICATION
# ==========================================

update_statistics()
show_page(dashboard_page)

app.mainloop()
 