import json
from tkinter import filedialog, messagebox
from docx import Document


def open_file():
    file_path = filedialog.askopenfilename(
        title="Choose a file",
        filetypes=[("Word Documents", "*.docx"), ("Word Documents", "*.doc")]
    )

    if file_path:
        try:
            text = extract_text(file_path)
            messagebox.showinfo("Info", "Text extracted successfully.")
            return text
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract text: {str(e)}")


def extract_text(file_path):
    document = Document(file_path)
    full_text = []

    for paragraph in document.paragraphs:
        full_text.append(paragraph.text)

    return '\n'.join(full_text)


def save_to_json(data, default_file_name="output.json"):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save as"
    )

    if file_path:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        messagebox.showinfo("Info", "Data saved successfully.")


def load_from_json():
    file_path = filedialog.askopenfilename(
        title="Load JSON file",
        filetypes=[("JSON files", "*.json")]
    )

    if file_path:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data