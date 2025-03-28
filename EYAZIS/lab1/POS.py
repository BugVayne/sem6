import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from collections import Counter
import spacy
import file_operations

nlp = spacy.load('en_core_web_trf')
translations = {
    'ADJ': 'adjective',
    'ADP': 'adposition',
    'ADV': 'adverb',
    'AUX': 'auxiliary',
    'CCONJ': 'coordinating conjunction',
    'DET': 'determiner',
    'INTJ': 'interjection',
    'NOUN': 'noun',
    'NUM': 'numeral',
    'PART': 'particle',
    'PRON': 'pronoun',
    'PROPN': 'proper noun',
    'PUNCT': 'punctuation',
    'SCONJ': 'subordinating conjunction',
    'SYM': 'symbol',
    'VERB': 'verb',
    'X': 'other',
}


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("App")
        self.root.geometry("1000x600")

        self.menu = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="Open", command=self.load_text_file)
        self.file_menu.add_command(label="Save to JSON", command=self.save_to_json)
        self.file_menu.add_command(label="Load from JSON", command=self.load_from_json)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.text = ""
        root.config(menu=self.menu)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10, fill="x")

        tk.Label(control_frame, text="Filter by POS:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar()
        self.filter_var.set("All")
        self.filter_combobox = ttk.Combobox(control_frame, textvariable=self.filter_var)
        self.filter_combobox['values'] = ["All"] + list(translations.values())
        self.filter_combobox.pack(side=tk.LEFT, padx=5)
        self.filter_combobox.bind("<<ComboboxSelected>>", self.apply_filter)

        tk.Label(control_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(control_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=5)

        self.search_button = tk.Button(control_frame, text="Search", command=self.search_symbols)
        self.search_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_filter)
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        self.empty_button = tk.Button(control_frame, text="Empty", command=self.empty_text)
        self.empty_button.pack(side=tk.RIGHT, padx=5)

        self.menu.add_command(label="User Guide", command=self.show_help)

        self.tree = ttk.Treeview(self.root, columns=("Word", "Morphologic info", "Occurrences"), show='headings')
        self.tree.heading("Word", text="Word")
        self.tree.heading("Morphologic info", text="Morphologic info")
        self.tree.heading("Occurrences", text="Occurrences")
        self.tree.column("Occurrences", width=5)
        self.tree.column("Word", width=10)
        self.data = []
        for item in self.data:
            self.tree.insert("", "end", values=item)

        self.tree.pack(expand=True, fill="both")
        self.tree.bind("<Double-1>", self.on_item_double_click)

    def show_help(self):
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("User Guide")
        help_dialog.geometry("600x350")

        help_text = (
            "Добро пожаловать в приложение!\n\n"
            "Функции:\n"
            "- Загрузите текстовый файл для анализа слов.\n"
            "- Фильтруйте слова по части речи (POS).\n"
            "- Ищите конкретные слова.\n"
            "- Редактируйте свойства слов в таблице.\n"
            "- Сохраняйте и загружайте данные в формате JSON.\n\n"
            "Инструкции:\n"
            "1. Используйте меню 'Файл' для загрузки текста или сохранения/загрузки данных JSON.\n"
            "2. Используйте фильтр для просмотра конкретных типов слов.\n"
            "3. Ищите слова с помощью строки поиска.\n"
            "4. Дважды щелкните по слову в таблице, чтобы редактировать его свойства.\n"
            "5. Нажмите 'Сбросить', чтобы очистить фильтры и поиски."
        )

        tk.Label(help_dialog, text=help_text, justify=tk.LEFT, padx=10, pady=10).pack()

        close_button = tk.Button(help_dialog, text="Close", command=help_dialog.destroy)
        close_button.pack(pady=5)

    def load_text_file(self):
        new_text = file_operations.open_file()
        if new_text:
            self.text += "\n" + new_text
            self.append_to_table(new_text)

    def append_to_table(self, new_text):
        doc = nlp(new_text.lower())

        word_counts = Counter(token.text for token in doc if not token.is_punct and not token.is_space)

        new_unique_words = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                if token.text not in new_unique_words:
                    number = token.morph.get("Number")
                    number_str = ', '.join(number) if number else 'N/A'

                    if token.pos_ == 'VERB':
                        is_passive = False

                        if token.dep_ == "ROOT" and any(child.dep_ == "nsubj:pass" for child in token.children):
                            is_passive = True
                        elif any(child.dep_ == "auxpass" for child in token.children):
                            is_passive = True
                        elif any(prep.text == "by" and prep.dep_ == "agent" for prep in token.children):
                            is_passive = True

                        if is_passive:
                            voice_str = "Passive"
                        else:
                            voice_str = "Active"

                        tense = token.morph.get("Tense")
                        tense_str = ', '.join(tense) if tense else 'N/A'

                        if any(child.text in ["will", "shall"] and child.dep_ == "aux" for child in token.children):
                            tense_str = "Future"

                        new_unique_words[token.text] = (
                            f"POS: {translations[token.pos_]}, LEMMA: {token.lemma_}, NUMBER: {number_str}, "
                            f"VOICE: {voice_str}, TENSE: {tense_str}",
                            word_counts[token.text],
                        )
                    elif token.pos_ == 'NOUN' or token.pos_ == "PRON":
                        possessed_by_ = None
                        possessor_of = None

                        for child in token.children:
                            if child.dep_ in ["poss", "nmod:poss"]:
                                possessed_by_ = child.text

                        if token.dep_ in ["poss", "nmod:poss"]:
                            possessor_of = token.head.text

                        if possessed_by_:
                            case_str = f"Possessed by: {possessed_by_}"
                        elif possessor_of:
                            case_str = f"Possesses: {possessor_of}"
                        else:
                            case_str = "N/A"

                        new_unique_words[token.text] = (
                            f"POS: {translations[token.pos_]}, LEMMA: {token.lemma_}, NUMBER: {number_str}, CASE: {case_str}",
                            word_counts[token.text],
                        )

                    else:
                        new_unique_words[token.text] = (
                            f"POS: {translations[token.pos_]}, LEMMA: {token.lemma_}, NUMBER: N/A",
                            word_counts[token.text],
                        )
        for word, (pos_info, count) in new_unique_words.items():
            existing_entry = next((item for item in self.data if item[0] == word), None)
            if existing_entry:
                index = self.data.index(existing_entry)
                self.data[index] = (word, pos_info, existing_entry[2] + count)
            else:
                self.data.append((word, pos_info, count))

        self.data = sorted(self.data)
        self.update_treeview()

    def save_to_json(self):
        data_to_save = {word: (pos, count) for word, pos, count in self.data}
        file_operations.save_to_json(data_to_save)

    def load_from_json(self):
        loaded_data = file_operations.load_from_json()
        if loaded_data:
            self.data = sorted([(word, pos, count) for word, (pos, count) in loaded_data.items()])
            self.update_treeview()

    def update_treeview(self, filtered_data=None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        data_to_display = filtered_data if filtered_data else self.data
        for item in data_to_display:
            self.tree.insert("", "end", values=item)

    def apply_filter(self, event=None):
        selected_filter = self.filter_var.get()
        if selected_filter == "All":
            self.update_treeview()
        else:
            filtered_data = [item for item in self.data if f"POS: {selected_filter}" in item[1]]

            if not filtered_data:
                messagebox.showinfo("No Data Found", f"No words with POS '{selected_filter}' were found.")
                self.update_treeview(filtered_data=[])
            else:
                self.update_treeview(filtered_data)

    def reset_filter(self):
        self.filter_var.set("All")
        self.update_treeview()

    def empty_text(self):
        self.data.clear()
        self.text = ""
        self.update_treeview()


    def search_symbols(self):
        query = self.search_var.get().strip().lower()
        if not query:
            messagebox.showinfo("Invalid Search", "Please enter a symbol or text to search.")
            return

        visible_data = [
            self.tree.item(item_id, "values")
            for item_id in self.tree.get_children()
        ]

        search_results = [
            item for item in visible_data if query in item[0].lower()
        ]

        if not search_results:
            messagebox.showinfo("No Matches Found", f"No words containing '{query}' were found.")
        else:
            self.update_treeview(filtered_data=search_results)

    def on_item_double_click(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            item_values = self.tree.item(selected_item, 'values')
            self.show_edit_dialog(item_values)

    def show_edit_dialog(self, item_values):
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit")
        dialog.geometry("300x600")

        dialog.after(0, dialog.grab_set)
        dialog.focus_set()

        tk.Label(dialog, text="Word:").pack(pady=5)
        word_var = tk.StringVar(value=item_values[0])
        tk.Entry(dialog, textvariable=word_var, state='readonly', width=31).pack(pady=5)

        tk.Label(dialog, text="POS:").pack(pady=5)
        pos_entry = ttk.Combobox(dialog, width=30)
        pos_entry['values'] = list(translations.values())
        pos_entry.pack(pady=5)
        pos_entry.set(item_values[1].split(",")[0].replace("POS: ", "").strip())

        tk.Label(dialog, text="LEMMA:").pack(pady=5)
        lemma_entry = tk.Entry(dialog, width=31)
        lemma_entry.pack(pady=5)
        lemma_entry.insert(0, item_values[1].split(",")[1].replace("LEMMA: ", "").strip())

        tk.Label(dialog, text="NUMBER:").pack(pady=5)
        num_entry = ttk.Combobox(dialog, width=30)
        num_entry['values'] = ['singular', 'plural', 'N/A']
        num_entry.pack(pady=5)
        num_entry.set(item_values[1].split(",")[2].replace("NUMBER:", "").strip())

        # Containers for additional fields, shown dynamically based on POS
        tense_label = None
        tense_entry = None
        voice_label = None
        voice_entry = None
        case_label = None
        case_entry = None
        possession_word_label = None
        possession_word_entry = None

        def update_dynamic_fields():
            nonlocal tense_label, tense_entry, voice_label, voice_entry, case_label, case_entry, possession_word_label, possession_word_entry

            if tense_label: tense_label.pack_forget()
            if tense_entry: tense_entry.pack_forget()
            if voice_label: voice_label.pack_forget()
            if voice_entry: voice_entry.pack_forget()
            if case_label: case_label.pack_forget()
            if case_entry: case_entry.pack_forget()
            if possession_word_label: possession_word_label.pack_forget()
            if possession_word_entry: possession_word_entry.pack_forget()

            selected_pos = pos_entry.get()

            if selected_pos == "verb":
                tense_label = tk.Label(dialog, text="TENSE:")
                tense_label.pack(pady=5)
                tense_entry = ttk.Combobox(dialog, width=30)
                tense_entry['values'] = ['Past', 'Present', 'Future', 'N/A']
                tense_entry.pack(pady=5)
                tense_value = item_values[1].split(",")[4].replace("TENSE:", "").strip() if "TENSE:" in item_values[
                    1] else "N/A"
                tense_entry.set(tense_value)

                voice_label = tk.Label(dialog, text="VOICE:")
                voice_label.pack(pady=5)
                voice_entry = ttk.Combobox(dialog, width=30)
                voice_entry['values'] = ['Active', 'Passive', 'N/A']
                voice_entry.pack(pady=5)
                voice_value = item_values[1].split(",")[3].replace("VOICE:", "").strip() if "VOICE:" in item_values[
                    1] else "N/A"
                voice_entry.set(voice_value)

            elif selected_pos == "noun":
                case_label = tk.Label(dialog, text="CASE:")
                case_label.pack(pady=5)
                case_entry = ttk.Combobox(dialog, width=30)
                case_entry['values'] = ['Possessed by', 'Possesses', 'N/A']
                case_entry.pack(pady=5)
                case_value = item_values[1].split(",")[3].replace("CASE:", "").strip() if "CASE:" in item_values[
                    1] else "N/A"
                case_entry.set(case_value)

                def update_possession_field(event=None):
                    nonlocal possession_word_label, possession_word_entry

                    # Clear previous possession fields
                    if possession_word_label: possession_word_label.pack_forget()
                    if possession_word_entry: possession_word_entry.pack_forget()

                    selected_case = case_entry.get()

                    if selected_case in ["Possessed by", "Possesses"]:
                        possession_word_label = tk.Label(dialog, text=f" Word for {selected_case.lower()}:")
                        possession_word_label.pack(pady=5, before=submit_button)
                        possession_word_entry = tk.Entry(dialog, width=31)
                        possession_word_entry.pack(pady=5, before=submit_button)

                        # Extract existing possession data if available
                        possession_value = item_values[1].split(",")[4].replace(f"{selected_case}:",
                                                                                "").strip() if f"{selected_case}:" in \
                                                                                               item_values[1] else ""
                        possession_word_entry.insert(0, possession_value)

                # Update possession field dynamically when CASE changes
                case_entry.bind("<<ComboboxSelected>>", update_possession_field)

                # Initialize possession field based on current case
                update_possession_field()

        # Update fields dynamically when POS changes
        pos_entry.bind("<<ComboboxSelected>>", lambda event: update_dynamic_fields())

        # Initialize dynamic fields based on current POS
        update_dynamic_fields()

        def submit():
            pos = pos_entry.get()
            lemma = lemma_entry.get()
            number = num_entry.get()

            # Update morphological information dynamically based on POS
            if pos == "verb":
                tense = tense_entry.get() if tense_entry else "N/A"
                voice = voice_entry.get() if voice_entry else "N/A"
                morph_info = f"POS: {pos}, LEMMA: {lemma}, NUMBER: {number}, VOICE: {voice}, TENSE: {tense}"

            elif pos == "noun":
                case = case_entry.get() if case_entry else "N/A"
                possession_word = possession_word_entry.get() if possession_word_entry else "N/A"
                morph_info = f"POS: {pos}, LEMMA: {lemma}, NUMBER: {number}, CASE: {case}: {possession_word}"
            else:
                morph_info = f"POS: {pos}, LEMMA: {lemma}, NUMBER: {number}"

            index = next((i for i, row in enumerate(self.data) if row[0] == item_values[0]), None)

            if index is not None:
                self.data[index] = (item_values[0], morph_info, self.data[index][2])

                self.update_treeview()
            dialog.destroy()

        submit_button = tk.Button(dialog, text="Edit", command=submit)
        submit_button.pack(pady=10)

    def create_table(self):
        doc = nlp(self.text.lower())

        word_counts = Counter(token.text for token in doc if not token.is_punct and not token.is_space)

        unique_words = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                if token.text not in unique_words:
                    number = token.morph.get("Number")
                    number_str = ', '.join(number) if number else 'N/A'
                    if token.pos_ in ['NOUN', 'VERB']:
                        unique_words[token.text] = (
                        "POS: " + translations[token.pos_] + ", LEMMA: " + token.lemma_ + ", NUMBER: " + number_str,
                        word_counts[token.text])
                    else:
                        unique_words[token.text] = (
                        "POS: " + translations[token.pos_] + ", LEMMA: " + token.lemma_ + ", NUMBER: N/A",
                        word_counts[token.text])

        self.data = sorted([(word, pos, count) for word, (pos, count) in unique_words.items()])

        self.update_treeview()


def main():
    root = tk.Tk()
    app = App(root)

    root.mainloop()


if __name__ == "__main__":
    main()