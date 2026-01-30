import tkinter as tk
from tkinter import filedialog, messagebox
import fastavro
import csv
import os

class AvroConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jarvis - Avro to CSV Converter")
        self.root.geometry("500x200")
        self.root.resizable(False, False)

        # Variables
        self.file_path = tk.StringVar()

        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self.root, text="Avro to CSV Converter", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # File Selection Frame
        frame = tk.Frame(self.root)
        frame.pack(pady=10, padx=20, fill="x")

        self.path_entry = tk.Entry(frame, textvariable=self.file_path, state='readonly', width=40)
        self.path_entry.pack(side="left", padx=5, fill="x", expand=True)

        browse_btn = tk.Button(frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side="right")

        # Convert Button
        self.convert_btn = tk.Button(self.root, text="Convert to CSV", command=self.convert_file, state="disabled", bg="#4CAF50", fg="white", font=("Arial", 12))
        self.convert_btn.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", fg="gray")
        self.status_label.pack(side="bottom", pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Avro File",
            filetypes=(("Avro files", "*.avro"), ("All files", "*.*"))
        )
        if filename:
            self.file_path.set(filename)
            self.convert_btn.config(state="normal")
            self.status_label.config(text="File selected. Ready to convert.")

    def convert_file(self):
        avro_path = self.file_path.get()
        if not avro_path:
            return

        # Ask where to save the CSV
        csv_path = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if not csv_path:
            return

        self.status_label.config(text="Converting...")
        self.root.update()

        try:
            self.perform_conversion(avro_path, csv_path)
            self.status_label.config(text="Conversion Complete!", fg="green")
            messagebox.showinfo("Success", f"Successfully converted to:\n{csv_path}")
        except Exception as e:
            self.status_label.config(text="Error occurred", fg="red")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def perform_conversion(self, avro_path, csv_path):
        with open(avro_path, 'rb') as f_in:
            reader = fastavro.reader(f_in)
            # fastavro reader automatically parses the schema from the file
            # If records are simple dictionaries, we can infer headers from the first record or the schema
            
            # Try to get fields from schema if available
            fields = []
            if reader.writer_schema and 'fields' in reader.writer_schema:
                fields = [f['name'] for f in reader.writer_schema['fields']]
            
            # Prepare CSV writer
            with open(csv_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = None
                
                for i, record in enumerate(reader):
                    if i == 0:
                        # If we didn't get fields from schema, try to get from first record keys
                        if not fields:
                            fields = list(record.keys())
                        
                        writer = csv.DictWriter(f_out, fieldnames=fields)
                        writer.writeheader()
                    
                    # Handle cases where record might have extra keys or missing keys based on schema
                    # dict_writer handles extras with 'extrasaction' (default raise) or we clean it
                    # We'll rely on standard behavior for now.
                    if writer:
                        writer.writerow(record)

if __name__ == "__main__":
    root = tk.Tk()
    app = AvroConverterApp(root)
    root.mainloop()
