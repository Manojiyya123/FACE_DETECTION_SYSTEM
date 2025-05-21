import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
from facerec import collect_samples, train_model, recognize_faces, list_people, delete_person

# Create main window
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("400x400")
root.resizable(False, False)

# Header label
tk.Label(root, text="Face Recognition System", font=("Helvetica", 16, "bold")).pack(pady=20)

# Button Functions
def register_person():
    name = simpledialog.askstring("Register", "Enter person's name:")
    if name:
        collect_samples(name)
        messagebox.showinfo("Done", f"Samples collected for {name}")

def train_model_gui():
    train_model()
    messagebox.showinfo("Training", "Model training complete!")

def recognize_faces_gui():
    messagebox.showinfo("Info", "Press 'Q' to exit recognition window.")
    recognize_faces()

def list_people_gui():
    output = []
    try:
        from io import StringIO
        import sys
        temp_stdout = StringIO()
        sys_stdout_backup = sys.stdout
        sys.stdout = temp_stdout
        list_people()
        sys.stdout = sys_stdout_backup
        output = temp_stdout.getvalue()
    except:
        output = "Error retrieving people list"
    
    msg = tk.Toplevel(root)
    msg.title("Registered People")
    text_area = scrolledtext.ScrolledText(msg, width=40, height=15)
    text_area.insert(tk.END, output)
    text_area.pack()
    text_area.configure(state='disabled')

def delete_person_gui():
    name = simpledialog.askstring("Delete", "Enter name to delete:")
    if name:
        delete_person(name)
        messagebox.showinfo("Deleted", f"{name} has been deleted.\nPlease retrain the model.")

# Buttons
tk.Button(root, text="Register New Person", width=25, command=register_person).pack(pady=8)
tk.Button(root, text="Train Model", width=25, command=train_model_gui).pack(pady=8)
tk.Button(root, text="Recognize Faces", width=25, command=recognize_faces_gui).pack(pady=8)
tk.Button(root, text="List Registered People", width=25, command=list_people_gui).pack(pady=8)
tk.Button(root, text="Delete Person", width=25, command=delete_person_gui).pack(pady=8)
tk.Button(root, text="Exit", width=25, command=root.quit).pack(pady=20)

# Start GUI loop
root.mainloop()
