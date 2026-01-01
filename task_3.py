# Python Software Foundation. (2025) *shutil — High-level file operations*. Available at: https://docs.python.org/3/library/shutil.html (Accessed: 19 October 2025)
# Python Software Foundation. (2025) *os.system — Execute the command (a string) in a subshell*. Available at: https://docs.python.org/3/library/os.html#os.system (Accessed: 19 October 2025)
# Python Software Foundation. (2025) *os.makedirs — Recursive directory creation function*. Available at: https://docs.python.org/3/library/os.html#os.makedirs (Accessed: 19 October 2025)
# Python Software Foundation. (2025) *os.stat — Perform a stat system call on the given path*. Available at: https://docs.python.org/3/library/os.html#os.stat (Accessed: 19 October 2025)
# Python Software Foundation. (2025) *shutil.rmtree — Delete an entire directory tree*. Available at: https://docs.python.org/3/library/shutil.html#shutil.rmtree (Accessed: 19 October 2025)

import zipfile
from zipfile import ZipFile
import sys
from datetime import datetime
import os
import shutil

log_file = "transactions.log"

# Function to record a log message
def record_log(msg):
    # Get current date and time for the log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open log file in append mode and write the message with timestamp
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


# Add a blank line and log that the program has started
with open(log_file, "a") as f:
    f.write("\n")
record_log("Program started.")


# Function to check if a provided file path is a valid zip file
def is_zip_valid(path):
    # Check if the file exists
    if not os.path.exists(path):
        print("File not found.")
        record_log("Invalid zip file input – file missing.")
        return False
    # Check if the file is a valid zip archive
    if not zipfile.is_zipfile(path):
        print("Invalid zip file.")
        record_log("Invalid zip file input – not a zip archive.")
        return False
    # Return True if all checks passed
    return True


# Function to load a zip file entered by the user
def load_zip():
    print("\n-- Load Zip File --")
    # Ask the user to type in the name of the zip file
    filename = input("Enter the zip filename (e.g., test1.zip): ").strip()
    # If a zip file is already loaded (files folder exists), skip loading again
    if os.path.exists("files"):
        print("A zip file is already loaded. Returning to main menu.")
        record_log("Load zip attempted but zip already loaded.")
        return filename
    # Validate the entered filename to ensure it’s a proper zip file
    if not is_zip_valid(filename):
        record_log(f"Invalid zip file entered: {filename}")
        return None
    # Confirm to the user that the zip file is valid and loaded
    print(f"{filename} loaded successfully.")
    record_log(f"Zip file loaded: {filename}")
    # Create a folder named 'files' to store extracted files later
    os.makedirs("files", exist_ok=True)
    record_log("Files folder created.")
    # Return the name of the loaded zip file for use in other functions
    return filename


# Function to display details about the loaded zip file
def show_info(zip_path):
    print("\n-- Zip File Details --")
    # Check that a zip file has been loaded before continuing
    if not zip_path or not os.path.exists(zip_path):
        print("No zip file loaded.")
        record_log("Unsuccessful zip file details operation.")
        return
    # Get full absolute path of the zip file
    abs_path = os.path.abspath(zip_path)
    # Retrieve size and date information using os.stat
    stats = os.stat(zip_path)
    # Display all gathered details about the zip file
    print(f"Absolute path: {abs_path}")
    print(f"Size: {stats.st_size} bytes")
    print(f"Created:  {datetime.fromtimestamp(stats.st_ctime)}")
    print(f"Modified: {datetime.fromtimestamp(stats.st_mtime)}")
    print(f"Accessed: {datetime.fromtimestamp(stats.st_atime)}")
    # Log that the details were successfully displayed
    record_log("Zip file details displayed.")


# Function to list all files inside the zip archive
def show_files(zip_path):
    print("\n-- Zip File Contents --")
    # Ensure a zip file is loaded before listing contents
    if not zip_path or not os.path.exists(zip_path):
        print("No zip file loaded.")
        record_log("Unsuccessful zip file contents operation.")
        return
    # Open the zip file and get information about each contained file
    with ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # Format the modified date of each file
            mod_time = datetime(*info.date_time)
            # Display filename, size, and modification date in a readable format
            print(f"{info.filename:40} {info.file_size:10} bytes  Modified: {mod_time}")
    # Log that the contents have been listed
    record_log("Zip file contents displayed.")


# Function to extract all files in the zip archive to the 'files' folder
def extract_all(zip_path):
    print("\n-- Extract Zip File --")
    # Check if a zip file has been loaded before trying to extract
    if not zip_path or not os.path.exists(zip_path):
        print("No zip file loaded.")
        record_log("Unsuccessful extract zip file operation.")
        return
    # Set the target folder where files will be extracted
    target = "files"
    # If the folder already exists, remove it first to avoid duplicate contents
    if os.path.exists(target):
        shutil.rmtree(target)
    # Create a new clean folder for extraction
    os.makedirs(target)
    # Open and extract all files from the zip to the target folder
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
        file_list = zf.namelist()
    # Display how many files were extracted and where they were saved
    print(f"Extraction complete. {len(file_list)} file(s) extracted to: {os.path.abspath(target)}")
    # Log the extraction and list each file extracted
    record_log("Zip file extracted.")
    for f in file_list:
        record_log(f"   {f}")


# Function to clean up and exit the program
def exit_prog(zip_path):
    print("\n-- Quit Program --")
    # If a zip was loaded, delete the 'files' folder to tidy up
    if zip_path and os.path.exists("files"):
        shutil.rmtree("files")
        print("Files folder deleted.")
        record_log("Files folder deleted.")
    # Log that the program has been terminated
    record_log("Program terminated.")
    # Add a blank line in the log file
    with open(log_file, "a") as f:
        f.write("\n")
    # Display a polite goodbye message to the user and end the program
    print("Thank you for using the Zip File Manager. Goodbye.")
    sys.exit(0)


# Function to display the main menu and handle user choices
def menu():
    # Clear the terminal for a clean interface (works on Windows and Linux)
    os.system("cls" if os.name == "nt" else "clear")
    # Display the program title and note about logging
    print("=== ZIP FILE MANAGER ===")
    print("All actions are logged in transactions.log")
    # Store the current zip path (None until a file is loaded)
    zip_path = None
    # Keep showing the menu until user decides to quit
    while True:
        # Display the available menu options
        print("\n-- Main Menu --")
        print("[1] Load zip file")
        print("[2] Zip file details")
        print("[3] Zip file contents")
        print("[4] Extract zip file")
        print("[5] Quit")
        # Ask the user for their choice and log it
        choice = input("Select option [1-5]: ").strip()
        record_log(f"Menu selection: {choice}")
        # Run the selected menu option
        if choice == "1":
            zip_path = load_zip()
        elif choice == "2":
            show_info(zip_path)
        elif choice == "3":
            show_files(zip_path)
        elif choice == "4":
            extract_all(zip_path)
        elif choice == "5":
            exit_prog(zip_path)
        else:
            # Warn the user if they enter an invalid menu option
            print("Invalid option. Please select a number from 1 to 5.")


# Main program execution starts here
# Call the menu function to start the interactive program
menu()
