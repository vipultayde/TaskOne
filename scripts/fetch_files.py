import os
import sys

def fetch_all_files(directory):
    """
    Recursively fetch all files in the given directory, handling errors.
    """
    all_files = []
    errors = []

    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Check if file is accessible (optional, but can catch some errors)
                    if os.path.exists(file_path):
                        all_files.append(file_path)
                except (OSError, PermissionError) as e:
                    errors.append(f"Error accessing file {file_path}: {str(e)}")
            # Handle directory errors if needed, but os.walk handles most
    except (OSError, PermissionError) as e:
        errors.append(f"Error walking directory {directory}: {str(e)}")

    return all_files, errors

if __name__ == "__main__":
    project_dir = "c:/ENGINEERING/Projects/TaskOne"  # Current working directory
    files, errs = fetch_all_files(project_dir)

    print("All files in the project directory:")
    for f in files:
        print(f)

    if errs:
        print("\nErrors encountered:")
        for e in errs:
            print(e)
    else:
        print("\nNo errors encountered.")
