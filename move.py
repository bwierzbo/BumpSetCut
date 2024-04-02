import os
import shutil

def combine_folders(main_directories, combined_directory):
    # Ensure the combined 'ball' and 'not_ball' folders exist
    os.makedirs(os.path.join(combined_directory, 'ball'), exist_ok=True)
    os.makedirs(os.path.join(combined_directory, 'notball'), exist_ok=True)

    # Initialize counters for naming conflicts
    ball_count = 0
    not_ball_count = 0

    for main_dir in main_directories:
        for category in ['ball', 'notball']:
            # Path to the specific category in the current main directory
            path = os.path.join(main_dir, category)
            
            for filename in os.listdir(path):
                # Source path of the current file
                src = os.path.join(path, filename)

                # New filename to avoid naming conflicts
                if category == 'ball':
                    new_filename = f"ball-{ball_count}.jpg"
                    ball_count += 1
                else:
                    new_filename = f"notball-{not_ball_count}.jpg"
                    not_ball_count += 1

                # Destination path for the new file
                dest = os.path.join(combined_directory, category, new_filename)

                # Copy the file
                shutil.copy2(src, dest)

# Example usage
main_directories = ['images/back', 'images/Langle', 'images/Rangle', 'images/side']
combine_folders(main_directories, 'images')
