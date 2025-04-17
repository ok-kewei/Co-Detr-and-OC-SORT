import os

def save_filenames_to_txt(folder_path, output_file="filenames.txt"):
    with open(output_file, "w") as file:
        for filename in os.listdir(folder_path):
            file.write(filename + "\n")
    print(f"Filenames saved to {output_file}")

# Example usage
folder_path = "/home/kewei/rain_data/oxford/10-29/left_undistort/"  # Change this to your actual folder path
save_filenames_to_txt(folder_path)