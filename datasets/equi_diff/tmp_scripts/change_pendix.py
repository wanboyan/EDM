import os

# Define the path to the directory containing the files
folder_path = '/data_sata/pack/latent/equi_inv_52499'

# Loop over each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('_equi.npy'):
        # Create the new file name by replacing '_equi.npy' with '.npy'
        new_filename = filename.replace('_equi.npy', '.npy')

        # Create full paths to the old and new files
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{old_file}' to '{new_file}'")
