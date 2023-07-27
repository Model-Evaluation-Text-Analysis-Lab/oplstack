# Camelot

'''import os
import numpy as np
import camelot

def extract_tables_from_pdf(pdf_path, output_dir):
    # Read the PDF file
    tables = camelot.read_pdf(pdf_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Go through each table one at a time
    for i, table in enumerate(tables):
        # Convert the table to a pandas DataFrame
        df = table.df

        # Convert the DataFrame to a 2D NumPy array
        np_array = df.to_numpy()

        # Save the 2D NumPy array as a .npy file
        output_path = os.path.join(output_dir, f'table_{i}.npy')
        np.save(output_path, np_array)

        print(f'Saved table {i} to {output_path}')

# Call the function with the paths to your PDF file and output directory

path_to_your_pdf_file = 'preprocessing_pipeline/documents/gzip.pdf'
path_to_your_output_dir = 'preprocessing_pipeline/output_files/PDF/tables'
extract_tables_from_pdf(path_to_your_pdf_file, path_to_your_output_dir)'''


import os
import numpy as np
import pdfplumber

def extract_tables_from_pdf(pdf_path, output_dir):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Go through each page one at a time
        for i, page in enumerate(pdf.pages):
            # Extract the tables from the page
            tables = page.extract_tables()

            # For each table
            for j, table in enumerate(tables):
                # Convert the table to a 2D NumPy array
                np_array = np.array(table)

                # Save the 2D NumPy array as a .npy file
                output_path = os.path.join(output_dir, f'table_{i}_{j}.npy')
                np.save(output_path, np_array)

                print(f'Saved table {j} of page {i} to {output_path}')

# Call the function with the paths to your PDF file and output directory
path_to_your_pdf_file = 'preprocessing_pipeline/documents/gzip.pdf'
path_to_your_output_dir = 'preprocessing_pipeline/output_files/PDF/tables'
extract_tables_from_pdf(path_to_your_pdf_file, path_to_your_output_dir)


# READ PDF

'''import numpy as np
import os
import pandas as pd


def print_all_tables_from_dir(output_dir):
    # Get a list of all .npy files in the output directory
    npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]

    # For each .npy file
    for npy_file in npy_files:
        # Construct the full path to the .npy file
        npy_path = os.path.join(output_dir, npy_file)

        # Load the .npy file into a numpy array
        table = np.load(npy_path, allow_pickle=True)

        # Convert the numpy array to a pandas DataFrame for prettier printing
        df = pd.DataFrame(table)

        # Print the table
        print(f'Table from file {npy_file}:')
        print(df)
        print('\n' + '='*50 + '\n')

# Call the function with the path to your output directory
path_to_your_output_dir = 'preprocessing_pipeline/output_files/PDF/tables'
print_all_tables_from_dir(path_to_your_output_dir)
'''