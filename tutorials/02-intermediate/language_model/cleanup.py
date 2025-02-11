import os
import re

def remove_whitespace_lines_from_folder(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    for line in lines:
                        if line.strip() and not re.search(r'\.{3,}', line) and len(line.strip()) >= 70:
                            outfile.write(line)

if __name__ == "__main__":
    input_folder = 'C:\\Users\\a.frost\\Downloads\\btd'
    output_file = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data\\output_file.txt'
    remove_whitespace_lines_from_folder(input_folder, output_file)