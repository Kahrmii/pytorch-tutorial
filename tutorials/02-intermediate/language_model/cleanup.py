import os

def remove_whitespace_lines_from_folder(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    for line in lines:
                        if line.strip():  # Überprüft, ob die Zeile nicht nur aus Leerzeichen besteht
                            outfile.write(line)
                    outfile.write('\n')  # Fügt einen Zeilenumbruch zwischen den Dateien hinzu

if __name__ == "__main__":
    input_folder = 'C:\\Users\\a.frost\\Downloads\\btd'
    output_file = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\tutorials\\02-intermediate\\language_model\\data\\output_file.txt'
    remove_whitespace_lines_from_folder(input_folder, output_file)