def generate_dicts(input_file, output_file1, output_file2):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file1, 'w', encoding='utf-8') as outfile1, \
         open(output_file2, 'w', encoding='utf-8') as outfile2:

        for i, line in enumerate(infile):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            entity_id = parts[0]
            entity_name = parts[1].split(',')[0]

            outfile1.write(f"{i}\t{entity_name}\n")
            outfile2.write(f"{i}\t{entity_id}\n")

# 入力ファイルと出力ファイルのパス
input_file = './data/WN18RR/entity2text.txt'
output_file1 = 'entities1.dict'
output_file2 = 'entities2.dict'

# 辞書ファイルを生成
generate_dicts(input_file, output_file1, output_file2)
