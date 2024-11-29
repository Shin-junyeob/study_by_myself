from Bio import SeqIO

sff_file = ['./16s_raw_102', './16s_raw_111', './16s_raw_125', './16s_raw_130']

for i in sff_file:
    input_file = i + '.sff'
    output_file =i + '.fastq'

    SeqIO.convert(input_file, 'sff', output_file, 'fastq')
    print(f'파일 변환 완료: {output_file}')