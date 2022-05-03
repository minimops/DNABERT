import sys

import numpy

print(repr(sys.argv))


list_name = ["a", "b", "c", "d"]
list_seq = ["".join(str(x) for x in numpy.repeat("A", 130)),
            "".join(str(x) for x in numpy.repeat("T", 30)),
            "".join(str(x) for x in numpy.repeat("G", 155)),
            "".join(str(x) for x in numpy.repeat("C", 4))]

list_seq2 = ["".join(str(x) for x in numpy.arange(130)),
            "".join(str(x) for x in numpy.arange(30)),
            "".join(str(x) for x in numpy.arange(155)),
            "".join(str(x) for x in numpy.arange(4))]

for i in range(len(list_name)):
    ofile = open("DNABERT/data_testing_stuff/fasta_test_files/fasta" +str(i)+ ".fasta", "w")
    ofile.write(">" + list_name[i] + "\n" + list_seq[i] + "\n")
    ofile.close()