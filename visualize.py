import sys
from utils import draw_genotype
import numpy as np
import genotypes
import os
from PyPDF2 import PdfFileMerger

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    if os.path.exists('./visualization/{}'.format(genotype_name)):
        for i in os.listdir('./visualization/{}'.format(genotype_name)):
            os.remove(os.path.join('./visualization/{}'.format(genotype_name), i))
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)
    
    if not os.path.exists('./visualization/{}'.format(genotype_name)):  
        os.mkdir('./visualization/{}'.format(genotype_name))

    draw_genotype(genotype.normal, np.max(list(genotype.normal_concat) + list(genotype.reduce_concat)) -1, "./visualization/{}/{}_normal".format(genotype_name, genotype_name), genotype.normal_concat)
    draw_genotype(genotype.reduce, np.max(list(genotype.normal_concat) + list(genotype.reduce_concat)) -1, "./visualization/{}/{}_reduction".format(genotype_name, genotype_name), genotype.reduce_concat)

    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(os.getcwd(), "./visualization/{}/{}_normal.pdf".format(genotype_name, genotype_name)))
    file_merger.append(os.path.join(os.getcwd(), "./visualization/{}/{}_reduction.pdf".format(genotype_name, genotype_name)))

    file_merger.write(os.path.join(os.getcwd(), "./visualization/{}/{}.pdf".format(genotype_name, genotype_name)))


