import math

# Test of encoding and decoding
#
# Encoding: Starting from two weight matrices (W^IH and W^HO, denoted w_ih and w_ho in
#           the Python code), of size Nh x (Ni+1) and No x (Nh+1), respectively,
#           generate a single chromosome, such that the first Nh x (Ni+1) genes are
#           used for obtaining W^IH and the remaining genes for W^HO.
#           The genes should take values in the range [0,1]
#           (e.g. 0.219412987333, 0.452951914512, and so on)
#
#
#######################################################################
#  ToDo: Write the encode_network function:
#  Do NOT change the interface!
#######################################################################


def encode_network(w_ih, w_ho, w_max):
    chromosome = []
    for row in w_ih:
        for w in row:
            chromosome.append((w + w_max) / (2 * w_max))
    for row in w_ho:
        for w in row:
            chromosome.append((w + w_max) / (2 * w_max))

    return chromosome


# Add code here!

#
# Decoding: Starting from a chromosome of length Nh x (Ni+1) + No x (Nh+1), use
#           The first Nh x (Ni+1) genes to obtain W^IH, and the remaining genes
#           to obtain W^OH. The weights and the biases (i.e., the elements of the matrices)
#           should take values in the range [-wMax,wMax], so the alleles (values of the
#           genes in the chromosome) must be rescaled correctly.
#
#
#######################################################################
#  ToDo: Write the decode_chromosome function:
#  Do NOT change the interface!
#######################################################################


def decode_chromosome(chromosome, n_i, n_h, n_o, w_max):
    idx = 0
    w_ih = []
    for _ in range(n_h):
        row = []
        for _ in range(n_i + 1):
            g = chromosome[idx]
            idx += 1
            row.append(g * 2 * w_max - w_max)
        w_ih.append(row)

    w_oh = []
    for _ in range(n_o):
        row = []
        for _ in range(n_h + 1):
            g = chromosome[idx]
            idx += 1
            row.append(g * 2 * w_max - w_max)
        w_oh.append(row)

    return w_ih, w_oh


############################################################
# Main program:
############################################################

# The maximum (absolute) value of weights and biases. Thus, they take values in
# the range [-w_max,w_max]
w_max = 5

# Sample network of size 3-3-2. Note the the number of rows in w_ih MUST be
# equal to the number of columns in w_ho, minus 1; see also the definition of nH below.
#
# Note: Your encoding and decoding methods should work for any values of nI, nH, and nO,
# not just for the example below! Thus, test your encoding and decoding functions by
# defining different set of matrices w_ih and w_ho (fulfilling the criterion on nH, see below)
#
w_ih = [[2, 1, -3, 1], [5, -2, 1, 4], [3, 0, 1, 2]]
w_ho = [[1, 0, -4, 3], [4, -2, 0, 1]]
n_i = len(w_ih[0]) - 1
n_h = len(
    w_ih
)  # % must be equal to len(w_ho[0])-1, for a valid set of matrices for an FFNN
n_o = len(w_ho)

chromosome = encode_network(w_ih, w_ho, w_max)
[new_w_ih, new_w_ho] = decode_chromosome(chromosome, n_i, n_h, n_o, w_max)

error_count = 0
tolerance = 0.00000001
for i in range(n_h):
    for j in range(n_i + 1):
        difference = abs(w_ih[i][j] - new_w_ih[i][j])
        if difference > tolerance:
            print("Error for element " + str(i) + " , " + str(j) + " in wIH")
            error_count += 1

for i in range(n_o):
    for j in range(n_h + 1):
        difference = abs(w_ho[i][j] - new_w_ho[i][j])
        if difference > tolerance:
            print("Error for element " + str(i) + " , " + str(j) + " in wHO")
            error_count += 1

if error_count == 0:
    print("Test OK")
else:
    print("Test failed")
