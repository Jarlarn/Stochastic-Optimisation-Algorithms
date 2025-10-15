def encode_network(w_ih, w_ho, w_max):
    chromosome = []
    for row in w_ih:
        for w in row:
            chromosome.append((w + w_max) / (2 * w_max))
    for row in w_ho:
        for w in row:
            chromosome.append((w + w_max) / (2 * w_max))

    return chromosome


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


w_max = 10

w_ih = [[2, 1, -3, 1], [5, -2, 1, 4], [3, 0, 1, 2]]
w_ho = [[1, 0, -4, 3], [4, -2, 0, 1]]
n_i = len(w_ih[0]) - 1
n_h = len(w_ih)
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
