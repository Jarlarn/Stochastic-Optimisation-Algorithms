import random
import function_data
import math

# --- Load data from function_data.py ---
_samples = function_data.load_function_data()
XS = [x for x, y in _samples]
YS = [y for x, y in _samples]

# === Parameters ===
POP_SIZE = 250
CHROMOSOME_LENGTH = 100
MAX_GENES = 4 * CHROMOSOME_LENGTH
M = 4
N = 4
CROSSOVER_RATE = 0.8

# === Hybrid Adaptive Control Parameters ===
HIGH_ERROR_THRESHOLD = 0.20
LOW_ERROR_THRESHOLD = 0.01
MUTATION_AT_HIGH_ERROR = 6.0
MUTATION_AT_LOW_ERROR = 0.5
STAGNATION_THRESHOLD = 75
HEAT_UP_MULTIPLIER = 1.5
COOL_DOWN_MULTIPLIER = 0.9
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.001

OPS = ["+", "-", "*", "/"]
OP_TO_IDX = {op: i for i, op in enumerate(OPS)}
IDX_TO_OP = {i: op for i, op in enumerate(OPS)}
CONSTANTS = [1.0, -1.0, 2.0, 0.0]  # CONSTANT_VALUES

MAX_ERROR = 1e10  # Cap for error to avoid overflow


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None
        self.error = None

    def evaluate(self):
        self.error = self.compute_error()
        self.fitness = 1.0 / (self.error + 1e-8)

    def compute_error(self):
        K = len(XS)
        total = 0.0
        for i in range(K):
            x = XS[i]
            y = YS[i]
            y_hat = execute_chromosome(self.chromosome, x)
            if not math.isfinite(y_hat):
                return MAX_ERROR
            diff = y_hat - y
            squared = diff * diff
            if not math.isfinite(squared) or abs(squared) > MAX_ERROR:
                return MAX_ERROR
            total += squared
        err = math.sqrt(total / K)
        if not math.isfinite(err) or err > MAX_ERROR:
            return MAX_ERROR
        return err


def random_gene(gene_idx):
    if gene_idx % 4 == 0:  # Operator
        return random.randint(0, 3)
    elif gene_idx % 4 == 1:  # Destination register
        return random.randint(0, M - 1)
    else:  # Operand register
        return random.randint(0, M + N - 1)


def random_chromosome():
    length = random.randint(10, 30)
    chromosome = []
    for i in range(length * 4):
        chromosome.append(random_gene(i))
    return chromosome


def decode_instruction(chromosome, idx):
    base = idx * 4
    op = IDX_TO_OP[chromosome[base]]
    dest = chromosome[base + 1]
    src1 = chromosome[base + 2]
    src2 = chromosome[base + 3]
    return (op, dest, src1, src2)


def execute_chromosome(chromosome, x):
    regs = [0.0] * M
    regs[0] = x
    const_regs = CONSTANTS
    num_instr = len(chromosome) // 4
    for idx in range(num_instr):
        op, dest, src1, src2 = decode_instruction(chromosome, idx)

        # Fetch operand values as in run_lgp_1.py
        def get_operand_value(reg_index):
            if reg_index < M:
                return regs[reg_index]
            else:
                return const_regs[reg_index - M]

        a = get_operand_value(src1)
        b = get_operand_value(src2)
        try:
            if op == "+":
                regs[dest] = a + b
            elif op == "-":
                regs[dest] = a - b
            elif op == "*":
                regs[dest] = a * b
            elif op == "/":
                regs[dest] = a / b if abs(b) > 1e-6 else 1.0
        except:
            regs[dest] = 0.0
    return regs[0]


def two_point_crossover(parent1, parent2):
    chrom1 = parent1.chromosome
    chrom2 = parent2.chromosome
    instr1 = len(chrom1) // 4
    instr2 = len(chrom2) // 4
    if instr1 < 2 or instr2 < 2 or random.random() > CROSSOVER_RATE:
        return Individual(chrom1[:]), Individual(chrom2[:])
    # Choose crossover points from 1 to len-1
    p1, p2 = sorted(random.sample(range(1, instr1), 2))
    q1, q2 = sorted(random.sample(range(1, instr2), 2))
    child1_chrom = chrom1[: p1 * 4] + chrom2[q1 * 4 : q2 * 4] + chrom1[p2 * 4 :]
    child2_chrom = chrom2[: q1 * 4] + chrom1[p1 * 4 : p2 * 4] + chrom2[q2 * 4 :]
    return Individual(child1_chrom), Individual(child2_chrom)


def tournament(pop, k=5):
    candidates = random.sample(pop, k)
    candidates.sort(key=lambda ind: ind.fitness, reverse=True)
    for candidate in candidates:
        if random.random() < 0.8:
            return candidate
    return candidates[-1]


def mutate_with_rate(individual, mutation_rate):
    chrom = individual.chromosome
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            gene_pos = i % 4
            if gene_pos == 0:  # Operator
                chrom[i] = random.randint(0, 3)
            elif gene_pos == 1:  # Destination register
                chrom[i] = random.randint(0, M - 1)
            else:  # Operand register
                chrom[i] = random.randint(0, M + N - 1)
    return individual  # For compatibility with rest of code


def save_best_chromosome(individual):
    with open("best_chromosome_4.py", "w") as f:
        f.write("BEST_CHROMOSOME = " + repr(individual.chromosome) + "\n")
        f.write("CONSTANTS = " + repr(CONSTANTS) + "\n")
        f.write("M = " + repr(M) + "\n")
        f.write("N = " + repr(N) + "\n")


TARGET_RMSE = 0.01  # Set your desired RMSE target


def main():
    population = [Individual(random_chromosome()) for _ in range(POP_SIZE)]
    for ind in population:
        ind.evaluate()
    best_ind = max(population, key=lambda ind: ind.fitness)
    best_fit = best_ind.fitness
    best_error = best_ind.error

    # --- Adaptive mutation control variables ---
    adaptive_multiplier = 1.0
    generations_without_improvement = 0
    best_overall_ind = best_ind

    gen = 0
    while best_error > TARGET_RMSE:
        # 1. Determine the GLOBAL base rate based on current quality
        if best_error > HIGH_ERROR_THRESHOLD:
            global_base_rate = MUTATION_AT_HIGH_ERROR
        elif best_error < LOW_ERROR_THRESHOLD:
            global_base_rate = MUTATION_AT_LOW_ERROR
        else:
            # Linear interpolation between high and low
            progress = (HIGH_ERROR_THRESHOLD - best_error) / (
                HIGH_ERROR_THRESHOLD - LOW_ERROR_THRESHOLD
            )
            global_base_rate = MUTATION_AT_HIGH_ERROR - progress * (
                MUTATION_AT_HIGH_ERROR - MUTATION_AT_LOW_ERROR
            )

        # 2. Check for stagnation/progress to adjust the LOCAL multiplier
        improvement_threshold = (
            best_overall_ind.fitness * SIGNIFICANT_IMPROVEMENT_THRESHOLD
        )
        if best_fit > best_overall_ind.fitness + improvement_threshold:
            # PROGRESS - cool down the adaptive multiplier
            adaptive_multiplier = max(0.5, adaptive_multiplier * COOL_DOWN_MULTIPLIER)
            best_overall_ind = best_ind
            generations_without_improvement = 0
            current_mutation_base = global_base_rate * adaptive_multiplier
            print(
                f"Gen {gen}:  NEW BEST!  Fitness = {best_fit:.6f}, "
                f"Error = {best_error:.6f}, "
                f"Length = {len(best_ind.chromosome)//4}, "
                f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                f"Final Mutation Base = {current_mutation_base:.3f}"
            )
        else:
            # STAGNATION - heat up the adaptive multiplier
            generations_without_improvement += 1
            if generations_without_improvement > STAGNATION_THRESHOLD:
                adaptive_multiplier = min(2.0, adaptive_multiplier * HEAT_UP_MULTIPLIER)
                generations_without_improvement = 0
                current_mutation_base = global_base_rate * adaptive_multiplier
                print(
                    f"Gen {gen}: Stagnation detected. "
                    f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                    f"Heating up to {current_mutation_base:.3f}"
                )

        # 3. Calculate the FINAL mutation base for this generation
        current_mutation_base = global_base_rate * adaptive_multiplier

        # Print periodic status updates
        if gen % 20 == 0:
            print(
                f"Gen {gen}: Error = {best_error:.6f}, "
                f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                f"Final Mutation Base = {current_mutation_base:.3f}"
            )

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        new_pop = [population[0]]  # elitism

        while len(new_pop) < POP_SIZE:
            parent1 = tournament(population)
            parent2 = tournament(population)

            # Only perform crossover with probability 0.8, else clone parents
            if random.random() < 0.8:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1 = Individual(parent1.chromosome[:])
                child2 = Individual(parent2.chromosome[:])

            # Discard children with more than 100 instructions
            if (len(child1.chromosome) // 4) > 100 or (
                len(child2.chromosome) // 4
            ) > 100:
                continue

            # Use the current, decaying mutation rate
            p_mutation1 = current_mutation_base / (len(child1.chromosome) + 1)
            p_mutation2 = current_mutation_base / (len(child2.chromosome) + 1)
            child1 = mutate_with_rate(child1, p_mutation1)
            child2 = mutate_with_rate(child2, p_mutation2)

            child1.evaluate()
            child2.evaluate()

            new_pop.append(child1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(child2)

        population = new_pop
        best_ind = max(population, key=lambda ind: ind.fitness)
        best_fit = best_ind.fitness
        best_error = best_ind.error
        gen += 1

    print("\nOptimization finished.")
    print(f"Best fitness: {best_fit:.6f}")
    print(f"Best RMSE: {best_error:.6f}")
    print(f"Generations: {gen}")
    save_best_chromosome(best_ind)


if __name__ == "__main__":
    main()
