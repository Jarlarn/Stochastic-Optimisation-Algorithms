import random, math, time, argparse, importlib, os, json, glob
from typing import List, Tuple

# ---------------- Configuration (can be overridden by CLI) ----------------
DEFAULTS = dict(
    POP_SIZE=800,  # Increased from 600
    MAX_GEN=3000,  # Increased from 1500
    TOURNAMENT_SIZE=4,  # Reduced from 7
    MUTATE_GENE_PROB=0.04,
    ADD_INSTR_PROB=0.12,
    DEL_INSTR_PROB=0.10,
    MAX_GENES=600,
    SOFT_LIMIT=400,
    SEED=1234,
    VARIABLE_REGS=6,
    CONST_REGS=8,  # Increased from 6
    OPERATORS=["+", "-", "*", "/"],
    TARGET_RMSE=1e-2,
    RESTART_AFTER_STAGNATION=500,  # New parameter
)

INSTR_LEN = 4
PROTECTED_DIV_EPS = 1e-12


# ---------------- Utility ----------------
def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        t = type(v)
        if t is list:
            continue
        p.add_argument(f"--{k.lower()}", type=t, default=v)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save_history", action="store_true")
    return p.parse_args()


def mutation_probability_change(
    previous_mutation_probability,
    all_maximum_fitnesses,
    number_of_instructions,
    time_window_size,
    probability_change_factor=2.0,
):
    initial_mutation_probability = 1.0 / (number_of_instructions * 4)
    if len(all_maximum_fitnesses) < time_window_size:
        new_mutation_probability = previous_mutation_probability
    else:
        fitness_difference = (
            all_maximum_fitnesses[-1] - all_maximum_fitnesses[-time_window_size]
        )
        if fitness_difference > 1e-6:
            new_mutation_probability = (
                previous_mutation_probability
                / probability_change_factor  # Changed to division
            )
        elif fitness_difference < 1e-12:
            new_mutation_probability = initial_mutation_probability * 1.5
        else:
            new_mutation_probability = initial_mutation_probability

    # Bound mutation probability to reasonable range
    return max(0.01, min(0.2, new_mutation_probability))


# ---------------- LGP Components ----------------
class LGP:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.SEED)
        # load data
        data_module = importlib.import_module("function_data")
        if hasattr(data_module, "XS") and hasattr(data_module, "YS"):
            self.XS = list(data_module.XS)
            self.YS = list(data_module.YS)
        elif hasattr(data_module, "load_function_data"):
            pairs = data_module.load_function_data()
            self.XS = [p[0] for p in pairs]
            self.YS = [p[1] for p in pairs]
        else:
            raise AttributeError(
                "function_data.py must define XS & YS or load_function_data()"
            )
        assert len(self.XS) == len(self.YS) > 0, "Empty dataset."
        self.K = len(self.XS)
        self.total_regs = cfg.VARIABLE_REGS + cfg.CONST_REGS
        # freeze constant registers
        random.seed(cfg.SEED + 999)
        # Replace the constant initialization in __init__
        self.const_values = []
        # Include values likely to appear in rational functions
        special_values = [
            0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            3.0,
            -3.0,
            0.25,
            -0.25,
            4.0,
            -4.0,
            0.1,
            10.0,
        ]
        for val in special_values[: min(len(special_values), cfg.CONST_REGS)]:
            self.const_values.append(val)
        # Fill remaining slots with random values
        remaining = cfg.CONST_REGS - len(self.const_values)
        if remaining > 0:
            self.const_values.extend([random.uniform(-5, 5) for _ in range(remaining)])
        self.history = []

    def _validate_chromosome(self, chrom):
        """Validate chromosome structure and indices."""
        if len(chrom) % INSTR_LEN != 0:
            raise ValueError(
                f"Chromosome length ({len(chrom)}) must be multiple of {INSTR_LEN}"
            )

        total_regs = self.total_regs
        for i in range(0, len(chrom), INSTR_LEN):
            op, dest, s1, s2 = chrom[i : i + INSTR_LEN]
            if not (0 <= op < len(self.cfg.OPERATORS)):
                raise ValueError(f"Operator index out of range at position {i}: {op}")
            if not (0 <= dest < self.cfg.VARIABLE_REGS):
                raise ValueError(
                    f"Destination register out of range at position {i}: {dest}"
                )
            if not (0 <= s1 < total_regs):
                raise ValueError(
                    f"Source register 1 out of range at position {i}: {s1}"
                )
            if not (0 <= s2 < total_regs):
                raise ValueError(
                    f"Source register 2 out of range at position {i}: {s2}"
                )
        return True

    # ----- Chromosome representation -----
    def random_instruction(self):
        op = random.randrange(len(self.cfg.OPERATORS))
        dest = random.randrange(
            self.cfg.VARIABLE_REGS
        )  # only variable regs as destinations
        s1 = random.randrange(self.total_regs)
        s2 = random.randrange(self.total_regs)
        return [op, dest, s1, s2]

    def random_chromosome(self):
        # Create diverse initial population
        if random.random() < 0.3:
            n_instr = random.randint(5, 15)  # Small programs
        elif random.random() < 0.6:
            n_instr = random.randint(15, 30)  # Medium programs
        else:
            n_instr = random.randint(30, 50)  # Large programs
        chrom = []
        for _ in range(n_instr):
            chrom.extend(self.random_instruction())
        return chrom

    # ----- Evaluation -----
    def execute(self, chrom, x_val):
        try:
            self._validate_chromosome(chrom)
            regs = [0.0] * self.cfg.VARIABLE_REGS
            regs[0] = x_val
            for i in range(0, len(chrom), INSTR_LEN):
                op, dst, a_idx, b_idx = chrom[i : i + INSTR_LEN]
                a = (
                    regs[a_idx]
                    if a_idx < self.cfg.VARIABLE_REGS
                    else self.const_values[a_idx - self.cfg.VARIABLE_REGS]
                )
                b = (
                    regs[b_idx]
                    if b_idx < self.cfg.VARIABLE_REGS
                    else self.const_values[b_idx - self.cfg.VARIABLE_REGS]
                )
                code = self.cfg.OPERATORS[op]
                if code == "+":
                    v = a + b
                elif code == "-":
                    v = a - b
                elif code == "*":
                    v = a * b
                else:
                    # protected division
                    if abs(b) < PROTECTED_DIV_EPS:
                        v = a
                    else:
                        v = a / b
                regs[dst] = v
            return regs[0]
        except Exception as e:
            print(f"Execution error: {e}")
            return float("nan")

    def evaluate(self, chrom) -> Tuple[float, float]:
        try:
            # RMSE and fitness
            sse = 0.0
            for x, y in zip(self.XS, self.YS):
                yhat = self.execute(chrom, x)
                if not math.isfinite(yhat):  # Check for NaN/Infinity
                    return float("inf"), -1.0
                diff = yhat - y
                sse += diff * diff

            # Accept perfect fits! Only reject non-finite SSE
            if not math.isfinite(sse):
                return float("inf"), -1.0

            rmse = math.sqrt(sse / self.K)
            # Handle perfect fit specially
            if rmse == 0.0:
                fitness = float("inf")  # Represent perfect fit with infinite fitness
            else:
                fitness = 1.0 / (rmse + 1e-15)

            # soft length penalty
            if len(chrom) > self.cfg.SOFT_LIMIT and not math.isinf(fitness):
                over_instr = (len(chrom) - self.cfg.SOFT_LIMIT) // INSTR_LEN + 1
                fitness *= 0.92**over_instr

            return rmse, fitness
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float("inf"), -1.0

    def quick_evaluate(self, chrom, sample_size=20):
        try:
            # Validate the chromosome first
            self._validate_chromosome(chrom)

            # Use random subset of data for initial screening
            indices = random.sample(range(self.K), min(sample_size, self.K))
            # Only calculate full fitness for promising solutions
            sse = 0.0
            for i in indices:
                x = self.XS[i]
                y = self.YS[i]
                yhat = self.execute(chrom, x)
                if not math.isfinite(yhat):  # Check for NaN/Infinity
                    return float("inf"), -1.0
                diff = yhat - y
                sse += diff * diff

            # Accept perfect fits! Only reject non-finite SSE
            if not math.isfinite(sse):
                return float("inf"), -1.0

            rmse = math.sqrt(sse / len(indices))
            # Handle perfect fit specially
            if rmse == 0.0:
                fitness = float("inf")  # Represent perfect fit with infinite fitness
            else:
                fitness = 1.0 / (rmse + 1e-15)

            return rmse, fitness
        except Exception as e:
            print(f"Quick evaluation error: {e}")
            return float("inf"), -1.0

    # ----- Selection -----
    def tournament(self, population, fitness_cache):
        best = None
        best_fit = -1.0
        for _ in range(self.cfg.TOURNAMENT_SIZE):
            ind = random.choice(population)

            # Evaluate if not already in cache
            if id(ind) not in fitness_cache:
                # First do a quick check with fewer data points
                quick_rmse, quick_fit = self.quick_evaluate(ind)
                # Relaxed threshold to 10.0 (from 1.0) to avoid rejecting potentially good solutions
                if quick_rmse < 10.0:  # Only fully evaluate promising solutions
                    rmse, fit = self.evaluate(ind)
                else:
                    rmse, fit = quick_rmse, quick_fit
                fitness_cache[id(ind)] = (rmse, fit)

            fit = fitness_cache[id(ind)][1]
            if fit > best_fit:
                best_fit = fit
                best = ind
        return best

    # ----- Crossover (two point, instruction aligned) -----
    def crossover(self, a, b):
        if a is b or len(a) <= INSTR_LEN or len(b) <= INSTR_LEN:
            return a[:], b[:]
        na = len(a) // INSTR_LEN
        nb = len(b) // INSTR_LEN
        c1a = random.randint(0, na - 1)
        c2a = random.randint(c1a, na - 1)
        c1b = random.randint(0, nb - 1)
        c2b = random.randint(c1b, nb - 1)
        seg_a = a[c1a * INSTR_LEN : (c2a + 1) * INSTR_LEN]
        seg_b = b[c1b * INSTR_LEN : (c2b + 1) * INSTR_LEN]
        child1 = a[: c1a * INSTR_LEN] + seg_b + a[(c2a + 1) * INSTR_LEN :]
        child2 = b[: c1b * INSTR_LEN] + seg_a + b[(c2b + 1) * INSTR_LEN :]
        return child1, child2

    # ----- Mutation -----
    def mutate(self, chrom, mutation_prob):
        # gene-wise
        for i in range(0, len(chrom), INSTR_LEN):
            if random.random() < mutation_prob:
                chrom[i] = random.randrange(len(self.cfg.OPERATORS))
            if random.random() < mutation_prob:
                chrom[i + 1] = random.randrange(self.cfg.VARIABLE_REGS)
            if random.random() < mutation_prob:
                chrom[i + 2] = random.randrange(self.total_regs)
            if random.random() < mutation_prob:
                chrom[i + 3] = random.randrange(self.total_regs)
        # structural
        if (
            random.random() < self.cfg.ADD_INSTR_PROB
            and len(chrom) + INSTR_LEN <= self.cfg.MAX_GENES
        ):
            # insert at random position
            pos_instr = random.randint(0, len(chrom) // INSTR_LEN)
            chrom[pos_instr * INSTR_LEN : pos_instr * INSTR_LEN] = (
                self.random_instruction()
            )
        if random.random() < self.cfg.DEL_INSTR_PROB and len(chrom) > 2 * INSTR_LEN:
            n_instr = len(chrom) // INSTR_LEN
            pos = random.randrange(n_instr)
            del chrom[pos * INSTR_LEN : (pos + 1) * INSTR_LEN]

    # ----- Evolution Loop -----
    def evolve(self):
        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints_current", exist_ok=True)

        population = [self.random_chromosome() for _ in range(self.cfg.POP_SIZE)]
        fitness_cache = {}
        gen = 0
        stagnation = 0
        start_time = time.time()
        best_fit = -1.0
        best_rmse = float("inf")
        best_chrom = None
        all_maximum_fitnesses = []
        mutation_prob = self.cfg.MUTATE_GENE_PROB
        time_window_size = 20

        while True:
            gen += 1
            fitness_cache.clear()
            fits = []
            rmses = []
            for chrom in population:
                rmse, fit = self.evaluate(chrom)
                fitness_cache[id(chrom)] = (rmse, fit)
                fits.append(fit)
                rmses.append(rmse)

            max_fit = max(fits)
            min_rmse = min(rmses)
            all_maximum_fitnesses.append(max_fit)

            if max_fit > best_fit:
                best_fit = max_fit
                best_rmse = min_rmse
                best_chrom = population[fits.index(max_fit)][:]
                stagnation = 0
                self.save_best(best_chrom, best_rmse)
            else:
                stagnation += 1

            # Restart if stagnant for too long
            if stagnation >= self.cfg.RESTART_AFTER_STAGNATION:
                print(f"Restarting after {stagnation} generations of stagnation...")
                # Keep 10% of current population
                elite_size = self.cfg.POP_SIZE // 10
                elite = []
                combined = list(zip(population, fits))
                combined.sort(key=lambda x: x[1], reverse=True)
                elite = [chrom for chrom, _ in combined[:elite_size]]

                # Generate new random population
                new_pop = [
                    self.random_chromosome()
                    for _ in range(self.cfg.POP_SIZE - elite_size)
                ]
                population = elite + new_pop
                stagnation = 0
                mutation_prob = (
                    self.cfg.MUTATE_GENE_PROB * 2
                )  # Higher mutation after restart
                fitness_cache.clear()

            # Checkpoint every 1000 generations
            if gen % 1000 == 0 and best_chrom is not None:
                checkpoint_file = f"checkpoints_current/best_chromosome_gen{gen}.py"
                self.save_best(best_chrom, best_rmse, checkpoint_file)
                print(f"Checkpoint saved to {checkpoint_file}")

                # Delete checkpoints older than 10,000 generations
                cutoff_gen = gen - 10000
                if cutoff_gen > 0:
                    old_checkpoints = glob.glob(
                        "checkpoints_current/best_chromosome_gen*.py"
                    )
                    for old_file in old_checkpoints:
                        try:
                            # Extract generation number from filename
                            file_gen = int(old_file.split("gen")[1].split(".py")[0])
                            if file_gen < cutoff_gen:
                                os.remove(old_file)
                                print(f"Removed old checkpoint: {old_file}")
                        except (ValueError, IndexError):
                            # Skip files that don't match the expected pattern
                            continue

            if gen == 1 or gen % 25 == 0:
                print(
                    f"Gen {gen:4d}  RMSE={best_rmse:.3e}  len={len(best_chrom)//INSTR_LEN if best_chrom else 0:3d} instr  fitness={best_fit:.3e}  stagnation={stagnation}"
                )

            if best_rmse <= self.cfg.TARGET_RMSE:
                print(f"Early stop: target RMSE reached at generation {gen}.")
                break

            # Adaptive mutation probability
            mutation_prob = mutation_probability_change(
                mutation_prob,
                all_maximum_fitnesses,
                number_of_instructions=len(population[0]) // INSTR_LEN,
                time_window_size=time_window_size,
                probability_change_factor=2.0,
            )

            # Selection and reproduction
            new_population = []
            while len(new_population) < self.cfg.POP_SIZE:
                p1 = self.tournament(population, fitness_cache)
                p2 = self.tournament(population, fitness_cache)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1, mutation_prob)
                self.mutate(c2, mutation_prob)
                new_population.append(c1)
                if len(new_population) < self.cfg.POP_SIZE:
                    new_population.append(c2)
            population = new_population

        elapsed = time.time() - start_time
        print(
            f"Finished. Best RMSE={best_rmse:.6e} length={len(best_chrom)//INSTR_LEN if best_chrom else 0} instructions. Time {elapsed:.2f}s."
        )
        self.save_best(best_chrom, best_rmse)

        if getattr(self.cfg, "save_history", False):
            self.save_history()

        return best_chrom, best_rmse, elapsed

    # ----- Saving -----
    def save_best(self, chrom, rmse, filename="checkpoints_current/best_chromosome.py"):
        # Extract directory and ensure it exists
        if "/" in filename or "\\" in filename:
            directory = os.path.dirname(filename)
            os.makedirs(directory, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Auto-generated best chromosome file\n")
            f.write(f"VARIABLE_REGS = {self.cfg.VARIABLE_REGS}\n")
            f.write(f"CONST_VALUES = {repr(self.const_values)}\n")
            f.write(f"OPERATORS = {repr(self.cfg.OPERATORS)}\n")
            f.write(
                f"CHROMOSOME = {repr(chrom)}  # flat list; 4 genes per instruction\n"
            )
            f.write(f"BEST_RMSE = {rmse}\n")
            print(f"Saved best chromosome to {filename}")

    def save_history(self):
        with open("evolution_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f)
        print("Saved evolution history.")


# ---------------- Entry ----------------
def main():
    args = parse_args()
    best_overall = None
    best_rmse_overall = float("inf")

    # Try multiple seeds
    seeds = [42, 5678, 9012, 3456, 7890]
    for seed in seeds:
        print(f"\n=== Running with seed {seed} ===")
        # freeze defaults into args namespace
        cfg = argparse.Namespace(
            **{
                k: getattr(args, k.lower()) if hasattr(args, k.lower()) else v
                for k, v in DEFAULTS.items()
            }
        )
        cfg.SEED = seed
        cfg.save_history = args.save_history
        lgp = LGP(cfg)
        best, rmse, _ = lgp.evolve()

        if rmse < best_rmse_overall:
            best_rmse_overall = rmse
            best_overall = best
            # Save this as the ultimate best
            lgp.save_best(best, rmse)

    print(f"Overall best RMSE across all seeds: {best_rmse_overall:.6e}")


if __name__ == "__main__":
    main()
