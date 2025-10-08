import random, math, time, argparse, importlib, os, json
from typing import List, Tuple

# ---------------- Configuration (can be overridden by CLI) ----------------
DEFAULTS = dict(
    POP_SIZE=150,
    MAX_GEN=800,
    TOURNAMENT_SIZE=5,
    MUTATE_GENE_PROB=0.03,
    ADD_INSTR_PROB=0.10,
    DEL_INSTR_PROB=0.08,
    MAX_GENES=400,  # hard limit (genes); 4 genes per instruction -> 100 instructions
    SOFT_LIMIT=300,  # soft penalty threshold (genes)
    SEED=1234,
    VARIABLE_REGS=6,  # first register (index 0) holds x for each data point
    CONST_REGS=6,
    OPERATORS=["+", "-", "*", "/"],
    NO_IMPROVEMENT_PATIENCE=120,
    TARGET_RMSE=1e-10,  # early stop if reached
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
        self.const_values = [random.uniform(-5, 5) for _ in range(cfg.CONST_REGS)]
        self.history = []

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
        n_instr = random.randint(8, 28)
        chrom = []
        for _ in range(n_instr):
            chrom.extend(self.random_instruction())
        return chrom

    # ----- Evaluation -----
    def execute(self, chrom, x_val):
        regs = [0.0] * self.cfg.VARIABLE_REGS
        regs[0] = x_val
        for i in range(0, len(chrom), INSTR_LEN):
            op, dst, a_idx, b_idx = chrom[i : i + 4]
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

    def evaluate(self, chrom) -> Tuple[float, float]:
        # RMSE and fitness
        sse = 0.0
        for x, y in zip(self.XS, self.YS):
            yhat = self.execute(chrom, x)
            diff = yhat - y
            sse += diff * diff
        rmse = math.sqrt(sse / self.K)
        fitness = 1.0 / (rmse + 1e-15)
        # soft length penalty (per extra instruction beyond soft limit)
        if len(chrom) > self.cfg.SOFT_LIMIT:
            over_instr = (len(chrom) - self.cfg.SOFT_LIMIT) // INSTR_LEN + 1
            fitness *= 0.92**over_instr
        return rmse, fitness

    # ----- Selection -----
    def tournament(self, population, fitness_cache):
        best = None
        best_fit = -1.0
        for _ in range(self.cfg.TOURNAMENT_SIZE):
            ind = random.choice(population)
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
    def mutate(self, chrom):
        # gene-wise
        for i in range(0, len(chrom), INSTR_LEN):
            if random.random() < self.cfg.MUTATE_GENE_PROB:
                chrom[i] = random.randrange(len(self.cfg.OPERATORS))
            if random.random() < self.cfg.MUTATE_GENE_PROB:
                chrom[i + 1] = random.randrange(self.cfg.VARIABLE_REGS)
            if random.random() < self.cfg.MUTATE_GENE_PROB:
                chrom[i + 2] = random.randrange(self.total_regs)
            if random.random() < self.cfg.MUTATE_GENE_PROB:
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
        pop = [self.random_chromosome() for _ in range(self.cfg.POP_SIZE)]
        fit_cache = {}
        best = None
        best_fit = -1.0
        best_rmse = float("inf")
        stagnation = 0
        start_time = time.time()
        for gen in range(1, self.cfg.MAX_GEN + 1):
            # evaluate (lazily)
            for ind in pop:
                if id(ind) not in fit_cache:
                    rmse, fit = self.evaluate(ind)
                    fit_cache[id(ind)] = (rmse, fit)
                    if fit > best_fit:
                        best_fit = fit
                        best_rmse = rmse
                        best = ind[:]
                        stagnation = 0
            stagnation += 1
            self.history.append((gen, best_rmse, len(best)))
            if gen == 1 or gen % 25 == 0:
                print(
                    f"Gen {gen:4d}  RMSE={best_rmse:.3e}  len={len(best)//INSTR_LEN:3d} instr  fitness={best_fit:.3e}"
                )
            # early stopping criteria
            if best_rmse <= self.cfg.TARGET_RMSE:
                print(f"Early stop: target RMSE reached at generation {gen}.")
                break
            if stagnation >= self.cfg.NO_IMPROVEMENT_PATIENCE:
                # increase mutation temporarily
                self.cfg.MUTATE_GENE_PROB = min(0.15, self.cfg.MUTATE_GENE_PROB * 1.5)
            else:
                # gentle decay back toward original
                self.cfg.MUTATE_GENE_PROB = max(
                    DEFAULTS["MUTATE_GENE_PROB"], self.cfg.MUTATE_GENE_PROB * 0.98
                )

            # produce next generation
            new_pop = []
            # (1) build offspring via selection + crossover, enforcing length limit
            while len(new_pop) < self.cfg.POP_SIZE:
                p1 = self.tournament(pop, fit_cache)
                p2 = self.tournament(pop, fit_cache)
                c1, c2 = self.crossover(p1, p2)
                if len(c1) <= self.cfg.MAX_GENES:
                    new_pop.append(c1)
                if len(new_pop) < self.cfg.POP_SIZE and len(c2) <= self.cfg.MAX_GENES:
                    new_pop.append(c2)
            # (2) apply mutation to all offspring (length checks inside mutate)
            for child in new_pop:
                self.mutate(child)
            # elitism: ensure best survives (replace worst if lost)
            if best not in new_pop:
                new_pop[0] = best[:]
            pop = new_pop
            fit_cache.clear()

        elapsed = time.time() - start_time
        print(
            f"Finished. Best RMSE={best_rmse:.6e} length={len(best)//INSTR_LEN} instructions. Time {elapsed:.2f}s."
        )
        self.save_best(best, best_rmse)
        if getattr(self.cfg, "save_history", False):
            self.save_history()
        return best, best_rmse, elapsed

    # ----- Saving -----
    def save_best(self, chrom, rmse):
        with open("best_chromosome.py", "w", encoding="utf-8") as f:
            f.write("# Auto-generated best chromosome file\n")
            f.write(f"VARIABLE_REGS = {self.cfg.VARIABLE_REGS}\n")
            f.write(f"CONST_VALUES = {repr(self.const_values)}\n")
            f.write(f"OPERATORS = {repr(self.cfg.OPERATORS)}\n")
            f.write(
                f"CHROMOSOME = {repr(chrom)}  # flat list; 4 genes per instruction\n"
            )
            f.write(f"BEST_RMSE = {rmse}\n")
        print("Saved best chromosome to best_chromosome.py")

    def save_history(self):
        with open("evolution_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f)
        print("Saved evolution history.")


# ---------------- Entry ----------------
def main():
    args = parse_args()
    # freeze defaults into args namespace
    cfg = argparse.Namespace(
        **{
            k: getattr(args, k.lower()) if hasattr(args, k.lower()) else v
            for k, v in DEFAULTS.items()
        }
    )
    cfg.save_history = args.save_history
    lgp = LGP(cfg)
    lgp.evolve()


if __name__ == "__main__":
    main()
