# Auto-generated LGP chromosome
# Generation: 0
# RMSE: 0.56550610
# Fitness: 1.76832753

import math

WORK_REGS = 6
CONST_REGS = [0.0, 1.0, -1.0, 0.5, 2.0, 3.14159, 0.1, 5.0]
OPS = ['+', '-', '*', '/']
CHROMOSOME_LENGTH = 100
INSTR_LEN = 4

CHROMOSOME = [0, 1, 9, 5, 0, 1, 6, 10, 0, 4, 1, 8, 2, 4, 11, 0, 2, 1, 3, 9, 1, 4, 2, 1, 1, 3, 4, 6, 3, 3, 5, 10, 2, 4, 13, 4, 1, 5, 1, 13, 3, 1, 2, 4, 0, 4, 4, 12, 0, 4, 4, 11, 2, 1, 0, 9, 0, 3, 7, 10, 3, 5, 3, 10, 0, 2, 13, 12, 1, 3, 9, 2, 0, 4, 5, 13, 1, 1, 12, 11, 1, 4, 5, 11, 3, 5, 6, 7, 1, 4, 6, 12, 1, 1, 11, 1, 3, 5, 1, 11, 0, 2, 13, 9, 0, 2, 5, 12, 2, 4, 7, 6, 0, 1, 10, 3, 3, 4, 5, 11, 0, 4, 2, 9, 1, 3, 12, 8, 0, 3, 6, 6, 1, 4, 8, 5, 1, 3, 13, 11, 1, 5, 5, 9, 2, 2, 11, 10, 3, 3, 5, 0, 1, 3, 12, 12, 0, 1, 10, 4, 2, 4, 2, 9, 2, 4, 2, 10, 0, 1, 6, 8, 1, 3, 7, 7, 3, 2, 3, 3, 2, 5, 1, 3, 3, 4, 9, 5, 0, 1, 5, 8, 0, 2, 8, 13, 0, 4, 1, 4, 1, 3, 1, 13, 2, 1, 8, 5, 1, 5, 6, 4, 3, 1, 5, 0, 2, 2, 8, 10, 3, 4, 8, 9, 1, 2, 11, 6, 1, 2, 6, 13, 3, 5, 10, 6, 1, 4, 1, 5, 3, 5, 7, 12, 1, 5, 7, 13, 1, 1, 12, 4, 1, 4, 2, 12, 2, 5, 6, 11, 3, 2, 4, 12, 1, 4, 0, 13, 3, 2, 0, 2, 1, 1, 9, 1, 2, 1, 1, 7, 3, 4, 13, 12, 0, 1, 6, 9, 0, 3, 1, 1, 3, 2, 0, 9, 0, 3, 13, 8, 3, 5, 5, 9, 2, 1, 6, 10, 1, 4, 4, 2, 3, 5, 1, 4, 3, 5, 13, 4, 3, 5, 13, 4, 3, 1, 12, 11, 2, 4, 13, 6, 1, 4, 7, 1, 2, 4, 4, 4, 2, 2, 1, 5, 1, 3, 9, 4, 0, 1, 7, 8, 2, 5, 11, 5, 2, 2, 12, 12, 3, 3, 2, 11, 1, 4, 5, 4, 3, 2, 1, 2, 0, 5, 11, 4, 2, 1, 4, 12, 2, 5, 3, 9, 1, 1, 13, 7, 0, 1, 12, 4, 1, 4, 2, 7, 2, 5, 0, 12]

def safe_div(a, b):
    try:
        if not math.isfinite(a):
            a = 0.0
        if not math.isfinite(b) or abs(b) < 1e-9:
            return a
        return a / b
    except Exception:
        return 0.0

def run_function(x):
    regs = [0.0 for _ in range(WORK_REGS)] + CONST_REGS[:]
    regs[0] = float(x)
    for i in range(0, len(CHROMOSOME), INSTR_LEN):
        op_idx, dest, s1, s2 = CHROMOSOME[i:i+INSTR_LEN]
        op_idx = int(op_idx); dest = int(dest); s1 = int(s1); s2 = int(s2)
        a = regs[s1] if 0 <= s1 < len(regs) else 0.0
        b = regs[s2] if 0 <= s2 < len(regs) else 0.0
        op = OPS[op_idx]
        if op == '+':
            regs[dest] = a + b
        elif op == '-':
            regs[dest] = a - b
        elif op == '*':
            regs[dest] = a * b
        elif op == '/':
            regs[dest] = safe_div(a, b)
        if not math.isfinite(regs[dest]):
            regs[dest] = 0.0
    return float(regs[WORK_REGS - 1])

if __name__ == '__main__':
    tests = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    for t in tests:
        print(f'x={t:.3f} -> y={run_function(t):.8f}')
