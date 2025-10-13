import math
from best_chromosome import CHROMOSOME, VARIABLE_REGS, CONST_VALUES, OPERATORS
import function_data
from typing import List
import sympy as sp
import matplotlib.pyplot as plt
import run_lgp

# replace direct access with a robust loader (works whether function_data exposes XS/YS or provides loader)
if hasattr(function_data, "XS") and hasattr(function_data, "YS"):
    XS = list(function_data.XS)
    YS = list(function_data.YS)
elif hasattr(function_data, "load_function_data"):
    _pairs = function_data.load_function_data()
    XS = [p[0] for p in _pairs]
    YS = [p[1] for p in _pairs]
else:
    raise AttributeError("function_data must define XS & YS or load_function_data()")

x = sp.Symbol("x")
TOTAL_REGS = VARIABLE_REGS + len(CONST_VALUES)
PROTECTED_DIV_EPS = getattr(run_lgp, "PROTECTED_DIV_EPS", 1e-12)
INSTR_LEN = 4


def validate_chromosome(chrom):
    if len(chrom) % INSTR_LEN != 0:
        raise ValueError("CHROMOSOME length must be multiple of 4")
    total = TOTAL_REGS
    for i in range(0, len(chrom), INSTR_LEN):
        op, dest, s1, s2 = chrom[i : i + INSTR_LEN]
        if not (0 <= op < len(OPERATORS)):
            raise ValueError(
                f"Operator index out of range at instr {i//INSTR_LEN}: {op}"
            )
        if not (0 <= dest < VARIABLE_REGS):
            raise ValueError(
                f"Destination register out of range at instr {i//INSTR_LEN}: {dest}"
            )
        if not (0 <= s1 < total) or not (0 <= s2 < total):
            raise ValueError(
                f"Source register index out of range at instr {i//INSTR_LEN}: {s1}, {s2}"
            )


def chromosome_symbolic_expression(chrom) -> sp.Expr:
    validate_chromosome(chrom)
    regs = [sp.Integer(0)] * VARIABLE_REGS
    regs[0] = x
    for i in range(0, len(chrom), INSTR_LEN):
        op, dest, s1, s2 = chrom[i : i + INSTR_LEN]
        a = regs[s1] if s1 < VARIABLE_REGS else CONST_VALUES[s1 - VARIABLE_REGS]
        b = regs[s2] if s2 < VARIABLE_REGS else CONST_VALUES[s2 - VARIABLE_REGS]
        code = OPERATORS[op]
        if code == "+":
            v = a + b
        elif code == "-":
            v = a - b
        elif code == "*":
            v = a * b
        else:
            # protected symbolic division: handle numeric and symbolic zero safely
            try:
                is_number = getattr(b, "is_Number", False) or isinstance(
                    b, (int, float)
                )
                if is_number:
                    try:
                        b_val = float(b)
                    except Exception:
                        b_val = None
                    if b_val is None or abs(b_val) < PROTECTED_DIV_EPS:
                        v = a
                    else:
                        v = a / b
                else:
                    # for symbolic expressions, treat exact zero as protected
                    if sp.simplify(b) == 0:
                        v = a
                    else:
                        v = a / b
            except ZeroDivisionError:
                v = a
            except Exception:
                # fallback protection for unexpected symbolic issues
                v = a
        regs[dest] = sp.simplify(v)
    return sp.simplify(regs[0])


def rational_polynomial_form(expr: sp.Expr):
    frac = sp.together(expr)
    num, den = sp.fraction(frac)
    num = sp.expand(num)
    den = sp.expand(den)
    num_poly = sp.Poly(num, x)
    den_poly = sp.Poly(den, x)
    a_coeffs = num_poly.all_coeffs()  # highest -> lowest
    b_coeffs = den_poly.all_coeffs()

    def coeffs_to_str(coeffs):
        deg = len(coeffs) - 1
        parts = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if c == 0:
                continue
            term = f"{float(c):.6g}"
            if power >= 1:
                term += "x" if power == 1 else f"x^{power}"
            parts.append(term)
        return " + ".join(parts) if parts else "0"

    g_str = f"g(x) = ({coeffs_to_str(a_coeffs)}) / ({coeffs_to_str(b_coeffs)})"
    return g_str, num_poly, den_poly


def evaluate(chrom):
    validate_chromosome(chrom)
    preds = []
    for x_val in XS:
        regs = [0.0] * VARIABLE_REGS
        regs[0] = x_val
        for i in range(0, len(chrom), INSTR_LEN):
            op, dst, s1, s2 = chrom[i : i + INSTR_LEN]
            a = regs[s1] if s1 < VARIABLE_REGS else CONST_VALUES[s1 - VARIABLE_REGS]
            b = regs[s2] if s2 < VARIABLE_REGS else CONST_VALUES[s2 - VARIABLE_REGS]
            if OPERATORS[op] == "+":
                v = a + b
            elif OPERATORS[op] == "-":
                v = a - b
            elif OPERATORS[op] == "*":
                v = a * b
            else:
                v = a if abs(b) < PROTECTED_DIV_EPS else a / b
            regs[dst] = v
        preds.append(regs[0])
    return preds


def rmse(preds):
    return math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, YS)) / len(YS))


if __name__ == "__main__":
    expr = chromosome_symbolic_expression(CHROMOSOME)
    g_str, num_poly, den_poly = rational_polynomial_form(expr)
    preds = evaluate(CHROMOSOME)
    error = rmse(preds)
    print(g_str)
    print(f"RMSE: {error:.6e}")
    # sort for plotting the fitted curve
    pairs = sorted(zip(XS, preds))
    xs_sorted, preds_sorted = zip(*pairs)
    plt.scatter(XS, YS, s=20, label="Data")
    plt.plot(xs_sorted, preds_sorted, "r-", label="Best fit")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("LGP Fit")
    plt.show()
