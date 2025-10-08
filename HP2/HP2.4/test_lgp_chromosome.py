import math
from best_chromosome import CHROMOSOME, VARIABLE_REGS, CONST_VALUES, OPERATORS
import function_data
from typing import List
import sympy as sp
import matplotlib.pyplot as plt

XS, YS = function_data.XS, function_data.YS
x = sp.Symbol('x')
TOTAL_REGS = VARIABLE_REGS + len(CONST_VALUES)

def chromosome_symbolic_expression(chrom) -> sp.Expr:
    regs = [sp.Integer(0)]*VARIABLE_REGS
    regs[0] = x
    for i in range(0, len(chrom), 4):
        op, dest, s1, s2 = chrom[i:i+4]
        a = regs[s1] if s1 < VARIABLE_REGS else CONST_VALUES[s1 - VARIABLE_REGS]
        b = regs[s2] if s2 < VARIABLE_REGS else CONST_VALUES[s2 - VARIABLE_REGS]
        if OPERATORS[op] == '+':
            v = a + b
        elif OPERATORS[op] == '-':
            v = a - b
        elif OPERATORS[op] == '*':
            v = a * b
        else:
            v = a / b
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
        deg = len(coeffs)-1
        parts=[]
        for i,c in enumerate(coeffs):
            power = deg - i
            if c == 0: 
                continue
            term = f"{float(c):.6g}"
            if power >= 1:
                term += "x" if power==1 else f"x^{power}"
            parts.append(term)
        return " + ".join(parts) if parts else "0"
    g_str = f"g(x) = ({coeffs_to_str(a_coeffs)}) / ({coeffs_to_str(b_coeffs)})"
    return g_str, num_poly, den_poly

def evaluate(chrom):
    preds=[]
    for x_val in XS:
        regs=[0.0]*VARIABLE_REGS
        regs[0]=x_val
        for i in range(0,len(chrom),4):
            op,dst,s1,s2=chrom[i:i+4]
            a = regs[s1] if s1 < VARIABLE_REGS else CONST_VALUES[s1-VARIABLE_REGS]
            b = regs[s2] if s2 < VARIABLE_REGS else CONST_VALUES[s2-VARIABLE_REGS]
            if OPERATORS[op] == '+': v=a+b
            elif OPERATORS[op] == '-': v=a-b
            elif OPERATORS[op] == '*': v=a*b
            else:
                v = a if abs(b)<1e-9 else a/b
            regs[dst]=v
        preds.append(regs[0])
    return preds

def rmse(preds):
    return math.sqrt(sum((p-y)**2 for p,y in zip(preds,YS))/len(YS))

if __name__ == "__main__":
    expr = chromosome_symbolic_expression(CHROMOSOME)
    g_str, num_poly, den_poly = rational_polynomial_form(expr)
    preds = evaluate(CHROMOSOME)
    error = rmse(preds)
    print(g_str)
    print(f"RMSE: {error:.6e}")
    plt.scatter(XS, YS, s=20, label="Data")
    plt.plot(XS, preds, 'r-', label="Best fit")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("LGP Fit")
    plt.show()