def calculate_old_tax(income, deduction_80c, deduction_80d, hra):

    standard_deduction = 50000

    taxable_income = income - standard_deduction - deduction_80c - deduction_80d - hra

    if taxable_income <= 250000:
        tax = 0

    elif taxable_income <= 500000:
        tax = (taxable_income - 250000) * 0.05

    elif taxable_income <= 1000000:
        tax = 12500 + (taxable_income - 500000) * 0.20

    else:
        tax = 112500 + (taxable_income - 1000000) * 0.30

    return max(tax,0)