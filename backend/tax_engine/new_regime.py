def calculate_new_tax(income):

    tax = 0

    if income <= 300000:
        tax = 0

    elif income <= 600000:
        tax = (income - 300000) * 0.05

    elif income <= 900000:
        tax = 15000 + (income - 600000) * 0.10

    elif income <= 1200000:
        tax = 45000 + (income - 900000) * 0.15

    elif income <= 1500000:
        tax = 90000 + (income - 1200000) * 0.20

    else:
        tax = 150000 + (income - 1500000) * 0.30

    return tax