from backend.tax_engine.old_regime import calculate_old_tax
from backend.tax_engine.new_regime import calculate_new_tax

def suggest_regime(income, d80c, d80d, hra):

    old_tax = calculate_old_tax(income, d80c, d80d, hra)

    new_tax = calculate_new_tax(income)

    if old_tax < new_tax:
        suggestion = "Old Regime"
    else:
        suggestion = "New Regime"

    return {
        "old_regime_tax": old_tax,
        "new_regime_tax": new_tax,
        "suggested_regime": suggestion
    }