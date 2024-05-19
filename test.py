def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(",", "").rstrip('0').rstrip('.')



a=maybe_remove_comma('2300.010')
print(a)