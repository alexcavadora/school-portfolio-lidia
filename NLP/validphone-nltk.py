# 10 digits mexico
# 9 digits spain
# 10-11 USA

# can include +, -, parenthesis,dots
# no letters or invalid symbols
# national, lofal international
# limit the task for mexico for now

#%%
import nltk
from nltk import CFG

#%%
gramatica_tel = CFG.fromstring("""
digito -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' 
numero -> lada tres cuatro
lada -> '(' digito digito digito ')' | digito digito digito | '(' digito digito digito ')' ' '
tres -> digito digito digito | digito digito digito ' '
cuatro -> digito digito digito digito
""")

parser = nltk.ChartParser(gramatica_tel)

# Input number
numero = "(462) 265 4115"

# Tokenize: keep digits and parentheses
tokens = list(numero)  
print("Tokens:", tokens)

# Parse
for tree in parser.parse(tokens):
    tree.pretty_print()
    print(" ".join(tree.leaves()))
 


# %%
