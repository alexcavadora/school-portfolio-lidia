
#%%
import nltk
from nltk import CFG

#%%
gramatica = CFG.fromstring("""
oracion -> sujeto predicado
sujeto -> articulo nombre
articulo -> 'el' | 'la'
nombre -> 'perro' | 'gata'
predicado -> verbo adverbio
verbo -> 'corre' | 'come'
adverbio -> 'deprisa' | 'mucho'
""")

parser = nltk.ChartParser(gramatica)

oracion = ['el', 'perro', 'corre', 'deprisa']


oracion2 = ['el', 'corre', 'deprisa']



for tree in parser.parse(oracion):
    tree.pretty_print()
    print(" ".join(tree.leaves()))

# %%

rules = CFG.fromstring("""
asign -> variable equals expression
equals -> '='
expression -> expression operation expression | number | variable
operation -> '+' | '-' | '*'
number -> '0' | '1' | '2' | '3' | '4' | '5'
variable -> 'a' | 'b' | 'x' | 'y'
""")

compiler_parser = nltk.ChartParser(rules)

sentence = 'a = 1 - 2 + 3 + 4'
oracion = sentence.split()

for tree in compiler_parser.parse(oracion):
    tree.pretty_print()
    print(" ".join(tree.leaves()))

# %%
