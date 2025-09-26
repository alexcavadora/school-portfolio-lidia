
#%%
import nltk
from nltk import CFG

#%%
gramatica = CFG.fromstring("""
oracion -> sujeto predicado | sujeto accion
sujeto -> articulo nombre
articulo -> 'el' | 'la'
nombre -> 'niño' | 'niña' | 'gato'
accion -> verbo sujeto
predicado -> verbo adverbio
verbo -> 've' | 'persigue'
adverbio -> 'deprisa' | 'mucho'
""")

parser = nltk.ChartParser(gramatica)

oracion = ['el', 'niño', 'persigue', 'el', 'gato']



for tree in parser.parse(oracion):
    tree.pretty_print()
    print(" ".join(tree.leaves()))
    

# %%
