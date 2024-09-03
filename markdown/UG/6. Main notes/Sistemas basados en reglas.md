[[Clases de 5to semestre]] [[notas  de clases]]
# Fundamentos de los sistemas basados en reglas
La representación en espacio de estados es útil aún en proglemas no muy bien estructurados y forma la base de la mayoría de los métodos clásicos de la IA. Términos utilizados comunmente en esta clase:

- **Problemática** : Es un conjunto de información que el agente utiliza para decidir lo que hará
- **Estado inicial**: Situación de inicio en la que se encuentra el agente.
- **Operador**: Es la descripción de una acción en función de la cual se alcanzará un estado, al emprender una acción en particular.
- **Espacios de estados del problema**: Es el conjunto de todos los estados que pueden alcanzarse a partir del estado inicial, mediante cualquier secuencia de acciones (RUTA).
- **Prueba de meta**: Se aplica al estado para decidir si se trata de un estado.
- **Costo de la ruta:** Una función mediante la cual se asigna un costo a una ruta determinada.
	- COSTO TOTAL = COSTO DE LA RUTA *(solución óptima)* + COSTO DE LA BÚSQUEDA
- **Solución:** La salida 
- **Abstracción:** Es el proceso  de eliminación de detalles en una representación.
---
La búsqueda por un espacio de estados, nos permite 4 cosas:
- Nos permite una definición formal, así como afrontar la necesidad de convertir una situación dada en otra deseada mediante operaciones permitidas. 
- Nos permite definir el proceso de solución como una combinación de técnicas y búsquedas (útiles si no hay una técnica directa).

==Ejemplo==:
Imagina que se tienen 2 jarras, tenemos una de 4L y otra de 3 litros. Si deseamos tener exactamente 2L en la primera jarra solamente utilizando estos 2 contenedores, de qué manera podríamos hacerlo?
## Definición del problema:
- Los movimientos válidos son:
	- Llenar las jarras
		- `if (x,y) and x < 4 => (4,`
	- Vaciar las jarras
	- Vertir el contenido de una jarra a otra
Espacio de estados del problema:
- La dimensión del espacio cartesiano sería el espacio muestral de (x, y)
- x E {0, 1, 2, 3, 4};
- y E {0, 1 , 2}
Estado inicial
- (x0, y0) = (0, 0)
Meta:
- (xi, yi) = (2, ?)

4, 0
1, 3
1, 0
4, 1
2, 3