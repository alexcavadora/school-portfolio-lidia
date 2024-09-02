Fundamentos de las bases de datos:
**Atomicity**: This means that each database transaction must be treated as a single, indivisible unit. Either all or none of the changes made by the transaction are applied to the database.
**Consistency**: The database must ensure that the transaction leaves the database in a consistent state. This means that the data is valid and follows the rules defined for it.
For example, if a bank account has $1000 as its balance, withdrawing $500 should leave the balance at $500, not -\$450 (which would be
inconsistent).
**Isolation**: Each transaction must operate independently of other transactions, even if they're executing simultaneously. This means that each transaction sees a consistent view of the database, unaffected by the actions of other transactions.
**Durability**: Once a transaction is committed, its effects must persist even if there's a system failure or crash.

Tipos de bases de datos no relacionales:

trabajando con memoria cache bases de datos redis:
  - key:value
  - almacenan las consultas más utilizadas en la memoria cache.
  - utilizada principalmente porque funciona sobre memoria, no sobre el disco duro.

bases de datos basadas en documentos:
  - MongoDB: utliza JSON

bases de datos basadas en grafos:
  - Neo4j: utiliza grafos

bases de deatos basadas en columnas:
  - Cassandra: utiliza columnas, que se puden escalar horizontalmente
