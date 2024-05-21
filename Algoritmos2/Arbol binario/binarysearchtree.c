/*
Author: Alejandro Alonso Sánchez
Subject: global_tree Binario de Búsqueda (Binary Search Tree)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TREENODE {
  int key;
  char name[50];  // Additional field for name
  struct TREENODE *left, *right;
} TREENODE;

TREENODE* createTreeNode(int key, char* name);
void insertNode(TREENODE** tree, int key, char* name);
void showTree(TREENODE* tree); // Consider in-order traversal for sorted output
void inOrder(TREENODE* tree);
TREENODE* searchByKey(TREENODE* tree, int key);
TREENODE* searchByName(TREENODE* tree, char* name);
void deleteTree(TREENODE** tree);
TREENODE* findSmallest(TREENODE* tree);
TREENODE* findLargest(TREENODE* tree);
void deleteNode(TREENODE** tree, int key);

typedef struct REGISTROS {
  char ClaveUDA[10];
  char NombreUDA[60];
  char TipoUDA[80];
  char AreaUDA[80];
  unsigned int NumCred;
  unsigned int Hrs;
} REGISTROS;
REGISTROS* GetRegistro(int indice, char filename[]);

REGISTROS* GetRegistro(int indice, char filename[]) {
  // Open the file in read mode
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("Error opening file: %s\n", filename);
    return NULL;
  }

  char line[256]; // max line size
  int recordCount = 0; // Keep track of the current record number

  // Read lines from the file and count records
  while (fgets(line, sizeof(line), file) != NULL)
  {
    recordCount++;
  }

  // Close the file temporarily to reset the file pointer
  fclose(file);

  // Check if the requested index (indice) is within the valid range
  if (indice < 1 || indice > recordCount) {
    printf("Invalid indice: %d. Records range from 1 to %d.\n", indice, recordCount);
    return NULL;
  }

  // Reopen the file in read mode
  file = fopen(filename, "r");
  if (file == NULL) {
    printf("Error opening file: %s\n", filename);
    return NULL;
  }

  // Seek to the beginning of the requested record using fseek
  int recordSize = 256; // Assuming each record is 256 bytes (adjust based on your file format)
  long offset = (indice - 1) * recordSize;
  if (fseek(file, offset, SEEK_SET) != 0) {
    printf("Error seeking in file.\n");
    fclose(file);
    return NULL;
  }

  // Read the data from the current record
  REGISTROS* registro = (REGISTROS*)malloc(sizeof(REGISTROS));
  if (registro == NULL) {
    printf("Failed to allocate memory.\n");
    fclose(file);
    return NULL;
  }

  // Assuming fields are separated by commas
  fscanf(file, "%s\n%s\n%s\n%s\n%u\n%u\n", registro->ClaveUDA, registro->NombreUDA, registro->TipoUDA, registro->AreaUDA, &registro->NumCred, &registro->Hrs);

  // Close the file and return the registro object
  fclose(file);
  return registro;
}

// Improved ReadDataFromFile function with error handling and name extraction
void ReadDataFromFile(TREENODE** tree, char filename[]) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("Error opening file: %s\n", filename);
    return;
  }

  char line[80]; // Increased size to accommodate potential longer lines
  int key;
  char name[50];

  // Loop for the known number of lines (72 in this case)
  for (int i = 0; i < 72; i++) {
    if (fgets(line, sizeof(line), file) == NULL) {
      // Handle unexpected end of file
      printf("Warning: Expected 72 lines in file, but reached end of file prematurely.\n");
      break;
    }

    if (ferror(file)) {
      printf("Error reading file.\n");
      break;
    }

    // Extract key and name from the line (assuming specific format)
    sscanf(line, "%d %s", &key, name);

    // Assuming GetRegistro uses the extracted key and filename to get full data
    REGISTROS* registro = GetRegistro(key, filename);
    if (registro != NULL) {
      // Insert full data (key, name from registro, and other fields) into the tree
      insertNode(tree, key, registro->NombreUDA);  // Use name from registro
    } else {
      printf("Error retrieving registro for key %d\n", key);
    }
  }

  fclose(file);
  printf("\nArchivo leido!\n");
}

// Function to create a new tree node with memory allocation
TREENODE* createTreeNode(int key, char* name) {
  TREENODE* ptr = (TREENODE*)malloc(sizeof(TREENODE));
  if (ptr == NULL) {
    printf("Failed to allocate memory.\n");
    exit(1);
  }
  ptr->key = key;
  strcpy(ptr->name, name);
  ptr->left = NULL;
  ptr->right = NULL;
  return ptr;
}

// Function to insert a node into the BST
void insertNode(TREENODE** tree, int key, char* name) {
  if (*tree == NULL)
    *tree = createTreeNode(key, name);
  else if (key < (*tree)->key)
    insertNode(&(*tree)->left, key, name);
  else
    insertNode(&(*tree)->right, key, name);
}

// Print the tree in-order traversal
void inOrder(TREENODE* tree)
{
    if (tree == NULL)
        return;
    inOrder(tree->left);
    printf("Clave UDA: %d\n", tree->key); // Assuming key is stored in tree->key
    printf("Nombre UDA: %s\n", tree->name); //jejecambio
    printf("Tipo UDA: %s\n", tree->registro->TipoUDA);
    printf("Area UDA: %s\n", tree->registro->AreaUDA);
    printf("Numero de creditos: %u\n", tree->registro->NumCred);
    printf("Horas: %u\n\n", tree->registro->Hrs);
    inOrder(tree->right);
}

// Function to show the tree using in-order traversal (prints sorted data)
void showTree(TREENODE* tree) {
  inOrder(tree);
}

// Function to search for a node by key
TREENODE* searchByKey(TREENODE* tree, int key) {
  if (tree == NULL || tree->key == key)
    return tree;
  else if (key < tree->key)
    return searchByKey(tree->left, key);
  else
    return searchByKey(tree->right, key);
}

// Function to search for a node by name
TREENODE* searchByName(TREENODE* tree, char* name) {
  if (tree == NULL)
    return NULL;
  else if (strcmp(name, tree->name) == 0)
    return tree;
  else {
    TREENODE* found = searchByName(tree->left, name);
    if (found != NULL)
      return found;
    else
      return searchByName(tree->right, name);
  }
}

// Function to delete a node with memory deallocation
void deleteNode(TREENODE** tree, int key) {
  if (*tree == NULL)
    return;

  TREENODE* parent = NULL;
  TREENODE* target = *tree;

  while (target->key != key) {
    parent = target;
    if (key < target->key) {
      target = target->left;
    } else {
      target = target->right;
    }

    if (target == NULL) {
      printf("Node with key %d not found.\n", key);
      return;
    }
  }

  // Case 1: Node has no children (leaf node)
  if (target->left == NULL && target->right == NULL) {
    if (parent == NULL) {
      *tree = NULL; // Delete the root node
    } else if (parent->left == target) {
      parent->left = NULL;
    } else {
      parent->right = NULL;
    }
    free(target);
    return;
  }

  // Case 2: Node has one child
  TREENODE* child;
  if (target->left != NULL) {
    child = target->left;
  } else {
    child = target->right;
  }

  if (parent == NULL) {
    *tree = child;
  } else if (parent->left == target) {
    parent->left = child;
  } else {
    parent->right = child;
  }
  free(target);

  if (child->right != NULL) {
    TREENODE* successor = child->right;
    while (successor->left != NULL) {
      successor = successor->left;
    }

    int tempKey = target->key;
    char tempName[50];
    strcpy(tempName, target->name);
    target->key = successor->key;
    strcpy(target->name, successor->name);
    successor->key = tempKey;
    strcpy(successor->name, tempName);
    deleteNode(&(child->right), successor->key);
  }
}

// Function to find the smallest node (leftmost)
TREENODE* findSmallest(TREENODE* tree) {
  if (tree == NULL || tree->left == NULL)
    return tree;
  else
    return findSmallest(tree->left);
}

// Function to find the largest node (rightmost)
TREENODE* findLargest(TREENODE* tree) {
  if (tree == NULL || tree->right == NULL)
    return tree;
  else
    return findLargest(tree->right);
}

// Function to delete the entire tree with memory deallocation
void deleteTree(TREENODE** tree) {
  if (*tree == NULL)
    return;

  deleteTree(&(*tree)->left);
  deleteTree(&(*tree)->right);
  free(*tree);
  *tree = NULL;
}

int main() {
  TREENODE* tree = NULL; // Initialize the tree as NULL

  int choice, key;
  char name[50];

  while (1) {
    printf("\nMenu:\n");
    printf("1. Read data from file\n");
    printf("2. Insert data\n");
    printf("3. Search by key\n");
    printf("4. Search by name\n");
    printf("5. Delete a node\n");
    printf("6. Show tree (sorted order)\n");
    printf("7. Find smallest element\n");
    printf("8. Find largest element\n");
    printf("9. Delete entire tree\n");
    printf("10. Exit\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    switch (choice) {
      case 1:
        ReadDataFromFile(&tree, "listadoNoOrden.txt");
        break;
      case 2:
        printf("Enter key: ");
        scanf("%d", &key);
        printf("Enter name: ");
        scanf(" %s", name); // Include a space to handle leading whitespace
        insertNode(&tree, key, name);
        break;
      case 3:
        printf("Enter key to search: ");
        scanf("%d", &key);
        TREENODE* found = searchByKey(tree, key);
        if (found != NULL) {
          printf("Node found (key: %d, name: %s)\n", found->key, found->name);
        } else {
          printf("Node not found.\n");
        }
        break;
      case 4:
        printf("Enter name to search: ");
        scanf(" %s", name); // Include a space to handle leading whitespace
        found = searchByName(tree, name);
        if (found != NULL) {
          printf("Node found (key: %d, name: %s)\n", found->key, found->name);
        } else {
          printf("Node not found.\n");
        }
        break;
      case 5:
        printf("Enter key to delete: ");
        scanf("%d", &key);
        deleteNode(&tree, key);
        break;
      case 6:
        printf("Tree (sorted order):\n");
        showTree(tree);
        break;
      case 7:
        found = findSmallest(tree);
        if (found != NULL) {
          printf("Smallest element (key: %d, name: %s)\n", found->key, found->name);
        } else {
          printf("Tree is empty.\n");
        }
        break;
      case 8:
        found = findLargest(tree);
        if (found != NULL) {
          printf("Largest element (key: %d, name: %s)\n", found->key, found->name);
        } else {
          printf("Tree is empty.\n");
        }
        break;
      case 9:
        deleteTree(&tree);
        printf("Tree deleted.\n");
        break;
      case 10:
        exit(0);
      default:
        printf("Invalid choice.\n");
    }
  }

  return 0;
}


