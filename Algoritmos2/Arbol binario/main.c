/*
Author: Alejandro Alonso Sánchez
Subject: Arbol binario
*/
#include <stdio.h>
#include <stdlib.h>

//NODE STRUCTURE
typedef struct TREENODE
{
	int data;
	struct TREENODE *leftson, *rightson;
}
TREENODE;


//FUNCTION PROTOTYPES
TREENODE* createTreeNode(int data);
void insertNode(int data, TREENODE **tree);
void showTree(TREENODE *tree);
void posOrder(TREENODE *tree);
void inOrder(TREENODE *tree);
void preOrder(TREENODE *tree);
TREENODE * searchData(int data, TREENODE *tree);
void deleteTree(TREENODE **tree);
TREENODE *findSmallest(TREENODE *tree);
TREENODE *findLargest(TREENODE *tree);
void deleteNode(int data, TREENODE *tree);

int main()
{
	TREENODE *arbol = NULL;
	insertNode(11, &arbol);
	insertNode(3,&arbol);
	insertNode(50, &arbol);
	insertNode(-7, &arbol);
	insertNode(8, &arbol);
	insertNode(13, &arbol);
	insertNode(40, &arbol);
	insertNode(0, &arbol);
	insertNode(3, &arbol);


	printf("Smallest node in general: %i\n", (findSmallest(arbol))->data);

	printf("Largest node in general: %i\n", (findLargest(arbol))->data);

	printf("Printing tree in-order:\n");
	inOrder(arbol);
	printf("\n");
	deleteNode(11, arbol);
	printf("\n");
	printf("Printing tree in-order:\n");
	inOrder(arbol);
	return 0;
}


//MEMORY ALLOCATION
TREENODE* createTreeNode(int data) //Create a tree node 
{
	TREENODE * ptr;
	ptr = (TREENODE *) malloc(sizeof(TREENODE));
	if(ptr == NULL)
	{
		printf("Failed to allocate memory");
		exit(1);
	}
	ptr->data = data;
	ptr->leftson = NULL;
	ptr->rightson = NULL;
	return ptr;
}

void posOrder(TREENODE *tree)
{
	if(tree == NULL)
	{
		printf("•");				   		// check if given tree is empty 
		return;
	
	}	
	printf("{");
	posOrder(tree->leftson);
	posOrder(tree->rightson);
	printf("}");
	printf(" %i ", tree->data);
}

void inOrder(TREENODE *tree)
{
	if(tree == NULL)
	{
		printf("~");				   		// check if given tree is empty 
		return;
	}	
	printf("{");
	inOrder(tree->leftson);
	printf(" %i ", tree->data);
	inOrder(tree->rightson);
	printf("}");
}

void preOrder(TREENODE *tree)
{
	if(tree == NULL)
	{
		printf("•");				   		// check if given tree is empty 
		return;
	}	
	printf(" %i ", tree->data);
	printf("{");
	preOrder(tree->leftson);
	preOrder(tree->rightson);
	printf("}");
}

//INSERT DATA INTO NODES, CAN'T ACCOUNT FOR FULL NODES THO
void insertNode(int data, TREENODE **tree)
{
	if(*tree == NULL)
	{
		TREENODE *newNode = createTreeNode(data);
		*tree = newNode;
	} 					   		// check if given tree is empty 
		
	else								   		// it contains a root node
		if(data < (*tree)->data) 	   		// try to insert at left son
			insertNode(data, &(*tree)->leftson);
		else									// insert at right son
			insertNode(data, &(*tree)->rightson);
}

//PRINTS A TREE NODE AND BRANCHES IF EXIST

void showTree(TREENODE *tree)
{
	if(tree==NULL)
		return;

	printf("[");

	//if(tree->leftson != NULL)
		showTree(tree->leftson);

	printf("%i", tree->data);

	//if(tree->rightson != NULL)
		showTree(tree->rightson);

	printf("]");
}


TREENODE* searchData(int data, TREENODE *tree)
{
    if(tree != NULL)
    {
        if(data == tree->data)
            return tree;
        else
        {
            if(data < tree->data)
                return searchData(data, tree->leftson);
            else
                return searchData(data, tree->rightson);
        }
    }
    else
        return NULL;
}

void deleteTree(TREENODE **tree)
{
    if(*tree == NULL)
        return;
    else
    {
        if((*tree)->leftson != NULL)
            deleteTree(&(*tree)->leftson);

        if((*tree)->rightson != NULL)
            deleteTree(&(*tree)->rightson);

        printf("\nDeleted node: %i\n", (*tree)->data);
        free(*tree);
        *tree = NULL;
    }
}



TREENODE *findSmallest(TREENODE *tree)
{
    if(tree->leftson != NULL)
    	return findSmallest(tree->leftson);
    else
        return tree;
}

TREENODE *findLargest(TREENODE *tree)
{
    if(tree->rightson != NULL)
    	return findLargest(tree->rightson);
    else
        return tree;
}

void deleteNode(int data, TREENODE *tree)
{
	if(tree == NULL)
		printf("Cannot delete data %i, it was not found.\n", data);
	else if(data < tree->data)
		deleteNode(data, tree->leftson);
	else if (data > tree->data)
		deleteNode(data, tree->rightson);
	else
	{
		printf("Found data %i, proceeding for deletion.\n", tree->data);
		if(tree->leftson != NULL && tree->rightson!= NULL)
		{
			TREENODE* predecessor = (findLargest(tree->leftson));
			printf("This is a case where the node to delete has two children.\n");
			printf("%i is predecessor of data %i.\n", predecessor -> data, tree->data);
			//printf("%i is successor of data %i.\n", (findSmallest(tree)) -> data, tree->data);
			tree->data = predecessor -> data;
			deleteNode(predecessor->data, tree->leftson);
		}
		else
		{

		}

	}
}

unsigned int nodeAmount(TREENODE *tree)
{
	if(tree == NULL)
		return 0;

	else
		return 1 + nodeAmount(tree->leftson) + nodeAmount(tree->rightson);
}

unsigned int leafAmount(TREENODE *tree)
{
    if(tree == NULL)
        return 0;
    if((tree->leftson == NULL) && (tree->rightson == NULL))
        return 1;
    else
        return leafAmount(tree->leftson) + leafAmount(tree->rightson);
}

unsigned int branchAmount(TREENODE *tree)
{
    if(tree == NULL)
        return 0;
    else
        return nodeAmount(tree) - leafAmount(tree) - 1;
}