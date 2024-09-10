/*
	Alejandro Alonso Sánchez 
	23/02/2024

	Program 3. Array management in java.
*/
import java.util.Scanner;
public class Arreglos
{
	public static void main(String[] args)
	{
		Scanner scanner = new Scanner(System.in);
		float[] y; // reference variable does 
				   // not need initialization
		System.out.print("Number of elements to capture: ");
		int n = scanner.nextInt();
		y = fill(n);
		printVector(y, n);
		System.out.println("average = " + average(y, n));

		System.out.println("Matrix now.");
		System.out.println("Number of rows: ");
		int rows = scanner.nextInt();
		System.out.println("Number of columns: ");
		int cols = scanner.nextInt();
		float[][] z;

		z = fillMatrix(rows, cols);

		printMatrix(z, rows, cols);
	}

	public static float[] fill(int n)
	{
		Scanner scanner = new Scanner(System.in);
		float[] x = new float[n];
		for (int i = 0; i < n; i++)
		{
			System.out.print("x[" + i + "] = ");
			x[i] = scanner.nextFloat();
		}
		System.out.println();
		return x;
	}

	public static void printVector(float[] x, int n)
	{
		for (int i = 0; i < n; i++)
			System.out.println("x[" + i + "] = " + x[i]);
		System.out.println();
	}

	public static float average(float[] x, int n)
	{
		float average = 0;
		for (int i = 0; i < n; i++)
			average += x[i];
		return average/n;
	}

	public static float [][] fillMatrix(int rows, int cols)
	{
		Scanner scanf = new Scanner(System.in);
		float [][] matrix = new float[rows][cols];

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
			{
				System.out.print("M["+i+"]["+j+"] = ");
				matrix[i][j] = scanf.nextFloat();
			}
		return matrix;
	}

	public static void printMatrix(float[][] matrix, int rows, int cols)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				System.out.print(matrix[i][j]+"\t");
			System.out.println();
		}
	}


}