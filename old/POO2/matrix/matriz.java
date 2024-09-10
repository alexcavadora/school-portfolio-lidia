import java.util.Scanner;

public class matriz
{
	public static void main(String[] args)
	{
		byte seleccion = 0;
		Scanner scanf = new Scanner(System.in);
		/*do
		{
			System.out.println("\nMenu:");
			System.out.println("1. Suma");
			System.out.println("2. Resta");
			System.out.println("3. Fibonacci (n)");
			System.out.println("4. Solución cuadrática");
			System.out.println("5. Salir");
			seleccion = scanf.nextByte();

			switch (seleccion)
			{
				case 0:
					break;
				case 1:
					break;
				case 2:
					break;
				case 3:
					break;
				case 4:
					break;
			}

		} while (seleccion != 5);*/

		Matrix a = new Matrix("A");
		Matrix b = new Matrix("B");
		a.initialize(3, 3, 5.1f);
		b.initialize(3, 3, 3.41f);

		System.out.print(a.add(b));
	}
}

class Matrix
{
	private int x = 0;
	private int y = 0;
	private float [][]matrix = {{0}};

	public String name = "";

	Matrix(String txt)
	{
		name = txt;
	}

	Matrix(int width, int height)
	{
		if (width < 0 && height < 0)
			return;
		matrix = new float[width][height];
		x = width;
		y = height;

		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] = 0;
	}

	Matrix(int width, int height, float starting_value)
	{
		if (width < 0 && height < 0)
			return;
		matrix = new float[width][height];
		x = width;
		y = height;

		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] = starting_value;
	}

	public void initialize(int width, int height)
	{
		if (width < 0 && height < 0)
			return;
		matrix = new float[width][height];
		x = width;
		y = height;

		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] = 0;
	}

	public void initialize(int width, int height, float starting_value)
	{
		if (width < 0 && height < 0)
			return;
		matrix = new float[width][height];
		x = width;
		y = height;

		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] = starting_value;
	}

	public void printMatrix()
	{
		System.out.println("Matrix " + name + ":");
		if (x == 0 && y == 0)
			System.out.println(matrix[0][0]);
		for(int i = 0; i < x; i++)
		{
			System.out.print("{");
			for(int j = 0; j < y; j++)
			{
				System.out.print(matrix[i][j]);
				if(j != y-1)
					System.out.print(", ");
			}
			System.out.println("}");
		}
	}

	private boolean isDimensionEqual(Matrix m)
	{
		if(x != m.x)
		{
			System.out.print("There's a difference in x dimension.");
		}
		if(y != m.y)
		{
			System.out.print("There's a difference in y dimension.");
		}
		if(x != m.x || y != m.y)
		{
			return false;
		}
		return true;
	}

	public float[][] add(Matrix m)
	{
		if(!isDimensionEqual(m))
			return null;
		if(x == 0 && y != 0)
		{
			for (int j = 0; j < y; j++)
				matrix[0][j] += m.matrix[0][j];
			return matrix;
		}
		if(y == 0 && x != 0)
		{
			for (int i = 0; i < x; i++)
				matrix[i][0] += m.matrix[i][0];
			return matrix;
		}
		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] += m.matrix[i][j];
		return matrix;
	}

	public float[][] substract(Matrix m)
	{

		if(!isDimensionEqual(m))
			return null;
		if(y == 0 && x != 0)
		{
			for (int i = 0; i < x; i++)
				matrix[i][0] -= m.matrix[i][0];
			return matrix;
		}
		for(int i = 0; i < x; i++)
			for(int j = 0; j < y; j++)
				matrix[i][j] -= m.matrix[i][j];
		return matrix;
	}

	public Matrix multiply(Matrix other)
	{
		if(other.x != m.y || other.y != m.x)
			return null;
		Matrix temp = new Matrix(m.x, other.y);
		for (int i = 0; i < m.y; i++)
			for(int j = 0; j < m-x; j++)
				for(int k = 0; j < )
				temp[j][i] +=  m[]
		return temp;
	}
}