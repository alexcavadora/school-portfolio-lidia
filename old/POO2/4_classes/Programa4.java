/*
Program 4. Creating an instanciating classes and objects.
		   Using classes for managing vectors.
*/

import java.util.Scanner;

class Vector
{	
	//attributes or member variables
	private float []dataArray;
	private int n; // size

	//methods or member functions
	Vector() //default constructor if not added, the compiler will initialize them anyways.
	{
		n = 0;
		dataArray = null;
	}

	Vector(int size)
	{
		if (size < 0)
			return;
		n = size;
		dataArray = new float[n];
	}

	Vector(int size, float fill_value)
	{
		if (size < 0)
			return;
		n = size;
		dataArray = new float[size];

		for (int i = 0; i < n; i++)
			dataArray[i] = fill_value;
	}

	Vector(float[] data)
	{
		n = data.length;
		dataArray = new float[n];
		dataArray = data;
	}

	public void printData()
	{
		System.out.print("{");

		for (int i = 0; i < n; i++)
		{
			System.out.print(dataArray[i]);
			if (i != n-1)
				System.out.print(", ");
		}

		System.out.println("}");
	}

	public void getData()
	{
		Scanner scanner = new Scanner(System.in);
		System.out.println("Capturing data, for array of size "+ n + ".");
		for( int i = 0; i < n; i++)
		{
			System.out.print("v["+i+"] = ");
			dataArray[i] = scanner.nextFloat();
		}
		System.out.println("Data captured correctly.");
	}

	public void getData(int size)
	{
		if (size < 0)
			return;
		n = size;
		dataArray = new float[n];
		Scanner scanner = new Scanner(System.in);
		System.out.println("Capturing data, for array of size "+ n + ".");
		for( int i = 0; i < n; i++)
		{
			System.out.print("v["+i+"] = ");
			dataArray[i] = scanner.nextFloat();
		}
		System.out.println("Data captured correctly.");
	}

	public void updateSize(int size)
	{
		if(size < 0 || size == n)
			return;

		float []datos = new float[size];
		for (int i = 0; i < size; i++)
		{
			datos[i] = dataArray[i];
		}
		dataArray = datos;
		n = size;
	}

	public void copyData(float[] data)
	{
		n = data.length;
		dataArray = new float[n];
		dataArray = data;
	}

	public int getSize()
	{
		return n;
	}

	public float[] readData()
	{
		return dataArray;
	}
}

class Programa4
{
	public static void main(String[] args)
	{
		Vector v1 = new Vector();
		Vector v2 = new Vector(5);
		Vector v3 = new Vector(10, 0.5f);
		float []array = {5f, 2.1f, 3.3f, 2.41f, 3.5f}, empty;
		Vector v4 = new Vector(array);

		// System.out.print("v1: " );
		// v1.printData();
		// System.out.print("v2: ");
		// v2.printData();
		// System.out.print("v3: ");
		// v3.printData();
		// System.out.print("v4: ");
		v4.printData();

		// System.out.println("Now let's fill vector v1...");
		// v1.getData();
		// System.out.print("v1 (updated): ");
		// v1.printData();

		// System.out.println("Now let's fill vector v1 using a custom size");
		// v1.getData(2);
		// System.out.print("v1 (size of 2): ");
		// v1.printData();

		v4.updateSize(3);
		v4.printData();


		v4.copyData(array);
		v4.printData();

		empty = v3.readData();
		v3.printData();
		System.out.print(empty);
	}
}//END of main or starting class 

