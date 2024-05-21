import java.util.Scanner;

public class NumComplejo
{
	public static void main(String[] args)
	{
		Scanner scanf = new Scanner(System.in);
		Complex c = new Complex(1,1);
		c.print();
	}
}

class Complex
{
	private float real;
	private float imaginary;
	Complex()
	{
		real = 0;
		imaginary = 0;
	}
	Complex(float r, float i)
	{
		real = r;
		imaginary = i;
	}

	Complex(Complex aux)
	{
		real = aux.real;
		imaginary = aux.imaginary;
	}

	public void print()
	{
		if (imaginary == 1)
		{
			System.out.println(real + " + i");
			return;
		}
		if (imaginary == -1)
		{
			System.out.println(real + " - i");
			return;
		}
		if (imaginary < 0)
		{
			System.out.println(real + " - " + imaginary + "i");
			return;
		}

		System.out.println(real + " - " + imaginary + "i");

	}
}