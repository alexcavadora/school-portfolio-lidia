import java.util.Scanner;
public class Cadena
{
	public static void main(String args[])
	{
		Scanner scanf = new Scanner(System.in);
		String text = "hola eso tilin, ete setch o el pepe? ", text2 = "otro string diferente ", text3;
		System.out.println(text + "hola");
		text3 = text2 + text;
		System.out.println(text3.length());
		System.out.println("Introduce un número");
		int entero = Integer.parseInt(scanf.nextLine());
		System.out.println(entero+1);

		System.out.println("Introduce un número");
		int i = scanf.nextInt();
		System.out.println(i+16);
	}
}