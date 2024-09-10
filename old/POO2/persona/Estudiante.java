public class Estudiante extends Persona
{
	private float cal1, cal2, cal3, promedio;
	private String prog_edu;

	Estudiante()
	{
		super();
		cal1 = -1.0f;
		cal2 = -1.0f;
		cal3 = -1.0f;
		promedio = -1.0f;
	}

	public void setCalificaciones(float c1, float c2, float c3)
	{
		cal1 = c1;
		cal2 = c2;
		cal3 = c3;
		promedio = c1+c2+c3;
		promedio /= 3.0f;
	}

	public void setProgramaEducativo(String newPrograma)
	{
		prog_edu = newPrograma;
	}

	public void setNUA(int newNUA)
	{
		setId(newNUA);
	}

	public String getProgramaEducativo()
	{
		return prog_edu;
	}

	public int getNUA()
	{
		return getId();
	}

	public float getCal1()
	{
		return cal1;
	}

	public float getCal2()
	{
		return cal2;
	}

	public float getCal3()
	{
		return cal3;
	}

	public void PrintEstudiante()
	{
		System.out.println("NUA: " + getNUA());
		Print();
		System.out.println("Programa Educativo: " + prog_edu);
		System.out.println("Calificación 1: " + cal1);
		System.out.println("Calificación 2: " + cal2);
		System.out.println("Calificación 3: " + cal3);
		System.out.println("Promedio: "+ promedio);
	}
}