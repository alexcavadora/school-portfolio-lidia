public class Principal
{
    public static void main(String[] arg)
    {
        Estudiante estudiante = new Estudiante();
        estudiante.setNUA(147668);
        estudiante.setNombre("Alejandro");
        estudiante.setApaterno("Alonso");
        estudiante.setAmaterno("Sánchez");
        estudiante.setDireccion("Revolución 337, Zona centro 36700");
        estudiante.setEdad((byte)21);
        estudiante.setCelular(4622654115l);
        estudiante.setProgramaEducativo("LIDIA");
        estudiante.setCorreo("a.alonsosanchez@ugto.mx");
        estudiante.setCalificaciones(10,7,9);

        System.out.println("Registro");
        estudiante.PrintEstudiante();
    }
}
