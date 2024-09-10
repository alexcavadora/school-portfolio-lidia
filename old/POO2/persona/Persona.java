import java.util.Scanner;
public class Persona
{
    private int Id;
    private String Nombre;
    private String Apaterno;
    private String Amaterno;
    private String Direccion;
    private byte Edad;
    private Long Celular;
    private String Correo;    
    
    Persona()
    {
        Id=0;
        Nombre=null;
        Apaterno=null;
        Amaterno=null;
        Direccion=null;
        Edad=0;
        Celular=(long)1;
        Correo=null;
    }

    public void setId(int aux)
    {
        Id=aux;
    }
    public void setNombre(String aux)
    {
        Nombre=aux;
    }
    public void setApaterno(String aux)
    {
        Apaterno=aux;
    }
    public void setAmaterno(String aux)
    {
        Amaterno=aux;
    }
    public void setDireccion(String aux)
    {
        Direccion=aux;
    }
    public void setCelular(Long aux)
    {
        Celular=aux;
    }
    public void setCorreo(String aux)
    {
        Correo=aux;
    }
    public void setEdad(byte aux)
    {
        Edad=aux;
    }

    public int getId()
    {
        return Id;
    }
    public String getNombre()
    {
        return Nombre;
    }
    public String getApaterno()
    {
        return Apaterno;
    }
    public String getAmaterno()
    {
        return Amaterno;
    }
    public String getDireccion()
    {
        return Direccion;
    }
    public byte getEdad()
    {
        return Edad;
    }
    public long getCelular()
    {
        return Celular;
    }
    public String getCorreo()
    {
        return Correo;
    }

    public void Captura()
    {
        Scanner scanf=new Scanner(System.in);
        System.out.print("Nombre(s): ");
        Nombre=scanf.nextLine(); 
        System.out.print("A. paterno: ");
        Apaterno=scanf.nextLine();
        System.out.print("A. materno: ");
        Amaterno=scanf.nextLine();
        System.out.print("Direccion: ");
        Direccion=scanf.nextLine();
        System.out.print("Edad: ");
        Edad=scanf.nextByte();
        System.out.print("Celular: ");
        Celular=scanf.nextLong();
        System.out.print("Correo: ");
        scanf.nextLine(); //Limpiar buffer del teclado 
        Correo=scanf.nextLine();
    }
    public void Print()
    {
        System.out.println("Nombre(s): "+Nombre);
        System.out.println("A. paterno: "+Apaterno);
        System.out.println("A. materno: "+Amaterno);
        System.out.println("Direccion: "+Direccion);
        System.out.println("Edad: "+Edad);
        System.out.println("Celular: "+Celular);
        System.out.println("Correo: "+Correo);
    }

}

