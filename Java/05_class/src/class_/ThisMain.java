package class_;

class This {
	private int b; //필드
	private static int c;
	
	public void setB(int b) { //함수의 구현.	인수(argument), 매개변수(parameter)
		System.out.println("this =" + this);
		this.b = b;
	}
	public void setC(int c) {
		this.c = c;
	}
	public int getB() { //반환되는 자료형 입력 
		return this.b;
	}
	public int getC() {
		return c;
	}
	
}
//---------------------
public class ThisMain {
	private int a; //필드 

	public static void main(String[] args) {
		ThisMain tm = new ThisMain(); //모든 클래스는 반드시 생성해야 한다.
		tm.a = 10;
		System.out.println("a = " + tm.a);
		System.out.println();
	
		//b에 20을 넣어 출력하시오
		This t = new This();
		System.out.println("객체 t = " + t);
		t.setB(20); //호출한 메서드는 반드시 돌아온다.
		System.out.println("t.b = " + t.getB());
		
	
		//c에 30을 넣어 출력
		t.setC(30);
		System.out.println("t.c = " + t.getC());
		System.out.println();
		
		This w = new This();
		System.out.println("객체 w = " + w);
		w.setB(40);
		w.setC(50);
		System.out.println("w.b = " + w.getB());
		System.out.println("w.c = " + w.getC());
	}

}
