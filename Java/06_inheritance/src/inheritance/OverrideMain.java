package inheritance;
class AA {
	public int a = 3;
	public void disp() {
		a += 5;
		System.out.println("AA : " + a + " ");
	}
}
//-------------------------
class BB extends AA {
	public int a = 8;
	
	public void disp() {
		this.a += 5;
		System.out.println("BB : " + a + " ");
	}
}
//-------------------------
public class OverrideMain {

	public static void main(String[] args) {
		BB aa = new BB();
		aa.disp(); //BB : 13
		System.out.println("aa : " + aa.a); //13
		System.out.println();
		
		AA bb = new BB(); //부모가 자식 클래스 참조
		bb.disp(); //BB : 13 (메소드는 override 가능)
		System.out.println("bb : " + bb.a); //3 (필드는 override 불가능)
		
		BB cc = (BB)bb; //'자식 = (자식)부모' 형태로 형변환 (-> 다형성) 
		cc.disp(); //BB : 18
		System.out.println("cc : " + cc.a); //18
		System.out.println();
		
		AA dd = new AA();
		dd.disp();
		System.out.println("dd : " + dd.a);
		System.out.println();

	}

}
