package Constructor;

public class ConstructorMain {
	private String name;
	private int age;
	
	public ConstructorMain() {
		System.out.println("default 생성자");
	}
	
	public ConstructorMain(String name) {
		this(); //위에 있는 다른 생성자 호출
		System.out.println("name 처리 생성자");
		this.name = name;
	}
	
	public ConstructorMain(int age) {
		this("코난"); //Overload된 다른 생성자를 호출할 수 있다.
		System.out.println("age 처리 생성자");
		this.age = age;
	}
	
	
	public static void main(String[] args) {
		ConstructorMain aa = new ConstructorMain(); //생성자 자동 호출 (new 할 때)
		System.out.println(aa.name + "\t" + aa.age);
		System.out.println();
		
		ConstructorMain bb = new ConstructorMain("홍길동");
		System.out.println(bb.name + "\t" + bb.age);
		System.out.println();
		
		ConstructorMain cc = new ConstructorMain(25);
		System.out.println(cc.name + "\t" + cc.age);
		System.out.println();
	}

}
