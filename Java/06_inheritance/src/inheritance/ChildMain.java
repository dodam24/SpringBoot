package inheritance;

public class ChildMain extends Super { //Super로부터 상속 
	private String name;
	private int age;


	ChildMain(){
		System.out.println("SubMain 기본 생성자");
	}
	ChildMain(String name, int age, Double weight, Double height){
		super(weight, height); //부모 생성자 호출
		
		this.name = name;
		this.age = age;
//		super.weight = weight; //this.weight = weight; 
//		//super는 부모 클래스의 주소값, this는 나 자신의 주소값
//		this.height = height;
	}
	
	public void disp() {
		System.out.println("이름 = " + name);
		System.out.println("나이 = " + age);
		super.disp();
	}
	
	public static void main(String[] args) {
		ChildMain aa = new ChildMain("홍길동", 25, 73.5, 182.6);
		aa.disp();
		System.out.println("============================");
		
		Super bb = new ChildMain("코난", 13, 53.5, 156.6);
		bb.disp();

	}

}
