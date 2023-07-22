package inheritance;

public class SubMain extends Super { //Super로부터 상속
	private String name;
	private int age;


	SubMain(){
		System.out.println("SubMain 기본 생성자");
	}
	SubMain(String name, int age, Double weight, Double height){
		this.name = name;
		this.age = age;
		super.weight = weight; //this.weight = weight; 
		//super는 부모 클래스의 주소값, this는 나 자신의 주소값
		this.height = height;
	}
	
	public void output() {
		System.out.println("이름 = " + name);
		System.out.println("나이 = " + age);
		this.disp(); //disp 불러옴 (this 생략 가능)
	}
	
	public static void main(String[] args) { //메인은 static이므로 this 사용 불가
		SubMain aa = new SubMain("홍길동", 25, 73.5, 182.6);
		aa.disp(); //부모 메소드 호출
		System.out.println("----------------------------");
		aa.output();
		System.out.println("============================");
		
		Super bb = new SubMain("코난", 13, 53.5, 156.6);
		//bb.output(); 에러 발생
		bb.disp();
	}

}

/*
자식클래스 메모리에 생성할 때
- 부모 클래스 생성 
- 자식 클래스 생성
*/