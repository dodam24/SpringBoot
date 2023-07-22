package basic;

class Test {	//클래스 여러 개도 가능
	int a = 10;
	static int b = 20;
	static String str;
}
//---------------------
public class Variable02 {	//public 붙으면 '주인'. 아래 main 있어야 가능
	int a; //필드(=전역변수). 값이 초기화되어 있음
	double b; //필드
	static int c; 
	
	public static void main(String[] args) {	//static은 실행하자 마자 메모리에 잡히므로 메모리 할당할 필요 없음
		int a = 5; //지역변수(local variable)은 초기값을 설정해줘야 함
		System.out.println("지역변수 a = " + a);
		
		Variable02 v = new Variable02(); //메모리 생성. 클래스 객체형은 new 연산자 필수!
		System.out.println("객체 = " + v);
		System.out.println("필드 a = " + v.a);
		System.out.println("필드 b = " + v.b);
		
		System.out.println("필드 c = " + c); //클래스 생성 안 해도 메모리에 자동으로 생성됨
		System.out.println("필드 c = " + Variable02.c);
		System.out.println();
		
		//Test클래스의 a값을 출력하시오.
		Test t = new Test();
		System.out.println("객체 = " + t);
		System.out.println("필드 a = " + t.a);
		System.out.println("필드 b = " + Test.b); //내 클래스 영역이 아니므로 이 때는 클래스(Test) 생략 불가능
		System.out.println("필드 Str = " + Test.str);	//string의 초기값은 null
	
	}

}
