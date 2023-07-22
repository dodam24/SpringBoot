package inheritance;

enum Color { // Enum : 상수들의 집합체
	RED, GREEN, BLUE
}
//-----------
class Final {
	public final String FRUIT = "사과";
	public final String FRUIT2;
	
	public static final String ANIMAL = "기린";
	public static final String ANIMAL2;	// static은 생성자에서 초기화 불가능
	// static은 실행하자 마자 메모리 자동으로 생성 (new 할 필요 없음)

	public static final int RED = 0;
	public static final int GREEN = 1;
	public static final int BLUE = 2;
	
	static { 
		System.out.println("static 초기화 영역");
		ANIMAL2 = "코끼리";
	}
	
	public Final() { // 생성자를 이용하여 FRUIT2의 값을 초기화
		System.out.println("기본 생성자");
		FRUIT2 = "딸기";
	}
	
}
//----------------------
public class FinalMain {

	public static void main(String[] args) {
		final int A = 10; // 상수화 
		// A = 20;	error (final은 값을 변경할 수 없음) 
		System.out.println("A = " + A);
		
		final int B;
		B = 30; // 초기값 설정할 수 있는 기회 1번 더 부여
		// B = 40; // 에러
		System.out.println("B = " + B);
		
		Final f = new Final();
		System.out.println("FRUIT = " + f.FRUIT);
		System.out.println("FRUIT2 = " + f.FRUIT2);
		
		System.out.println("ANIMAL = " + Final.ANIMAL);
		System.out.println("ANIMAL2 = " + Final.ANIMAL2);
		
		System.out.println("빨강 = " + Color.RED);
		System.out.println("빨강 = " + Color.RED.ordinal());
		
		for(Color data : Color.values()) { // 확장형 for문
			System.out.println(data + "\t" + data.ordinal());
		}
		
	}

}
