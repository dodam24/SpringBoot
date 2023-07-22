package basic;

public class Variable01 {

	public static void main(String[] args) {
		System.out.println(Integer.MIN_VALUE + ", " + Integer.MAX_VALUE);
		System.out.println(Long.MIN_VALUE + ", " + Long.MAX_VALUE);
		System.out.println();
		
		boolean a; //방의 이름을 a로 잡음
		a = 25 > 36;
		System.out.println("a = " + a);
		
		char b;
		b = 'A'; //65, 0100 0001
		System.out.println("b = " + b);
		
		char c; //char로 잡아서 문자 출력
		c = 65; //위의 'A'와 동일. char형이라 65가 아닌 'A'로 출력됨
		System.out.println("c = " + c);
		
		byte d = 0; //1byte, 8bit, -128 ~ +127
		//d = 128; : error
		System.out.println("d = " + d); //지역 변수 초기화해줘야 에러 발생하지 않음
		
		int e;
		e = 65; //0100 0001
		System.out.println("e = " + e);
		
		int f; //인트(integer)형으로 잡아서 65가 나옴
		f = 'A';
		System.out.println("f = " + f); //65
		
		long g;
		g = 25L; //25L은 long형 상수
		System.out.println("g = " + g); //
		
		float h;
		h = 43.8F; //43.8은 double형 상수
		System.out.println("h = " + h); //
		// 방법 1. h = (float)43.8; //강제형변환 (이 때만 바꿔주고, 원래 값 자체는 바뀌지 않음)
		// 방법 2. h = 43.8F; //43.8은 float형 상수
	
	}

}
