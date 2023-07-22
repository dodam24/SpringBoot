package inheritance;

class Test { // (extends Object) : 모든 클래스는 Object로부터 상속 받음
	
}
//---------------------
class Sample {
	@Override
	public String toString() {
		return getClass() + "@개바부"; // getClass는 클래스명을 가져옴
	}
}
//---------------------
class Exam {
	private String name = "홍길동";
	// private int age = 25;
	
	@Override
	public String toString() {
		return super.toString(); // super는 부모 클래스의 생성자를 호출
		// return name + "\t" + age; // 주소값이 아닌 결과값을 가져옴 (가장 많이 쓰는 형태)
	}
}
//---------------------
public class ObjectMain { // (extends Object)

	public static void main(String[] args) {
		Test t = new Test();
		System.out.println("객체 t = " + t); // 클래스명@16진수
		System.out.println("객체 t = " + t.toString());
		System.out.println("객체 t = " + t.hashCode()); // 주소값이 10진수로 출력됨
		System.out.println();
		
		Sample s = new Sample();
		System.out.println("객체 s = " + s.toString());
		
		Exam e = new Exam();
		System.out.println("객체 e = " + e.toString());
		System.out.println();
		
		String aa = "apple";
		System.out.println("객체 aa = " + aa); // 오버라이딩 한 결과, 주소값이 아닌 문자열이 추출됨
		System.out.println("객체 aa = " + aa.toString());
		System.out.println("객체 aa = " + aa.hashCode()); // 문자열은 무한대로 표기되므로 10진수 표기 X (10진수로 출력되지만 거짓의 값)
		System.out.println();
		
		String bb = new String("apple");
		String cc = new String("apple");
		System.out.println("bb==cc : " + (bb==cc)); // false (주소 비교)
		System.out.println("bb.equals(cc) : " + bb.equals(cc)); // true (문자열 비교)
		System.out.println();
		
		Object dd = new Object(); // Object 클래스는 무조건 주소값만 비교 (equals는 String일 때만 문자열 비교)
		Object ee = new Object();
		System.out.println("dd==ee : " + (dd==ee)); // false
		System.out.println("dd.equals(ee) : " + dd.equals(ee)); // false
		System.out.println();
		
		Object ff = new String("apple"); // 부모 = 자식
		Object gg = new String("apple");
		System.out.println("ff==gg : " + (ff==gg)); // false
		System.out.println("ff.equals(gg) : " + ff.equals(gg)); // true
		System.out.println();
	}

}
