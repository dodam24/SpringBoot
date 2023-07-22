package operator;

public class Operator05 {

	public static void main(String[] args) {
		boolean a = 25 > 36;
		System.out.println("a = " + a);	//False
		System.out.println("a = " + !a);	//True
		System.out.println();
		
		String b = "apple";	//literal 생성.	문자열 literal 제외 나머지는 다 new를 사용해야 함
		String c = new String("apple");	//String literal, new String 둘 다 가능

		String result = b == c ?	 "같다" : "다르다";	//주소 비교 (apple 같은지 여부 아님)
		System.out.println("b == c : " + result);
		System.out.println("b == c : " + (b != c ? "참" : "거짓"));	//같지 않다
		
		String result2 = b.equals(c)? "같다" : "다르다";	//문자열 비교
		System.out.println("b.equals(c) : " + result2);
		System.out.println("!b.equals(c) : " + ((b != c ? "참" : "거짓")));	//같지 않다
		System.out.println();
	}

}
