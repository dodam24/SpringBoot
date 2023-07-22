package class__;

import java.util.StringTokenizer;

public class Token {

	public static void main(String[] args) {
		String str = "학원,집,,게임방";
		
		StringTokenizer st = new StringTokenizer(str, ",");
		System.out.println("토큰 개수 = " + st.countTokens()); //3;
		
		while(st.hasMoreTokens()) { //토큰의 유무 확인 (있다(true), 없다(false))
			System.out.println(st.nextToken()); //토큰을 꺼내고 다음 토큰으로 이동
		} //while
		System.out.println("----------------");
		
		String[] ar = str.split(","); //split함수는 토큰과 달리 비어있는 값도 가져옴
		
		for(String data : ar) {
			System.out.println(data);
		}
	}

}
