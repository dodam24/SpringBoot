package basic;

import java.util.Arrays;
import java.util.Random;

public class MethodTest02 {

	public static void main(String[] args) {
		//난수: 컴퓨터에서 불규칙적으로 발생하는 수
		// 0 < = 난수 < 1
		
		double a = Math.random();
		System.out.println("난수 = " + a);
		
		Random r = new Random();
		double b = r.nextDouble();
		System.out.println("난수 = " + b);
		
		int[] ar = new int[5]; //배열. 방 5개 만들라는 뜻
		ar[0] = 25;
		ar[1] = 13;
		ar[2] = 45;
		ar[3] = 30;
		ar[4] = 15;
		System.out.println(ar[0] + ", " + ar[1] + ", " + ar[2] + ", " + ar[3] + ", " + ar[4]);
		
		Arrays.sort(ar);//오름차순으로 정렬.	void는 넘겨주는 값은 있지만 들어오는 값이 없으므로 앞에 '변수명 =' 을 써주지 않음
						//java.lang에 없으므로 위에서 자동으로 import 해줌
		System.out.println(ar[0] + ", " + ar[1] + ", " + ar[2] + ", " + ar[3] + ", " + ar[4]);
	}

}
