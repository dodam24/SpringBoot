package for_;

import java.util.Scanner;

public class For06 {

	public static void main(String[] args) {
		
		System.out.println("숫자 입력 : ");
		Scanner scan = new Scanner(System.in);
		int num = scan.nextInt();
		
		int fac = 1;
		for(int i=num; i>=1; i--) {
			fac *= i;
		}
		System.out.println(num + "! = " + fac);
	}

}


/*
//for03 문제와 유사한 형태

[문제] 팩토리얼을 구하시오 (for)
- 입력되는 숫자는 1 ~ 10 사이만 입력한다.

[실행결과]
숫자 입력 : 3
3! = 6 (1*2*3)
---------------------

숫자 입력 : 5
5! = 120 (1*2*3*4*5)
*/