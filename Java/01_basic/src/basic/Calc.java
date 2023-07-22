package basic;

public class Calc {

	public static void main(String[] args) {
		int a;
		a = 320;
		//System.out.println(a);
		//123
		int b;
		b = 258;
		//System.out.println(b);
		
		int sum = a + b;
		System.out.println(String.format("%.2f", "320 + 258 : " + sum));
	
		int sub = a - b;
		System.out.println("320 - 258 : " + sub);
		
		int mul = a * b;
		System.out.println("320 * 258 : " + mul);
		
		int div = a / b;
		System.out.println("320 / 258 : " + div);
		
	}

}

//1줄 주석
/*
[문제] 320(a), 258(b)을 변수에 저장하여 합(sum), 차(sub), 곱(mul), 몫(div)을 구하시오.
단, 소수이하 2째자리까지 출력하시오.

[실행결과]
320 + 258 = xxx
320 - 258 = xxx
320 * 258 = xxx
320 / 258 = x.xx
*/


