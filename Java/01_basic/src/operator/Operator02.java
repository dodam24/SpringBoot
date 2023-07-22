package operator;

import java.util.Scanner;

public class Operator02 {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		
		System.out.print("점수를 입력하세요 : ");
		int num = scan.nextInt();
		
		//짝수 ? "짝수" : "홀수";
		String result = num%2 == 0? "짝수" : "홀수";
						//2의 배수 입니까?
						//2로 나누면 나머지 0 입니까? num % 2 == 0
						//홀수는 num % 2 == 1
		
		//num이 2와 3의 공배수입니까? "공배수이다" : "공배수 아니다";
		String result2 = num%2 == 0 && num%3 == 0? "공배수이다" : "공배수 아니다";

		System.out.println(result);
		System.out.println(result2);
		
	}

}
