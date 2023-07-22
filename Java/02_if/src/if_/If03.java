package if_;

import java.util.Scanner;

public class If03 {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		
		System.out.print("a의 값 : ");
		int a = scan.nextInt();
		
		System.out.print("b의 값 : ");
		int b = scan.nextInt();
		
		System.out.print("c의 값 : ");
		int c = scan.nextInt(); 
		
		//순서대로 출력
		if(a<b && a<c) {		//a가 가장 작은 경우
				if(b<c) System.out.println(a  + "\t" + b + "\t" + c);
				else System.out.println(a  + "\t" + b + "\t" + c);
				
		} else if (b<c)	{	//b가 가장 작은 경우
				if(a<c) System.out.println(b  + "\t" + a + "\t" + c);
				else System.out.println(b  + "\t" + c + "\t" + a);
		
		} else {		//c가 가장 작은 경우
			if(a<b) System.out.println(c  + "\t" + a + "\t" + b);
			else System.out.println(c  + "\t" + a + "\t" + b);
		}	 
				
	}
}


/*
[문제] 3개의 숫자(a,b,c)를 입력받아서 순서대로 출력하시오 (if문 사용하시오)

[실행결과]
a의 값 : 98
b의 값 : 90
c의 값 : 85

85 90 98
---------------------
a의 값 : 75
b의 값 : 25
c의 값 : 36

25 36 75
*/