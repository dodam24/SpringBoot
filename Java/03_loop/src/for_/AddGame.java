package for_;

import java.util.Scanner;

public class AddGame {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		
		int a, b, dab, count=0;
		
		//5문제 제공
		for(int i=1; i<=5; i++) {
			a = (int)(Math.random() * (99-10+1) + 10);		//10 ~ 99
			b = (int)(Math.random() * 90 + 10);
			
			System.out.print("[" + i + "] " + a + " + " + b + " = ");
			dab = scan.nextInt();
			
			if(dab == a+b) {
				System.out.println("참 잘했어요"); 
				count++;
			}
			else System.out.println("틀렸다");
		}
		System.out.println();
		System.out.println("당신은 총 " + count + "문제를 맞추어서 점수 " + count*20 + "점 입니다.");
	
	}

}


/*
[문제] 덧셈 문제
- 2자리 숫자(10 ~ 99)만 제공한다.
- 총 5문제를 제공한다 (for문)
- 1문제당 20점씩 처리한다.

[실행결과] 	//틀리면 한 번 더 기회 주기 (for, if, break 사용)
	a		b
[1] 87 + 56 = 78
틀렸다
[1] 87 + 56 = 143
참 잘했어요

[2] 17 + 64 =81
참 잘했어요

[5] 82 + 45 = 78
틀렸다

당신은 총 x문제를 맞추어서 점수 xx(count변수 *20)점 입니다.
*/
