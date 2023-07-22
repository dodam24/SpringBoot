package operator;

import java.text.DecimalFormat;
import java.util.Scanner;

public class Money {

	public static void main(String[] args) {
		int money;

		Scanner scan = new Scanner(System.in); //키보드로부터 입력받는 Scanner 클래스를 생성
		System.out.print("돈 입력 : ");
		money = scan.nextInt();

		int rm1 = money % 1000;
		int rm2 = rm1 % 100;
		int rm3 = rm2 % 10;
		
		DecimalFormat df = new DecimalFormat();
	
		System.out.println("현금 : " + df.format(money) + "원"); //3자리마다 쉼표(,) 찍기
			
		int div1 = money / 1000;
		System.out.println("천원 : " + div1 + "장");
		
		int div2 = (rm1 / 100);
		System.out.println("백원 : " + div2 + "개");
		
		int div3 = (rm2 / 10);
		System.out.println("십원 : " + div3 + "개");
		
		int div4 = (rm3 / 1); 
		System.out.println("일원 : " + div4 + "개");

	}

}

/*
[문제] 동전 교환기 - 현금 5378원이 있습니다.

[실행결과]
현금 : 5378원
천원 : 5장
백원 : 3개
십원 : 7개
일원 : 8개
*/


/*
int money = 5378;
int th = money / 1000;
int th_mod = money % 1000;

int hd = th_mod / 100;
int hd_mod = th_mod % 100;

int ten = hd_mod / 10;
int one = hd_mod % 10;

sysout(현금: money)
sysout(천원: th)
sysout(백원: hd)
sysout(십원: ten)
sysout(일원: one)

*/