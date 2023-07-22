package switch_;

import java.util.Scanner;

public class Switch01 {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		int days;
		//int days = 0;
		
		System.out.print("월 입력 : ");
		int month = scan.nextInt();
		
		switch(month) {
		case 1 : 
		case 3 : 
		case 5 : 
		case 7 : 
		case 8 : 
		case 10 : 
		case 12 : days=31; break;
		
		case 2 : days=28; break;
		
		case 4 : 
		case 6 : 
		case 9 : 
		case 11 : days=30; break;
		
		default : days=0;
		}
		
		System.out.println(month + "월은 " + days + "일 입니다");		//case  외의 값이 들어오지 않을 경우를 대비하여 지역변수를 초기화 시켜줘야 함

	}

}
