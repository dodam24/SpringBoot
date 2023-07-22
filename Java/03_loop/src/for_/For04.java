package for_;

import java.util.Scanner;

public class For04 {

	public static void main(String[] args) {
		
		Scanner scan = new Scanner(System.in);
		int mul = 1;
		
		System.out.print("x의 값 입력 : ");
		int x = scan.nextInt();
		
		System.out.print("y의 값 입력 : ");
		int y = scan.nextInt();
	
		
		for(int i=1; i<=y; i++) {			//x의 y승 프로그램
			mul *= x;	
		}
		
		System.out.println(x + "의 " + y + "승은 " + mul);
	}

}

/*
[문제] 제곱 계산 - x의 y승

x의 값 입력 : 2
y의 값 입력 : 5

2의 5승은 32

*/
