package while_;

import java.util.Scanner;

public class NumberGame {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		int com, user, count=0;
		while(true) {
		com = (int)(Math.random() * (100-1+1) +1);	// 1 ~ 100 사이의 난수 발생
		
		System.out.println("1 ~ 100 사이의 숫자를 맞추세요");
		System.out.println();
		
		while(true) {
			System.out.print("숫자 입력 : ");
			user = scan.nextInt();
			count++;	// count = count + 1
			if(com == user) break;		// while1 빠져나가기
			
			if(com > user) System.out.println(user + "보다 큰 숫자입니다.");
			else if(com < user) System.out.println(user + "보다 작은 숫자입니다.");
		
		}	// while1

		System.out.println("\n 딩동댕..." + count + "번 만에 맞추셨습니다.");
		
		System.out.print("\n 한 번 더? (y/n) : ");
		// int yn = scan.nextInt();  // y or n		// A가 65이므로 정수형으로 받는 것도 가능
		String yn = scan.next();					// y, n을 문자열로 받기
		
		// if(yn 'n' || yn == 'N') break;		// while2 빠져나가기
		if(yn.equals("n") || yn.equals("N")) break;
		}	// while2 (한 번 더 반복하는 부분)
		
		System.out.println("프로그램을 종료합니다.");
	}

}

/*
[문제] 숫자 맞추기 게임
- 컴퓨터가 1 ~ 100사이의 난수를 발생하면, 발생한 난수를 맞추는 게임
- 몇 번만에 맟주었는지 출력한다.

[실행결과]
1 ~ 100사이의 숫자를 맞추세요 (70)

숫자 입력 : 50
50보다 큰 숫자입니다.

숫자 입력 : 85
85보다 작은 숫자입니다.

~~~

숫자 입력 : 70
딩동댕...x번 만에 맞추셨습니다.

한 번 더? (y/n) :		// 반복되는 부분 체크할 것!
프로그램을 종료합니다.
*/