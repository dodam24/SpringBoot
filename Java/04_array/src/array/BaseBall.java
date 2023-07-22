package array;

import java.util.Scanner;

public class BaseBall {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		int[] com = new int[3];
		int[] user = new int[3];
		String yn;
		int strike = 0;
		int ball = 0;

		do {
			System.out.print("게임을 실행하시겠습니까(Y/N) : ");
			yn = scan.next();
			
		}while(!yn.equals("Y") && !yn.equals("y") && !yn.equals("N") && !yn.equals("n")); //y나 n이 안 들어오면 반복 
		
		if(!yn.equals("Y") || !yn.equals("y")) {
			System.out.println("게임을 시작합니다");
			
			//컴퓨터가 난수 발생
			for(int i=0; i<com.length; i++ ) {
				com[i] = (int)(Math.random()*9+1);
				
				//중복 제거
				for(int j=0; j<i; j++) {
					if(com[i] == com[j]) {
						i--;
						break; //for j 빠져나감
					}
				} //for j
			} //for i
			System.out.println(com[0] + ", " + com[1] + ", " + com[2]);
			
			//사용자 숫자 입력
			while(true) {
				System.out.println();
				System.out.print("숫자 입력 : ");
				int num = scan.nextInt();
				
				//숫자 분리해서 user 배열에 넣기 
				user[0] = num/100;
				user[1] = (num%100)/10;
				user[2] = (num%100)%10;
				System.out.println(user[0] + ", " + user[1] + ", " + user[2]);
				
				//비교 (com vs. user)
				strike = ball = 0;
				for(int i=0; i<com.length; i++) {
					for(int j=0; j<user.length; j++) {
						
						if(com[i] == user[j]) {
							if(i == j) strike++;
							else ball++;
						}
					} //for j
				} //for i 
				
				System.out.println(strike + "스트라이크\t" + ball + "볼");
				
				if(strike == 3) {
					System.out.println("정답!");
					break; //while 빠져나감 
				}
			} //while
				
		}else
			System.out.println("프로그램을 종료합니다.");
	}

}

/*
[문제] 야구게임
크기가 3개인 정수형 배열을 잡고 1~9사이의 난수를 발생한다
발생한 숫자를 맞추는 게임
단, 중복은 제거한다

[실행결과]
게임을 실행하시겠습니까(Y/N) : w
게임을 실행하시겠습니까(Y/N) : u
게임을 실행하시겠습니까(Y/N) : y

게임을 시작합니다 <- Y 입력하면 시작

숫자입력 : 123
0스트라이크 0볼

숫자입력 : 567
0스트라이크 2볼

숫자입력 : 758
1스트라이크 2볼
...

숫자입력 : 785
3스트라이크 0볼

프로그램을 종료합니다. <- N 입력하면 종료
*/