package class__;

import java.util.Scanner;

public class StringBufferMain {
	private int dan; //여기서 선언하면 클래스 전체에서 사용 가능 
	
	public void input() {
		Scanner scan = new Scanner(System.in);
		System.out.print("원하는 단을 입력 : ");
		dan = scan.nextInt();
	}
	
	public void output() {
		StringBuffer buffer = new StringBuffer(); //append(), delete()
		
		for(int i=1; i<=9; i++) {
		//System.out.println(dan + "*" + i + "=" + dan*i) ;
			
		buffer.append(dan);
		buffer.append("*");
		buffer.append(i);
		buffer.append("=");
		buffer.append(dan*i);
		
		System.out.println(buffer.toString()); //StringBuffer -> String 변환
		
		buffer.delete(0, buffer.length());
		
		
	} //for
}
	public static void main(String[] args) { //main은 부르는 일만 수행
		StringBufferMain sbm = new StringBufferMain();
		sbm.input();
		sbm.output();
	}
}

/*
[문제] 구구단 

원하는 단을 입력 : 5
-------------------------- input() 이용
5*1=5
5*2=10
...
5*9=45
-------------------------- output() 이용
*/



