package class__;

import java.util.Scanner;

public class Exam {
	private String name;
	private String dap;
	private char[] ox = new char[5];
	private int score;
	private final String JUNG = "11111";

	public Exam(){
		Scanner sc = new Scanner(System.in);
		System.out.print("이름 입력 : ");
		this.name = sc.next();
		System.out.print("답 입력 : ");
		this.dap = sc.next();
		
	}
	
	public void compare() {
		for(int i=0; i<ox.length; i++) {
			if(dap.charAt(i) == JUNG.charAt(i)) { //문자열 하나 가져와서 비교
				ox[i] = 'O';
				score += 20;
			}else
			ox[i] = 'X';
		}
	}

	public String getName() {
		return name;
	}
	public char[] getOx(){
		return ox;
	}
	public int getScore() {
		return score;
	}
}

