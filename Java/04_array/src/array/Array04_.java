package array;
import java.util.Scanner;

public class Array04_ {

	public static void main(String[] args) {
		int num, position;
		boolean [] ar = new boolean[5];
		
		Scanner scan = new Scanner(System.in);
		System.out.print("위치 입력 : ");
		
		while(true) {
			System.out.println();
			System.out.println("주차장 관리 프로그램");
			System.out.println("**************");
			System.out.println("   1. 입차");
			System.out.println("   2. 출차");
			System.out.println("   3. 리스트");
			System.out.println("   4. 종료");
			System.out.println("**************");
			System.out.print("  번호 입력 : ");
			num = scan.nextInt();
			
			if(num == 4) break;
			
			if(num == 1) {
				System.out.print("위치 입력 : ");
				position = scan.nextInt();
				if(ar[position-1]) {
					System.out.println("이미 주차되어 있습니다\n");
				}else {
					ar[position-1] = true;
					System.out.println(position + "위치에 입차\n");
				}
				
			}else if(num == 2) {
				System.out.print("위치 입력 : ");
				position = scan.nextInt();
				if(ar[position-1]) {
					ar[position-1] = false;
					System.out.println(position + "위치에 출차\n");
				}else {
					System.out.println("주차되어 있지 않습니다\n");
				}
				
			}else if(num == 3) {
			for(int i=0; i<ar.length; i++)
				System.out.println((i+1) + "위치 : " + ar[i]);
			}
				
		
		}	// while문
	System.out.println("프로그램을 종료합니다.");

	}

}