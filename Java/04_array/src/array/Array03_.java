package array;
import java.util.Scanner;

public class Array03_ {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		int size;
		int[] ar;
		int sum = 0;
		
		System.out.print("배열 크기 입력 : ");
		size = scan.nextInt();
		
		ar = new int[size];		// 배열 생성
		
		for(int i=0; i<size; i++) {
			System.out.print("ar[" + i + "] 입력 : ");
			ar[i] = scan.nextInt();
		
			sum += ar[i];
		}	// for문
		System.out.println();
		
		for(int data : ar) {		// 25 13 57 출력 부
			System.out.print(data + "	");
		}
		System.out.println();
		System.out.println("합 = " + sum);
	}
}