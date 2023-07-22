package for_;

public class For01 {

	public static void main(String[] args) {
		int i;
		for(i=1; i<=10; i++) {		//맨 앞과 맨 뒤의 값만 비교해서 몇 번 돌아가는지 확인
		
			System.out.println("Hello Java!! : " + i);
		}
		
		System.out.println("탈출 i = " + i);
		System.out.println();
		
		
		//10 9 8 7 6 5 4 3 2 1 
		for(i=10; i>=1; i--) {
			System.out.print (i + " ");
		}
		
		System.out.println();
		
		
		//A  B C D E F G ~~~~~ X Y Z
		for(i='A'; i<='Z'; i++) {
			System.out.print((char)i + " ");
		}
		
		System.out.println();
	}

}
