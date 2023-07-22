package array;

public class Array01_ {

	public static void main(String[] args) {
		int[] ar;
		ar = new int[5];	// 배열명은 배열의 주소를 나타냄. 배열명 ar = [I@7ad041f3
		System.out.println("배열명 ar = " + ar);
		
		ar[0] = 25;
		ar[1] = 37;
		ar[2] = 55;
		ar[3] = 42;
		ar[4] = 30;
		
		System.out.println("배열 크기 = " + ar.length);
		
		for(int i=0; i<ar.length; i++) {
			System.out.println("ar[" + i + "] = " + ar[i]);
		}
		System.out.println();
		
		
		System.out.println("거꾸로 출력");
		
		for(int i=ar.length-1; i>=0; i--) {
			System.out.println("ar[" + i + "] = " + ar[i]);
		}
		System.out.println();
		
		System.out.println("홀수 데이터만 출력");
		for(int i=0; i<ar.length; i++) {
			if(ar[i] % 2 == 1) {
				System.out.println("ar[" + i + "] = " + ar[i]);
			}
				else  System.out.println();
			
			
			System.out.println("확장 for");
			for(int data : ar) {
				System.out.println(data);
			}
		}
	}

}