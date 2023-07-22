package multi;

public class MultiArray02 {

	public static void main(String[] args) {
		int[][] ar = new int[10][10];
		int num = 0;
		
		// 입력 
		for(int i=0; i<ar.length; i++) {
			for(int j=0; j<ar[i].length; j++) {
				num++;
				ar[i][j] = num;
			} // for j 
		} // for i
		
//		ar[0][0] = 1;	ar[0][1] = 11;
//		ar[1][0] = 2;	ar[1][1] = 12;
//		ar[2][0] = 3;	ar[2][1] = 13;
		
		
		// 입력 (거꾸로 대입)
//		for(int i=0; i<ar.length; i--) {
//		for(int j=0; j<ar[i].length; j--) {
//			num++;
//			ar[i][j] = num;
//		} // for j 
//	} // for i
//		
//		ar[9][9] = 1;	ar[8][9] = 11;
//		ar[9][9] = 2;	ar[8][8] = 12;
//		ar[9][7] = 3;	ar[8][7] = 13;
		
		
		// 출력 
		for(int i=0; i<ar.length; i++) {
			for(int j=0; j<ar[i].length; j++) {
				System.out.print(String.format("%4d", ar[i][j]));
			} // for j
			System.out.println();
		} // for i 

	}

}
