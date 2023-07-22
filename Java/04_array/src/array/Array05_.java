package array;

public class Array05_ {

	public static void main(String[] args) {
		char[] ar = new char[50];
		int[] count = new int[26];
		
		for(int i=0; i<(ar.length)/10; i++) {
			for(int j=0; j<10; j++) {
				char alphabet = (char)((int)(Math.random() * 26) + 65);
				System.out.print(alphabet + " ");	
				count[alphabet-65]++;
				}
				System.out.println();
				}
		System.out.println();
		for(int i=0; i<count.length; i++) {
			System.out.println((char)(i + 65) + " : " + count[i] + "개");
		}
		
	}
	
}



/* 다중 for문 사용
[문제] 크기가 50개인 문자배열을 잡아서 65~90사이의 난수를 발생하여 저장 후 출력하시오
- 1줄에 10개씩 출력
- A의 개수? B의 개수? C의 개수? ~~~ Z의 개수 ?

[실행결과]
D F A G H I J K L T
O P W E R F V A S B
P Y R O L E E Z L I
F E U Z T U P P P A
S P G B F F O W J C

A : 2
B : 3
...
X : 0
Y : 1
Z : 0
*/