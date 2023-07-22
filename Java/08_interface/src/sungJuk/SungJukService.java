package sungJuk;

import java.util.ArrayList;
import java.util.Scanner;

public class SungJukService {
	private ArrayList<SungJukDTO> arrayList = new ArrayList<SungJukDTO>(); // <>, generic : 사용할 객체 타입 지정

	public void menu() {
			Scanner scan = new Scanner(System.in);
			SungJuk sungJuk = null;
			int num;
			
			while(true) {
				System.out.println();
				System.out.println("*************");
				System.out.println("1. 입력 ");
				System.out.println("2. 출력");
				System.out.println("3. 수정");
				System.out.println("4. 삭제");
				System.out.println("5. 정렬");
				System.out.println("6. 끝");
				System.out.println("*************");
				System.out.print("	번호 : ");
				num = scan.nextInt();
				
				if(num == 6) break;
				
				if(num == 1) sungJuk = new SungJukInsert(); // 부모 = 자식
				else if(num == 2) sungJuk = new SungJukList();
				else if(num == 3) sungJuk = new SungJukUpdate();
				else if(num == 4) sungJuk = new SungJukDelete();
				else if(num == 5) sungJuk = new SungJukSort();
				else {
					System.out.println("1 ~ 6번까지만 입력하세요");
					continue; // 반복문의 처음으로 이동
				}
				sungJuk.execute(arrayList); // 호출
			} // while
		} // menu()

}


