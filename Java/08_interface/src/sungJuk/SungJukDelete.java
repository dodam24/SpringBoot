package sungJuk;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

public class SungJukDelete implements SungJuk {

	@Override
	public void execute(ArrayList<SungJukDTO> arrayList) { // for문 돌 때마다 ArrayList 사이즈 계산 
		System.out.println();
		Scanner scan = new Scanner(System.in);
		
		System.out.print("삭제 할 이름 입력 : ");
		String name = scan.next();
		
		int count = 0;
//		for(int i=0; i<arrayList.size(); i++) {
//			if(arrayList.get(i).getName().equals(name)) { // arrayList 이름 가져와서 name이랑 같은지 비교 
//				arrayList.remove(i);
//				count++;
//			}
//		} // for
		
//		for(SungJukDTO sungJukDTO : arrayList) { // java.util.CurrentModificationExcption 에러 발생
//			if(sungJukDTO.getName().equals(name)) {
//				arrayList.remove(sungJukDTO);
//				count++;
//			}
//		} // for
		
		
		Iterator<SungJukDTO> it = arrayList.iterator();
		
		while(it.hasNext()) { // 항목이 없을 때까지 while문 반복 
			SungJukDTO sungJukDTO = it.next(); // 항목 꺼내서 sungJukDTO에 보관 후, 다음 항목으로 이동
			
			if(sungJukDTO.getName().equals(name)) {
				it.remove(); // it.next()가 반환하는 항목 삭제 (it.remove() 사용하려면 it.next() 먼저 사용해야 함)
				count++;
				
			}
			
			
		} // while 
		
		
		
		
		
		
		if(count == 0) // 삭제한 것이 없다면 
			System.out.println("회원의 정보가 없습니다");
		else
			System.out.println(count + "건을 삭제하였습니다");
	}

}
