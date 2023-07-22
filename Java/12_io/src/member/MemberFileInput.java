package member;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;
import io.PersonDTO;

public class MemberFileInput implements Member {

	@Override
	public void execute(List<MemberDTO> list) {
		System.out.println();
		
		list.clear(); //초기화

		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream("member.text"));
			
			while(true) {
				try {
					MemberDTO memberDTO = (MemberDTO)ois.readObject();
					list.add(memberDTO);
				}catch(EOFException e) {
					break;
				}
			}
				
			ois.close();
			System.out.println("파일 읽기 완료");
		
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
}
		