package class_;

public class SungJukMain2 {

	public static void main(String[] args) {

		SungJuk[] ar = new SungJuk[3]; //객체 배열 (객체형은 기본값이 null로 설정되어 있음)
		ar[0] = new SungJuk(); //클래스 생성 
		ar[0].setData("홍길동", 91, 95, 100); //호출
		ar[0].calc();
		
		ar[1].setData("프로도", 100, 89, 75); //호출
		ar[1].calc();
		
		ar[2].setData("죠르디", 75, 80, 48); //호출
		ar[2].calc();
		
		for(int i=0; i<ar.length; i++) {
			ar[i].calc();
		System.out.println(ar[i].getName() + "\t"
						+ ar[i].getName() + "\t"
						+ ar[i].getKor() + "\t"
						+ ar[i].getEng() + "\t"
						+ ar[i].getMath() + "\t"
						+ ar[i].getTot() + "\t"
						+ ar[i].getAvg() + "\t"
						+ ar[i].getGrade());
		} //for i
	}
}
