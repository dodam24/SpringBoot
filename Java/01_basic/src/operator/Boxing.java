package operator;

public class Boxing {

	public static void main(String[] args) {
		int a = 25;
		
		double b = (double)a / 3;		//Casting, 앞에서 강제형변환하면 뒤에는 자동형변환 됨 
		
		String c = "25";
		
		//int d = c;		//객체형을 기본형으로 강제형변환 불가능 -> 메소드 필요!
		int d = Integer.parseInt(c);
		
		int e = 5;
		Integer f = e;	//JDK 5.0~, AutoBoxing(기본형에서 자동으로 객체화시킴)
		//Integer f = new Integer(e);		//JDK 5.0 이전 버전에서 사용
		
		Integer g = 5;
		//int h = g;		//JDK 5.0~, AutoUnboxing(객체형을 기본형으로 자동 변환)
		int h = g.intValue();		//JDK 5.0 이전
		
	}

}
