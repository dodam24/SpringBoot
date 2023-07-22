package class_;

class Person {
	private String name; //필드, 초기값
	private int age;
	
	public void setName(String n) {//메소드 직접 구현: public 결과형 메소드명(인수형 인수)
		name = n;
	}
	
	public void setAge(int a) { //메소드 구현
		age = a;
	}
	
	public void setData(String n, int a) {
		name = n;
		age = a;
	}
	
	public void setData() {} //데이터 받는 것도 없고 하는 일도 없음 (위의 초기값을 가져옴)
	
	public String getName() { //꺼내오기, String 반환
		return name; //반환값
	}
	
	public int getAge() { //
		return age;
	}
}
//------------------------------
public class PersonMain {

	public static void main(String[] args) {
		Person a; //객체 선언
		a = new Person(); //생성
		System.out.println("객체 a = " + a);
		a.setName("홍길동"); //호출한 함수는 반드시 제자리로 돌아온다.
		a.setAge(25); //호출
		System.out.println("이름=" + a.getName() + "\t 나이 =" + a.getAge());
		
		
		Person b = new Person();
		System.out.println("객체 b = " + b);
		System.out.println("객체 b = " + b);
		b.setName("코난"); //호출한 함수는 반드시 제자리로 돌아온다.
		b.setAge(13); //호출
		System.out.println("이름=" + b.getName() + "\t 나이=" + b.getAge());
		
		Person c = new Person();
		System.out.println("객체 c = " + c);
		c.setData("둘리", 100); //인수 2개
		System.out.println("이름=" + c.getName() + "\t 나이=" + c.getAge());
		
		Person d = new Person();
		System.out.println("객체 d = " + d);
		d.setData();
		System.out.println("이름=" + d.getName() + "\t 나이=" + d.getAge());
	}

}
