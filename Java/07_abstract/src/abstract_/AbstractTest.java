package abstract_;

public abstract class AbstractTest { // POJO (Plain Old Java Object)
	protected String name; // 생성자 or setter 이용해서 만들기

	public AbstractTest() { // 기본 생성자

	}
	
	public AbstractTest(String name) {
		super(); // 부모 생성자 호출 (여기서는 Object를 의미함)
		this.name = name;
	}

	public String getName() { // getter로 꺼내옴 (구현)
		return name;
	}
	
	public abstract void setName(String name); // 추상메소드
	
}
