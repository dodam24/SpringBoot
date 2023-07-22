package Constructor;

public class MemberDTO { //1인분 (DTO: Data Transfer Object)
	private String name;
	private int age;
	private String phone;
	private String address;
	
	//생성자를 통해서 데이터 4개를 받아온다. (set 데이터 대신)
	public MemberDTO(String name, int age, String phone, String address) {
		this.name = name;
		this.age = age;
		this.phone = phone;
		this.address = address;
	}
	
	//get
	public String getName() {
		return name;
	}
	public int getAge() {
		return age;
	}
	public String getPhone() {
		return phone;
	}
	public String getAddress() {
		return address;
	}
	
	
	//set 
	public void setName(String name) {
		this.name = name;
	}
	
	public void setPhone(String phone) {
		this.phone = phone;
	}
	
	public void setAddress(String address) {
		this.address = address;
	}
}
