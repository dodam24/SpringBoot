package interface_;

public class ComputeMain {

	public static void main(String[] args) {
		ComputeService computeService = new ComputeService();
		computeService.menu(); // 메뉴 함수 호출
		System.out.println("프로그램을 종료합니다.");

	}

}
