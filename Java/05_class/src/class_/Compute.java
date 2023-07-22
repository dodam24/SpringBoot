package class_;

public class Compute { //1인분
	private int x, y, sum, sub, mul;
	private double div;
	
	public void setData(int x, int y) { //여기에 있는 this는 필수! 절대 생략 불가함
		this.x = x; //this는 클래스쪽에 있는 것을 의미함 (인수 x, y가 아님)
		this.y = y;
	}
	
	public void calc() {
		this.sum = this.x + this.y;
		sub = x - y;
		mul = x * y;
		div = (double)x / y;
	}
	
	public int getX() {
		return this.x; //this는 생략 가능
	}
	public int getY() {
		return y;
	}
	public int getSum() {
		return sum;
	}
	public int getSub() {
		return sub;
	}
	public int getMul() {
		return mul;
	}
	public String getDiv() {
		return String.format("%.3f", div);
	}
	
}