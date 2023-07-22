package abstract_;

import java.util.Scanner;

abstract class Shape {
	protected double area;
	protected Scanner scan = new Scanner(System.in);
	
	public Shape() {
		System.out.println("Shape 기본생성자");
	}
	public abstract void calcArea(); // 추상메소드
	public abstract void dispArea(); // 추상메소드 

}
//--------------------
class Sam extends Shape {
	protected int base, height;
	
	public Sam() { //생성자
		System.out.println("Sam 기본생성자");
		System.out.print("밑변 : ");
		base = scan.nextInt();
		System.out.print("높이 : ");
		height = scan.nextInt();
	}
	//@Override
	public void calcArea() {
		area = base * height / 2.0;
	}
	
	//@Override
	public void dispArea() {
		System.out.println("삼각형 넓이 = " + area);
	}
}

//--------------------
class Sa extends Shape {
	protected int width, height;
	
	public Sa() { //생성자
		System.out.println("Sa 기본생성자");
		System.out.print("가로 : ");
		width = scan.nextInt();
		System.out.print("세로 : ");
		height = scan.nextInt();
	}
	//@Override
	public void calcArea() {
		area = width * height;
	}
	
	//@Override
	public void dispArea() {
	System.out.println("사각형 넓이 = " + area);	
	}
	
}
//--------------------
class Sadari extends Shape {
	protected int top, bottom, height;
	
	public Sadari() {
		System.out.println("Sadari 기본생성자");
		System.out.print("윗변 : ");
		top = scan.nextInt();
		System.out.print("밑변 : ");
		bottom = scan.nextInt();
		System.out.print("높이 : ");
		height = scan.nextInt();
	}
	//@Override
	public void calcArea() {
		area = (top + bottom) * height / 2.0;
	}
	
	//@Override5
	public void dispArea() {
		System.out.println("사다리꼴 넓이 = " + area);
	}
}
//--------------------
public class ShapeMain {

	public static void main(String[] args) {
		
		//다형성 : 부모가 자식 클래스를 참조 가능 (부모 = 자식)
		Shape shape;
		shape = new Sam();
		shape.calcArea();
		shape.dispArea();
		System.out.println();
		
		shape = new Sa();
		shape.calcArea();
		shape.dispArea();
		System.out.println();
		
		shape = new Sadari();
		shape.calcArea();
		shape.dispArea();
		System.out.println();
	}

}
