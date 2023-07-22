package inheritance;

import java.util.Scanner;

class Shape {
	protected double area;
	protected Scanner scan = new Scanner(System.in);
	
	public Shape() {
		System.out.println("Shape 기본생성자");
	}
	public void calcArea() {
		System.out.println("도형을 계산합니다.");
	}
	public void dispArea() {
		System.out.println("도형을 출력합니다.");
	}
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
		/*
		Sam sam = new Sam();
		sam.calcArea(); //부모꺼 호출
		sam.dispArea();
		System.out.println();
		
		//사각형
		Sa sa = new Sa();
		sa.calcArea();
		sa.dispArea();
		System.out.println();
		
		//사다리꼴 
		Sadari sadari = new Sadari();
		sadari.calcArea();
		sadari.dispArea();
		System.out.println();
		*/
		
		//다형성 : 부모가 자식 클래스를 참조 가능 
		Shape shape; //리모컨만 필요
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
