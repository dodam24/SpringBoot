package class__;

class Fruit { // 1인분
	private String pum;
	private int jan, feb, mar, tot;
	private static int sumJan, sumFeb, sumMar;
	
	public Fruit(String pum, int jan, int feb, int mar) {
		this.pum = pum;
		this.jan = jan;
		this.feb = feb;
		this.mar = mar;
	}
	public void calcTot() {
		tot = (jan + feb + mar);
		sumJan += jan;
		sumFeb += feb;
		sumMar += mar;
	}
	public void display() {
		System.out.println(pum + "\t" + jan + "\t" + feb + "\t" + mar + "\t" + tot);
	}
	public static void output() {
		System.out.println("\t" + sumJan + "\t" + sumFeb + "\t" + sumMar);
	}
}
//----------------------
public class FruitMain {

	public static void main(String[] args) {
		Fruit[] ar = {new Fruit("사과", 100, 80, 75), 
					  new Fruit("포도", 30, 25, 10), 
					  new Fruit("딸기", 25, 30, 90)};
		
		System.out.println("-----------------------------------");
		System.out.println("PUM\tJAN\tFEB\tMAR\tTOT");
		System.out.println("-----------------------------------");
		
		for(Fruit data : ar) {
			data.calcTot();
			data.display();
		}
		System.out.println("-----------------------------------");
		Fruit.output();

	}

}


/*
[문제] 
과일 판매량 구하기
월별 매출합계도 구하시오

클래스 : Fruit
필드 : pum, jan, feb, mar, tot
	  sumJan, sumFeb, sumMar (데이터는 스캐너 쓰지 말고 생성자 통해서 데이터 넣기) 

메소드 : 생성자(품명, 1월, 2월, 3월)
		calcTot()
		display()
		public static void output() - 마지막에 output 처리

클래스 : FruitMain

[실행결과]
---------------------------------
PUM      JAN   FEB   MAR      TOT
---------------------------------
사과    100    80    75     255	(1인분) 객체배열 사용
포도     30    25    10     65
딸기     25    30    90     145
---------------------------------
        155   135   175            output(로 처리

*/