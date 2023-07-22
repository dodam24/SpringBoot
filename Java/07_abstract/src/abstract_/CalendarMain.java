package abstract_;

import java.util.Calendar;
import java.util.Scanner;

class CalendarEx {
	private int year, month, week, lastDay;
	
	public CalendarEx() {
		Scanner sc = new Scanner(System.in);
		
		System.out.print("년도 입력 : ");
		this.year = sc.nextInt();
		System.out.print("월 입력 : ");
		this.month = sc.nextInt();
	}
	public void calc() {
		Calendar cal = Calendar.getInstance(); // 메소드 이용하여 클래스 생성
		
		// 생성할 때 기준인 시스템 날짜를 내가 원하는 날짜로 변경
//		cal.set(Calendar.YEAR, this.year); // 년도
//		cal.set(Calendar.MONTH, this.month-1); // 월
//		cal.set(Calendar.DAY_OF_MONTH, 1); // 일 
		
		cal.set(year, month-1, 1);
		week = cal.get(Calendar.DAY_OF_WEEK); // 1일인 요일 구하기, 일요일:-1, 월요일:-2, 화요일:-3
		lastDay = cal.getActualMaximum(Calendar.DAY_OF_MONTH); // 28, 29, 30, 31
	}
	
	public void display() {
		System.out.println("일\t월\t화\t수\t목\t금\t토");
		
		for(int i=1; i<week; i++) {
			System.out.print("\t");
		} // for i
	
		for(int i=1; i<=lastDay; i++) {
			System.out.print(i + "\t");
			
			if(week%7 == 0) System.out.println();
			week++;
						
		} // for i
	}
}
//-------------------------
public class CalendarMain {

	public static void main(String[] args) {
		CalendarEx c = new CalendarEx();
		c.calc();
		c.display();
	}

}


/*
문제] 만년달력
- 년도, 월을 입력하여 달력을 작성하시오


클래스명 : CalendarEx
필드 : 
메소드 :
- 기본생성자 : 월, 일을 입력
  calc()   : 매달 1일의 요일이 무엇인지? (Calendar에 메소드 준비)
             매달 마지막이 28, 29, 30, 31 무엇인지? (Calendar에 메소드 준비)
  display() 출력

클래스명 : CalendarMain 


[실행결과]
년도 입력 : 2002
월 입력 : 10   

일   월   화   수   목   금   토
      	 1    2   3    4    5
6    7   8    9   10   11   12
13   14  15   16  17   18   19
20   21  22   23  24   25   26
27   28  29   30  31
*/