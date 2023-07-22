package operator;

import java.text.DecimalFormat;
import java.util.Scanner;

public class Salary {

	public static void main(String[] args) {
		
		//실행 결과
		Scanner sc = new Scanner(System.in);
		DecimalFormat df = new DecimalFormat();	//3자리마다 쉼표 찍기
		
		String name, position;
		int basePay, extraPay, total, tax, salary;
		double taxRate;
		
		
		System.out.print("이름 입력 : ");
		name = sc.nextLine();	//sc.next도 가능 (nextLine은 자동 줄 바뀜)
		
		System.out.print("직급 입력 : ");
		position= sc.nextLine();
		
		System.out.print("기본급 입력 : ");
		basePay = sc.nextInt();
		
		System.out.print("수당 입력 : ");
		extraPay = sc.nextInt();
		
		total = basePay + extraPay;
		taxRate = total>=5000000 ? 0.03 : total>=300000 ? 0.02 : 0.01;
		tax = (int)(total * taxRate);	//정수 + 실수 계산 안되므로 강제 형변환	
		salary = total - tax;
		
		
		//결과 출력
		System.out.println("*** " + name + "\t" + position + " 월급 ***");
		System.out.println("기본급 : " + df.format(basePay) + "원");
		System.out.println("수당 : " + df.format(extraPay) + "원");
		System.out.println("합계 : " + df.format(total) + "원");
		System.out.println("세금 : " + df.format(tax) + "원");
		System.out.println("월급 : " + df.format(salary) + "원");
	}
		
}

/*
[문제] 월급 계산 프로그램 - 조건 연산자
이름, 직급, 기본급, 수당을 입력하여 합계, 세금, 월급을 출력하시오
단 합계가 5,000,000원 이상이면 3%
       3,000,000원 이상이면 2%
       아니면 1%
       
합계 = 기본급 + 수당
세금 = 합계 * 세율
월급 = 합계 - 세금

[실행결과]
이름 입력 : 홍길동
직급 입력 : 부장
기본급 입력 : 4900000
수당 입력 : 200000

*** 홍길동 부장 월급 ***
기본급 : 4,900,000원
수당 : 200,000원
합계 : 5,100,000원
세금 : 153,000원
월급 : 4,947,000원
*/