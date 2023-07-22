package operator;

public class Comp {

	public static void main(String[] args) {
		char ch = 'B';
		//char ch = 'e';
		

		//ch가 대문자 입니까? 소문자로 변환 : 대문자로 변환;
		int result = ch>='A' && ch<='Z'? ch+32 : ch-32; //안에 들어갈 때는 65라는 2진수로 들어감
														//'A' + 3 = 68 (정수형으로 출력됨)
		System.out.println(ch + "-> " + (char)result);
		
	}
}

/*
[문제] 변수의 값이 대문자이면 소문자로 변환해서 출력, 소문자이면 대문자로 변환해서 출력하시오

[실행결과]
B → b         e → E
*/



//65~90 대문자
//97~122 소문자
