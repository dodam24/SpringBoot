package class_;

public class SungJukMain { //위에 있는 SungJuk Class는  메뉴판에 불과하므로 SungJukMain에서 불러와야 함
	
	public static void main(String[] args) {
		SungJuk s;
		s = new SungJuk();
		s.setData("홍길동", 91, 95, 100); //호출
		s.calc(); //호출
		System.out.println("---------------------------------------------------");
		System.out.println("이름  \t국어  \t영어  \t수학  \t총점  \t평균  \t학점");
		System.out.println("---------------------------------------------------");
		System.out.println(s.getName() + "\t" + s.getKor() + "\t" + s.getEng() + "\t" + s.getMath() + "\t" + s.getTot() + "\t" + String.format("%.2f", s.getAvg()) + "\t" + s.getGrade());
	
		
		SungJuk s2;
		s2 = new SungJuk();
		s2.setData("프로도", 100, 89, 75); //호출
		s2.calc(); //호출
		System.out.println(s2.getName() + "\t" + s2.getKor() + "\t" + s2.getEng() + "\t" + s2.getMath() + "\t" + s2.getTot() + "\t" + String.format("%.2f", s2.getAvg()) + "\t" + s2.getGrade());
		
		
		SungJuk s3;
		s3 = new SungJuk();
		s3.setData("죠르디", 75, 80, 48); //호출
		s3.calc(); //호출
		System.out.println(s3.getName() + "\t" + s3.getKor() + "\t" + s3.getEng() + "\t" + s3.getMath() + "\t" + s3.getTot() + "\t" + String.format("%.2f", s3.getAvg()) + "\t" + s3.getGrade());
		System.out.println("---------------------------------------------------");

	}

}

/*
[문제] 성적 처리
- 총점, 평균, 학점을 구하시오
- 평균은 소수이하 2째자리까지 출력

총점 = 국어 + 영어 + 수학
평균 = 총점 / 과목수
학점은 평균이 90 이상이면 'A'
      평균이 80 이상이면 'B'
      평균이 70 이상이면 'C'
      평균이 60 이상이면 'D'
      그외는 'F'
      
클래스명 	: SungJuk
필드    	: name, kor, eng, math, tot, avg, grade  
메소드   : setData(이름, 국어, 영어, 수학)
         calc() - 총점, 평균, 학점 계산
         getName()
         getKor()
         getEng()
         getMath()
         getTot()
         getAvg()
         getGrade()
         
클래스명 : SungJukMain         

[실행결과]
----------------------------------------------------
이름      국어      영어      수학      총점      평균      학점
----------------------------------------------------
홍길동   90      95      100
----------------------------------------------------
*/