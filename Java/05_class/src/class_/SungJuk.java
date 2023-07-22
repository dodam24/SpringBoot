package class_;

public class SungJuk {
	private String name;
	private int kor, eng, math, tot;
	private double avg;
	private char grade;
	
	public void setData(String n, int k, int e, int m) { //함수의 구현 (입력만!)
		name = n;
		kor = k;
		eng = e;
		math = m;
	}
	public void calc() { //(계산만!) 내보내는 것도 없으므로 void (내보내는 것은 아래에 있는 getter가 수행)
		tot = kor + eng + math;
		avg = (double)tot / 3;
		if(avg >= 90) grade = 'A';
		else if(avg >= 80) grade = 'B';
		else if(avg >= 70) grade = 'C';
		else if(avg >= 60) grade = 'D';
		else grade = 'F';
	}
	public String getName() { //(받는 것만!)
		return name;
	}
	public int getKor() {
		return kor;
	}
	public int getEng() {
		return eng;
	}
	public int getMath() {
		return math;
	}
	public int getTot() {
		return tot;
	}
	public double getAvg() {
		return avg;
	}
	public char getGrade() {
		return grade;
	}
}