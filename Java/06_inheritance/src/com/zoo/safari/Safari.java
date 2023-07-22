package com.zoo.safari;

import com.zoo.Zoo;

public class Safari extends Zoo {

	public static void main(String[] args) {
		Zoo z = new Zoo();
		z.tiger();
		// z.giraffe(); // protected는 다른 패키지 자식 클래스까지 가능
		// z.elephant();
		// z.lion();
		System.out.println();
		
		Safari s = new Safari();
		s.tiger();
		s.giraffe(); // 자식 클래스로 생성해야 가능 (protected)
		// s.elephant();
		// s.lion();

	}

}
