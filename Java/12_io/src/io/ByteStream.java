package io;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class ByteStream {

	public static void main(String[] args) throws IOException {
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("data.txt")));
		int data;
		
		while ( (data = bis.read()) != -1) {
			System.out.print((char)data); //Enter가 숫자 1310이 아닌 제대로 된 문자 형태로 출력  
		}
		System.out.println();
				
	}

}
