package io;

import java.io.File;
import java.io.IOException;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.TargetDataLine;

public class WriteWavFile {

	public static void main(String args[]){
		AudioFormat format = new AudioFormat(8000, 16, 2, true, false);
		
		DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
		
		try {
			final TargetDataLine line = (TargetDataLine) AudioSystem.getLine(info);
			line.open();
			System.out.println("aufnehmen...");
			line.start();
			Thread t = new Thread(){
				public void run(){
					AudioInputStream in = new AudioInputStream(line);
					File f = new File("hallo.wav");
					try {
						AudioSystem.write(in, AudioFileFormat.Type.WAVE, f);
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			};
			t.start();
			try {
				Thread.sleep(3000);
				line.stop();
				line.close();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		} catch (LineUnavailableException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
