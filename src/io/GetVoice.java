package io;


import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.TargetDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;

public class GetVoice {
	byte []buffer; //sammlung an Samples von denen dann der min und Max wert geholt wird
	byte []b = new byte[2];
	
	int []min;
	int []max;
	int length=-1;
	int z=0;
	int start=-1;
	
	short [] samples;
	short [] samplesEnd;
	
	T1 t1;
	AudioFormat format = new AudioFormat(8000, 16, 1, true, false); // gesammt gr��e = Zeit * 8000
	DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
	AudioInputStream in;
	URL url = GetVoice.class.getResource("compare.wav");
	final File f = new File(url.getPath());
	int vergleich=1000;
	
	public void start(){
		try {
			final TargetDataLine line = (TargetDataLine) AudioSystem.getLine(info);
			line.open();
			System.out.println("aufnehmen...");
			line.start();
			Thread t = new Thread(){
				public void run(){
					AudioInputStream in = new AudioInputStream(line);
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
				t1 = new T1();
				t1.start();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		} catch (LineUnavailableException e) {
			e.printStackTrace();
		}
	}
	public boolean chekForRealEnd(){
		for (int j = 0; j < 100; j++) {
			if(samples[start+z+j]>vergleich||samples[start+z+j]<-vergleich){
							return false;
				}
			}
		return true;
	}
	public void getSamplesLength(){
		loopl:while(true){
				if(samples[start+z]<vergleich||samples[start+z]>-vergleich){
					if(chekForRealEnd()){//auf das richtige ende schauen nicht da er irgendwo mittendrinne abbricht
						break loopl;
					}else{//wenn fehl alarm weiter machen
				length+=1;
				z++;
					}
				}else{
					length+=1;
					z++;
				}
				}
	}
	public void getSamplesInfos(){
		loopl:for (int i = 0; i < samples.length; i++) {
			if(samples[i]>vergleich||samples[i]<-vergleich){
				if(start==-1){start=i;}//setzte den start fest von wo aus die Samples die wichtig sind starten
				getSamplesLength();
				break loopl;//die wichtigen samples bekommen also alle gr��er als 100 oder keiner als -100
			}
		}
	}
	class T1 extends Thread{
		public void run(){
			samples = new short[(int)(f.length()-44)/2];
			buffer = new byte[(int) (f.length()-44)];
			int h=0;
			try {
				in = AudioSystem.getAudioInputStream(f);
				in.read(buffer, 0, buffer.length);//lese die ganze Datei in den buffer
				System.out.println("fertig");
				for (int i = 0; i < samples.length; i++) {// immer 2 bytes = 1 sample
					b[0]=buffer[i];
					b[1]=buffer[i+=1];
					ByteBuffer bb = ByteBuffer.wrap(b);
					bb.order(ByteOrder.LITTLE_ENDIAN);
					samples[h]= bb.getShort();
					h++;
				}
				 h=0;
				 getSamplesInfos();//den start der Samples bekommen
				 samplesEnd = new short[length];
				 
				 for (int i = 0; i < length; i++) {
					samplesEnd[i]=samples[start+i];
				}
				 int x=0;
				 int y=0;
				 boolean positiv;
				 min = new int [length];
				 max = new int [length];
				 min[0]=0;
				 max[0]=0;
				 if(samplesEnd[0]<0){
					positiv=false;
				}else{
					positiv=true;
				}
				 for (int i = 0; i < max.length-1; i++) {
					 if(positiv){
						if(samplesEnd[i]>0&&samplesEnd[i]>max[x]){
							max[x]=samplesEnd[i];
							if(samplesEnd[i+1]<max[x]){
								positiv=false;
								if(x<length-1){
							x++;
							max[x]=0;
							}
							}
						}
					 }else if (!positiv){
						if(samplesEnd[i]<0&&samplesEnd[i]<min[y]){
							min[y]=samplesEnd[i];
							if(samplesEnd[i+1]>min[y]){
								positiv=true;
							if(y<length-1){
							y++;
							min[y]=0;
							}
							}
						}
					 }
				 }
				 
			} catch (IOException e) {e.printStackTrace();} catch (UnsupportedAudioFileException e) {
				e.printStackTrace();
			}
		}
	}
}
