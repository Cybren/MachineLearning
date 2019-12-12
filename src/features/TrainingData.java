package features;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import algorithms.KMeans;

public class TrainingData {
	
	private FileInputStream in;
	private double[] samples;
	private ByteBuffer bb;
	private int sampleRate = 16000;//in Hz
	private double frameWidth = 0.025;//25 ms 
	private double frameStride = 0.01;//10ms
	private int frameSize;//in samples
	private int frameStep;//in samples
	private int frames;
	private int fftSize = 512;
	private FeatureBuilder fb;
	
	public double[][] features;
	public int [] obs;
	
	public TrainingData(File f) {
		//read File
		samples = getSamples(f);
		samples = preemphasis(samples);
		frameStep = (int) Math.round(sampleRate*frameStride);//160 samples okay
		frameSize = (int) Math.round(sampleRate*frameWidth);//400 samples okay
		frames = (int)Math.round((samples.length-frameSize)/frameStep);
		fb = new FeatureBuilder(createFrames());
		features = fb.features.clone();
	}
	
	public void setObs(KMeans km) {
		obs = km.classify(features);
	}
	
	private double[] getSamples(File f){
		System.out.println(f.length());
		byte[] b = new byte[(int) (f.length()-44)];
		samples = new double[b.length/2];
		try {
			in = new FileInputStream(f);
			for (int i = 0; i < b.length+44; i++) {
				if(i>43) {
					b[i-44]=(byte) in.read();
				}else {
					in.read();
				}
			}
			bb= ByteBuffer.wrap(b);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < samples.length; i++) {
				samples[i]=bb.getShort();
			}
			return samples;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	private double[] preemphasis(double [] signal) { //okay
		double alpha = 0.95; 
		double []temp = signal.clone();
		for (int i = 1; i < signal.length; i++) {
			temp[i]=signal[i]-alpha*signal[i-1];
		}
		return temp;
	}
	
	//frame the signal
	private double[][] createFrames() {
		double real[][] = new double[frames][fftSize];
		for (int i = 0; i < frames; i++) {
			for (int j = 0; j < frameSize; j++) {
				real[i][j] = samples[i*frameStep+j];
			}
		}
		return real;
	}
}