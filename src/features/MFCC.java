package features;

import algorithms.FFT;

public class MFCC {
	
	FFT fft;
	private double[] signal;
	private int sampleRate;
	private int numFilters = 40;
	private int upperFrequenzy = 8000, lowerFrequency = 300;
	
	public double[] mfccs;
	public double energy;
	
	public MFCC(double[] signal,int sampleRate) {
		this.signal = signal.clone();
		this.sampleRate = sampleRate;
		if(sampleRate/2<upperFrequenzy) {
			upperFrequenzy = sampleRate/2;
		}
		setMFCCS();
		setEnergy();
	}
	
	private void setMFCCS() {
		double [] window = getHammingWindow(signal.length);
		double [] signalWindow = new double[signal.length];
		for (int i = 0; i < window.length; i++) {
			signalWindow[i] = signal[i]*window[i];
		}
		
		//set the imaginary part of the complex number for the fft i.e. set everything 0;
		double [] imag = new double[signal.length];
		for (int i = 0; i < window.length; i++) {
			imag[i] = 0;
		}
		
		//compute FFT and powerspectrum
		fft = new FFT(signal, imag);
		//only take half of the powerspectrum cause of the symetrie
		double[] half = new double[signal.length/2+1];
		for (int i = 0; i < half.length; i++) {
			half[i]=fft.power[i];
		}
		
		//apply melfilterbank
		double[][] filterbank = getFilterbank();
		double[][] halfFiltered = new double[filterbank.length][filterbank[0].length];
		for (int i = 0; i < halfFiltered.length; i++) {
			for (int j = 0; j < halfFiltered[0].length; j++) {
				halfFiltered[i][j] = half[j]*filterbank[i][j];
			}
		}
		
		double [] sum= new double[halfFiltered.length];
		for (int i = 0; i < halfFiltered.length; i++) {
			for (int j = 0; j < halfFiltered[i].length; j++) {
				sum[i]+=halfFiltered[i][j];
			}
		}
		
		double []logedSum = new double[sum.length];
		for (int i = 0; i < logedSum.length; i++) {
			logedSum[i] = Math.log(sum[i]);
		}
		mfccs = DCT(logedSum);
	}
	
	private double[] getHammingWindow(int length) {
		double out[] = new double[length];
		for (int i = 0; i < out.length; i++) {
			out[i]=0.54-0.46*Math.cos((2*Math.PI*i)/(length-1));
		}
		return out;
	}
	
	private double[][] getFilterbank() {
		int windowLength = signal.length/2+1;
		int fftLength = signal.length;
		
		double [][] out = new double[numFilters][windowLength];
		double upperMel=hzToMel(upperFrequenzy);
		double lowerMel=hzToMel(lowerFrequency);
		double space = (upperMel-lowerMel)/(numFilters+1);
		
		double [] points = new double[numFilters+2];
		for (int i = 0; i < points.length; i++) {
			points[i]=lowerMel+(space*i);
		}
		
		double[] pointsHz = new double[numFilters+2];
		for (int i = 0; i < pointsHz.length; i++) {
			pointsHz[i]=melToHz(points[i]);
		}
		
		double[] pointsFloored = new double[numFilters+2];
		for (int i = 0; i < pointsFloored.length; i++) {
			pointsFloored[i]=Math.floor((fftLength+1)*pointsHz[i]/sampleRate);
		}
		
		for (int i = 0; i < out.length; i++) {
			for (int j = 0; j < out[i].length; j++) {
				if(j<pointsFloored[i]) {
					out[i][j]=0;
				}else if(j>=pointsFloored[i]&&j<pointsFloored[i+1]) {
					out[i][j-1]=(j-pointsFloored[i])/(pointsFloored[i+1]-pointsFloored[i]);
				}else if(j>=pointsFloored[i+1]&&j<pointsFloored[i+2]) {
					out[i][j-1]=(pointsFloored[i+2]-j)/(pointsFloored[i+2]-pointsFloored[i+1]);
				}else if(j>pointsFloored[i+2]) {
					out[i][j]=0;
				}
			}
		}
		return out;
	}
	
	private double hzToMel(double hz) {
		return (1125*Math.log(1+hz/700));
	}
	
	private double melToHz(double mel) {
		return (700*(Math.exp(mel/1125)-1));
	}
	
	private void setEnergy(){
		energy = 0;
		for (int i = 0; i < signal.length; i++) {
			energy += signal[i]*signal[i];
		}
	}
	
	private double[] DCT(double[] input){
    	double[] back = new double[input.length];
    	double ze;
    	for (int i = 0; i < back.length; i++) {
    		ze = 0;
			for (int j = 0; j < back.length; j++) {
				ze += input[j]*Math.cos(((Math.PI/input.length)*(j+0.5)*i));
			}
			back[i] = ze;
		}
    	return back;
    }
}
