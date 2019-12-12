package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import algorithms.KMeans;
import features.TrainingData;
import statisticalmodels.HMM;

public class HMMTrainer {
	private HMM hmm;
	private KMeans km;
	private TrainingData[] all, train, test,negativ;
	private String path;
	private int numStates, numEmissions;
	public HMMTrainer(String path, int samples, int negatives) {
		this.path = path;
		all = new TrainingData[samples];
		int temp = (int)Math.floor(samples*3/4);
		System.out.println(temp);
		train = new TrainingData[temp];
		test = new TrainingData[samples-temp];
		for (int i = 0; i < all.length; i++) {
			System.out.println(i);
			all[i] = new TrainingData (new File(path+"\\"+i+".wav"));
		}
		negativ = new TrainingData[negatives];
		for (int i = 0; i < negativ.length; i++) {
			negativ[i] = new TrainingData(new File(path+"\\negativ\\"+i+".wav"));
		}
		splitData();
		setStructure();
		hmm = createHMM();
		setKMeans();
	}

	
	public void train() {
		for (int i = 0; i < test.length; i++) {
			test[i].setObs(km);
			System.out.println("obs: "+Arrays.toString(test[i].obs));
		}
		for (int i = 0; i < train.length; i++) {
			train[i].setObs(km);
		}
		for (int i = 0; i < negativ.length; i++) {
			negativ[i].setObs(km);
		}
		int input[][][] = new int[3][train.length/3][];
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				input[i][j] = train[i*j+j].obs.clone();
			}
		}
		
		int iteration = 0;
		double sum = 0, sumLast=1;
		double temp;
		for (int i = 0; i < input.length; i++) {
			hmm.baumWelch(input[i]);
		}
		
		sum = 0;
		for (int i = 0; i < test.length; i++) {
			temp = hmm.evalLog(test[i].obs);
			if(!Double.isNaN(temp)) {
				sum += temp;
				System.out.println(temp);
			}
		}
		System.out.println("Prob at itertion "+iteration+": "+sum/test.length+"\n");
		for (int i = 0; i < negativ.length; i++) {
			System.out.println("negativ-"+i+": "+hmm.evalLog(negativ[i].obs));
		}
		do{
			sumLast = sum;
			iteration++;
			for (int i = 0; i < input.length; i++) {
				hmm.baumWelch(input[i]);
				sum = 0;
				for (int j = 0; j < test.length; j++) {
					temp = hmm.evalLog(test[j].obs);
					if(!Double.isNaN(temp)) {
						sum += temp;
						System.out.println(temp);
					}
				}
				for (int j = 0; j < negativ.length; j++) {
					System.out.println("negativ-"+j+": "+hmm.evalLog(negativ[j].obs));
				}
			}
			System.out.println("Prob at itertion "+iteration+": "+sum/test.length+"\n");
		}while(sumLast>sum||iteration<20);
	}
	
	private void splitData(){
		ArrayList<TrainingData> temp = new ArrayList<TrainingData>();
		for (int i = 0; i < all.length; i++) {
			temp.add(all[i]);
		}
		int x;
		for (int i = 0; i < train.length; i++) {
			x = (int)Math.floor(Math.random()*((double)temp.size()-1));
			System.out.println(x);
			train[i] =
					temp.remove(x);
		}
		for (int i = 0; i < test.length; i++) {
			test[i] = temp.get(i);
		}
		System.out.println(Arrays.toString(train));
		System.out.println(Arrays.toString(test));
	}
	
	private void setStructure() {
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(new File(path+"\\hmm.structure")));
			String temp = br.readLine();
			numStates = Integer.parseInt(temp.split(" ")[1]);
			temp = br.readLine();
			numEmissions = Integer.parseInt(temp.split(" ")[1]);
			br.close();
		} catch (IOException e) {e.printStackTrace();}
	}
	
	private HMM createHMM() {
		double[][] a = new double[numStates][numStates];
		double sum;
		double delta = 2;
		for (int i = 0; i < a.length; i++) {
			sum = 0;
			for (int j = 0; j < a[i].length; j++) {
				if(i<=j && !(j>i+delta)) {
					a[i][j] = Math.random();
					sum += a[i][j];
				}
			}
			for (int j = 0; j < a[i].length; j++) {
				a[i][j] /= sum;
			}
		}
		double[][] b = new double[numStates][numEmissions];
		for (int i = 0; i < b.length; i++) {
			sum = 0;
			for (int j = 0; j < b[i].length; j++) {
				b[i][j] = Math.random();
				sum += b[i][j];
			}
			for (int j = 0; j < b[i].length; j++) {
				b[i][j] /= sum;
			}
		}
		double[] ini = new double[numStates];
		ini[0] = 1;
		return new HMM(a,b,ini);
	}
	
	private void setKMeans() {
		File codeBook = new File(path+"\\"+numEmissions+".codebook");
		if(!codeBook.exists()) {
			int sum = 0;
			for (int i = 0; i < all.length; i++) {
				sum += all[i].features.length;
			}
			double[][] x = new double[sum][];
			int index = 0;
			for (int i = 0; i < all.length; i++) {
				for (int j = 0; j < all[i].features.length; j++) {
					x[index] = all[i].features[j].clone();
					index++;
				}
			}
			KMeans[] temp = new KMeans[100];
			for (int i = 0; i < temp.length; i++) {
				System.out.println("creating KMeans nr. "+i);
				temp[i] = new KMeans(numEmissions,x);
			}
			km = temp[0];
			for (int i = 1; i < temp.length; i++) {
				//System.out.println(temp[i].destortion);
				if(temp[i].destortion<km.destortion) {
					km = temp[i];
				}
			}
			System.out.println(km.destortion);
			km.saveCodebook(path);
		}else {
			km = new KMeans(path, numEmissions);
		}
	}
	
	public void saveHMM(String path){
		File f = new File(path+"\\hiddenMarkov.model");
		try {
			f.createNewFile();
			BufferedWriter b = new BufferedWriter(new FileWriter(f));
			b.write(Arrays.deepToString(hmm.getA()));
			System.out.println(Arrays.deepToString(hmm.getA()));
			b.write(Arrays.deepToString(hmm.getB()));
			System.out.println(Arrays.deepToString(hmm.getB()));
			b.write(Arrays.toString(hmm.getIni()));
			System.out.println(Arrays.toString(hmm.getIni()));
			b.close();
		} catch (IOException e) {e.printStackTrace();}
	}
	
	public void loadHMM(String path) {
		File f = new File(path+"\\code.book");
		try {
			BufferedReader br = new BufferedReader(new FileReader(f));
			String s = br.readLine();
			
			s = s.substring(1, s.length()-1);
			s = s.replace("[", "");
			String strings[] = s.split("],");
			String[] temp;
			double[][] result = new double[strings.length][];
			for (int i = 0; i < strings.length; i++) {
				temp = strings[i].replace("]", "").split(",");
				result[i] = new double[temp.length];
				for (int j = 0; j < temp.length; j++) {
					result[i][j] = Double.parseDouble(temp[j]);
				}
			}
			hmm.setA(result.clone());
			
			s = br.readLine();
			s = s.substring(1, s.length()-1);
			s = s.replace("[", "");
			strings = s.split("],");
			result = new double[strings.length][];
			for (int i = 0; i < strings.length; i++) {
				temp = strings[i].replace("]", "").split(",");
				result[i] = new double[temp.length];
				for (int j = 0; j < temp.length; j++) {
					result[i][j] = Double.parseDouble(temp[j]);
				}
			}
			hmm.setB(result.clone());
			
			s = br.readLine();
			temp = s.replace("[", "").replace("]", "").split(",");
			double[] out = new double[temp.length];
			for (int j = 0; j < temp.length; j++) {
				out[j] = Double.parseDouble(temp[j]);
			}
			hmm.setIni(out.clone());
			br.close();
		} catch (IOException e) {e.printStackTrace();}
		
	}

	public static void main(String[] args) {
		HMMTrainer t = new HMMTrainer("C:\\Users\\ben-g\\Desktop\\train\\Aurora", 100,16);
		t.train();
	}
}
