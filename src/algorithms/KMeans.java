package algorithms;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class KMeans {
	private int k;
	private int n;
	private double[][] x;
	private double threshhold = 0.000000001;
	private ArrayList<Integer>[] clusters;
	
	public double destortion;
	public double[][] codeVectors;
	
	@SuppressWarnings("unchecked")
	public KMeans(int k, double[][] x) {
		this.k = k;
		this.x = x;
		n = x[0].length;
		for (int i = 1; i < x.length; i++) {
			if(x[i].length!=n) {
				throw new RuntimeException("Ensure that all input vectors have the same dimensions!");
			}
		}
		codeVectors = new double[k][n];
		clusters = new ArrayList[k];
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
		createCodebook();
	}
	
	@SuppressWarnings("unchecked")
	public KMeans(double[][] x, String path, int k) {
		this.k = k;
		this.x = x;
		n = x[0].length;
		for (int i = 1; i < x.length; i++) {
			if(x[i].length!=n) {
				throw new RuntimeException("Ensure that all input vectors have the same dimensions!");
			}
		}
		loadCodebook(path);
		if(this.k != codeVectors.length) {
			throw new RuntimeException("The codebook file "+path+" does not contain the right amount of Vectors("+k+")!");
		}
		for (int i = 0; i < codeVectors.length; i++) {
			if(codeVectors[i].length!=n) {
				throw new RuntimeException("Ensure that the input vectors have the same dimensions as the codebook vectors!");
			}
		}
		clusters = new ArrayList[k];
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
		updateCodebook();
	}
	
	@SuppressWarnings("unchecked")
	public KMeans(String path, int k) {
		this.k = k;
		loadCodebook(path);
		if(this.k != codeVectors.length) {
			throw new RuntimeException("The codebook file "+path+" does not contain the right amount of Vectors("+k+")!");
		}
		this.n = codeVectors[0].length;
		for (int i = 0; i < codeVectors.length; i++) {
			if(codeVectors[i].length!=n) {
				throw new RuntimeException("Ensure that the codebook vectors all have the same dimensions!");
			}
		}
		clusters = new ArrayList[k];
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
	}
	
	public int[] classify(double[][] x) {
		double minDestortion;
		int minIndex;
		double destortion;
		int[] back = new int[x.length];
		
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
		for (int i = 0; i < x.length; i++) {
			minDestortion = destortion(x[i], codeVectors[0]);
			minIndex = 0;
			for (int j = 1; j < codeVectors.length; j++) {
				destortion = destortion(x[i], codeVectors[j]);
				if(destortion<minDestortion) {
					minDestortion = destortion;
					minIndex = j;
				}
			}
			back[i] = minIndex;
		}
		return back;
	}
	
	//update an already existing codebook with the given x;
	public void updateCodebook() {
		double desOld;
		do {
			classify();
			desOld = overAllDestortion();
			reestimateCentroids();
		}while(desOld-overAllDestortion()>threshhold);
		destortion = overAllDestortion();
	}
	
	public void saveCodebook(String path){
		File f = new File(path+"\\"+k+".codebook");
		try {
			f.createNewFile();
			BufferedWriter b = new BufferedWriter(new FileWriter(f));
			b.write(Arrays.deepToString(codeVectors));
			System.out.println(Arrays.deepToString(codeVectors));
			b.close();
			
		} catch (IOException e) {e.printStackTrace();}
	}
	
	private void loadCodebook(String path) {
		File f = new File(path+"\\"+k+".codebook");
		try {
			BufferedReader br = new BufferedReader(new FileReader(f));
			String s = br.readLine();
			br.close();
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
			codeVectors = result.clone();
		} catch (IOException e) {e.printStackTrace();}
		
	}
	
	//create a codebook of size k given x;
	private void createCodebook() {
		randomIniCodeVectors();
		double desOld;
		int iterations = 0;
		do {
			iterations++;
			classify();
			desOld = overAllDestortion();
//			System.out.println(desOld);
			reestimateCentroids();
//			System.out.println(overAllDestortion());
		}while(desOld-overAllDestortion()>threshhold);
		destortion = overAllDestortion();
		System.out.println("final destortion: "+destortion);
		System.out.println("terations: "+iterations);
	}
	
	private void randomIniCodeVectors() {
		double[] min = new double[n];
		double[] max = new double[n];
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				if(x[i][j]<min[j]) {
					min[j] = x[i][j];
				}else if(x[i][j]>max[j]) {
					max[j] = x[i][j];
				}
			}
		}
		for (int i = 0; i < codeVectors.length; i++) {
			for (int j = 0; j < codeVectors[i].length; j++) {
				codeVectors[i][j] = Math.random()*(Math.abs(min[j])+Math.abs(max[j]))+min[j];
			}
		}
	}
	
	private void classify() {
		double minDestortion;
		int minIndex;
		double destortion;
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
		for (int i = 0; i < x.length; i++) {
			minDestortion = destortion(x[i], codeVectors[0]);
			minIndex = 0;
			for (int j = 1; j < codeVectors.length; j++) {
				destortion = destortion(x[i], codeVectors[j]);
				if(destortion<minDestortion) {
					minDestortion = destortion;
					minIndex = j;
				}
			}
			clusters[minIndex].add(i);
		}
	}
	
	private void reestimateCentroids() {
		double[] sum;
		double[] tempVector;
		for (int i = 0; i < clusters.length; i++) {
			if(clusters[i].size()!=0) {
				sum = new double[n];
				for (int j = 0; j < clusters[i].size(); j++) {
					tempVector = x[clusters[i].get(j)];
					for (int j2 = 0; j2 < tempVector.length; j2++) {
						sum[j2] += tempVector[j2];
					}
				}
				for (int j = 0; j < sum.length; j++) {
					sum[j] /= clusters[i].size();
				}
				codeVectors[i] = sum.clone();
			}
		}
	}
	
	private double  destortion(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < y.length; i++) {
			sum += (x[i]-y[i])*(x[i]-y[i]);
		}
		sum /= x.length;
	return sum;
	}
	
	//average destortion over all clusters
	private double overAllDestortion() {
		double[] sum = new double[clusters.length];
		double out = 0;
		for (int i = 0; i < clusters.length; i++) {
			if(clusters[i].size()!=0) {
				sum[i] = 0;
				for (int j = 0; j < clusters[i].size(); j++) {
					//System.out.println(codeVectors[i].length);
					sum[i] += destortion(x[clusters[i].get(j)], codeVectors[i]);
				}
				sum[i] /= clusters[i].size();
			}
		}
		for (int i = 0; i < sum.length; i++) {
			out += sum[i];
		}
		out /= (clusters.length*x.length);
		return out;
	}
	
//	public static void main(String[] args) {
//		double[][] x = {{1,1},{2,2},{3,3},{4,4},{1,4},{2,3},{3,2},{4,1} , {10,10},{9,9},{8,8},{7,7},{7,10},{8,9},{9,8},{10,7},{7,8}};
//		KMeans k = new KMeans(2, x);
//		k.createCodebook();
//		System.out.println(Arrays.deepToString(k.codeVectors));
//		k.saveCodebook("C:\\Users\\ben-g\\Desktop\\train\\Aurora");
//		k.loadCodebook("C:\\Users\\ben-g\\Desktop\\train\\Aurora");
//		System.out.println(Arrays.deepToString(k.codeVectors));
//		System.out.println(Arrays.toString(k.classify(x)));
//	}
}
