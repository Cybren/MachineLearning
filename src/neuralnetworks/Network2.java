package neuralnetworks;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class Network2 {
	String path = "C:/Users/ben-g/Desktop/HW/";
	int[] layer;
	Layer[] layers;
	double learningrate = 0.00095;
	
	public Network2(int[]layer){
		this.layer = layer;
		
		layers = new Layer[layer.length-1];
		for (int i = 0; i < layers.length; i++) {
			layers[i]= new Layer(layer[i],layer[i+1],learningrate);
		}
	}

	public double[] feedForward(double[]inputs){
	
		layers[0].feedForward(inputs);
		
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i-1].outputs);
		}
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double []expected){
		layers[layers.length-1].backPropOutput(expected);
		for (int i = layers.length-2; i > -1; i--) {
				layers[i].backPropHidden(layers[i+1].gamma, layers[i+1].weights);
		}
		for (int i = 0; i < layers.length; i++) {
			layers[i].update();
		}
	}
	
	public void saveANN(){
		String out ="";
		List<String> lines= new LinkedList<String>(Arrays.asList(""));
		for (int i = 0; i < layer.length; i++) {
			out+=layer[i]+" ";
		}
		out+=System.lineSeparator();
		for (int i = 0; i < layers.length; i++) {
			for (int j = 0; j < layers[i].numberOfOutputs; j++) {
				for (int j2 = 0; j2 < layers[i].numberOfInputs; j2++) {
					out+=layers[i].weights[j][j2]+" ";
				}
				out+=System.lineSeparator();
			}
			out+=System.lineSeparator();
		}
		for (int i = 0; i < layers.length; i++) {
			for (int j = 0; j < layers[i].numberOfOutputs; j++) {
				out+=layers[i].bias[j]+" ";
			}
			out+=System.lineSeparator();
		}
		lines.add(out);
		Path p = FileSystems.getDefault().getPath(path,"Handwrite0055.ann");
		try {
			Files.delete(p);
			Files.write(p, lines,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);
		} catch (IOException e) {e.printStackTrace();}
	}
	
	public class Layer{
		
		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of nurons in curent layer
		double learningrate;
		double [] outputs;
		double [] inputs;
		double [][] weights;
		double [][] weightsDelta;
		double [] z;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		
		public Layer(int numberOfInputs, int numberOfOutputs,double rate){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			
			outputs = new double[numberOfOutputs];
			z = new double[numberOfOutputs];
			inputs = new double [numberOfInputs];
			
			weights = new double[numberOfOutputs][numberOfInputs];
			weightsDelta = new double[numberOfOutputs][numberOfInputs];
			
			gamma = new double[numberOfOutputs];
			error = new double[numberOfOutputs];
			
			bias = new double[numberOfOutputs];
			biasDelta = new double[numberOfOutputs];
			
			ini();
		}
		
		public void ini(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Math.random()-0.5;
				}
				bias[i]= Math.random()-0.5;
			}
		}
		
		public double[] feedForward(double []inputs){
			this.inputs = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				z[i]=0;
				for (int j = 0; j < numberOfInputs; j++) {
					z[i]+= inputs[j]*weights[i][j];
				}
				z[i]+=bias[i];
				outputs[i] = sigmoid(z[i]);
			}
			
			return outputs;
		}
		
		public void backPropOutput(double[] expected){
			for (int i = 0; i < numberOfOutputs; i++) {
				error[i] = outputs[i]-expected[i];
				gamma[i] = error[i]*sigmoidPrime(z[i]);
			}
			for (int i = 0; i < weightsDelta.length; i++) {
				for (int j = 0; j < weightsDelta[i].length; j++) {
					weightsDelta[i][j]= gamma[i]*inputs[j];
				}
				biasDelta[i]=gamma[i];
			}
		}
		public void backPropHidden(double[]gammaForward, double [][] weightsForward){
			for (int i = 0; i < numberOfOutputs; i++) {
				gamma[i] = 0;
				for (int j = 0; j < gammaForward.length; j++) {
					gamma[i] += gammaForward[j]*weightsForward[j][i];
				}
				gamma[i]*= sigmoidPrime(z[i]);
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]= gamma[i]*inputs[j];
				}
				biasDelta[i] = gamma[i];
			}
		}
		
		public void update(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] -= weightsDelta[i][j]*learningrate;
				}
				bias[i]-=biasDelta[i];
			}
		}
		
		public double sigmoid(double x){
			return(1/(1+Math.exp(-x)));
		}
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}
	}
	
	public boolean train(){//umschreiben, das hier die epochs etc erstellt werden und dann x mal train durchgeführt wird
		//----------------------------initialize-------------------------------------
		int e =0;
		double error = 1.0;
		int u=28*28;//number of pixels
		int epochsize = 10;
		double[] trainingImages = loadTrainingImages();
		byte[] trainingNumbers = loadTrainingNumbers();
		double[] testImages = loadTestImages();
		byte[] testNumbers = loadTestNumbers();
		double[][] pictures = new double[trainingImages.length/u][u];
		for (int i = 0; i < pictures.length; i++) {
			for (int j = 0; j < u; j++) {
				pictures[i][j]=trainingImages[(i*u)+j];
			}
		}
		double[][] tPictures = new double[testImages.length/u][u];
		for (int i = 0; i < tPictures.length; i++) {
			for (int j = 0; j < u; j++) {
				tPictures[i][j]=testImages[(i*u)+j];
			}
		}
		//System.out.println(pictures.length);
		int [] newOrder = shuffle(pictures.length);//shuffle the pictures so everytime we train we get another order => no pattern
		double[][] newPictures = new double[trainingImages.length/u][u];
		for (int i = 0; i < pictures.length; i++) {
			newPictures[i]=pictures[newOrder[i]];
		}
		byte [] newNumbers = new byte[trainingNumbers.length];//shuffle numbers the same way
		for (int i = 0; i < newNumbers.length; i++) {
			newNumbers[i]=trainingNumbers[newOrder[i]];
		}
		int n=1000;
		double n2 = n;
		double[][][] newTP= new double[pictures.length/n][n][];
		for (int i = 0; i < newTP.length; i++) {
			for (int j = 0; j < newTP[i].length; j++) {
				newTP[i][j]=pictures[newTP.length*i+j];
			}
		}
		byte[][] newTN = new byte[trainingNumbers.length/n][n];
		for (int i = 0; i < newTN.length; i++) {
			for (int j = 0; j < newTN[i].length; j++) {
				newTN[i][j]=trainingNumbers[n*i+j];
			}
		}
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				System.out.print(Math.round(newTP[0][e][i*28+j]*1000.0)/1000.0+" ");
			}
			System.out.println();
		}
		System.out.println(newTN[0][e]);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				System.out.print(Math.round(newPictures[0][i*28+j]*1000.0)/1000.0+" ");
			}
			System.out.println();
		}
		System.out.println(newNumbers[0]);
		//System.out.println(Arrays.toString(newPictures[0]));
		//------------------------------train--------------------------------------
		int epoch =0;
		System.out.println(newPictures.length);
		while(error>0.055&&epoch<50){
		byte o = 0;
			for (int i = 0; i <newPictures.length/*/epochsize*/; i++) {//train the network
				feedForward(newPictures[i]);
				double[]train = new double[layers[layers.length-1].numberOfOutputs];//create Array for the desired result eg [0,0,0,0,1,0,0,0,0,0] for 4
				for (int j = 0; j < train.length; j++) {
					if(j==newNumbers[i]){
						train[j]=1;
					}else{
						train[j]=0;
					}
				}
				backProp(train);
			}
			System.out.println("training finished");
			int right=0;
			for (int h = 0; h < newTP[o].length; h++) {
				double high = -1.0;
				int at = -1;
				double[] out = feedForward(newTP[o][h]);
				for (int j = 0; j < out.length; j++) {
					if(out[j]>high){
						high = out[j];
						at = j;
					}
				}
				if(newTN[o][h]==at){
					right++;
				}
			}
			if(o+1<newTP.length){
				o++;
			}else{
				o=0;
			}
			System.out.println(layers[layers.length-1].weights[0][0]);
			error = (n2-right)/n2;
			System.out.println("error in epoch"+epoch+": "+error);
			epoch++;
		}	
		if(error<=0.055){
//			saveANN();
			return true;
		}else{
			return false;
		}
	}
	
	public int[] shuffle(int length) {
		int[] array= new int[length];
		  List<Integer> list = new ArrayList<>();
		  for (int i = 0; i < length; i++) {
			list.add(i);
		}

		  Collections.shuffle(list);

		  for (int i = 0; i < list.size(); i++) {
		    array[i] = list.get(i);
		  }
		  return array;
	}
	
	public byte[] loadTrainingNumbers(){//check
		byte[] trainingNumbers=null;
		Path trainNum = Paths.get(path+"train-labels.idx1-ubyte");
		try {
			trainingNumbers = Files.readAllBytes(trainNum);
		} catch (IOException e) {
			e.printStackTrace();
		}
		byte[] nums = new byte[trainingNumbers.length-8];
		
		for (int i = 0; i < nums.length; i++) {
			nums[i]=trainingNumbers[i+8];
		}
		trainingNumbers = nums;
		System.out.println("Traininglabels Loaded!");
		return trainingNumbers;
	}
	
	public double[] loadTrainingImages(){
		double[] trainingImages;
		Path trainImg = Paths.get(path+"train-images.idx3-ubyte");
		byte[] data={};
		try {
			data = Files.readAllBytes(trainImg);
		} catch (IOException e) {
			e.printStackTrace();
		}
		int[] temp = new int[data.length];
		trainingImages = new double[data.length-16];
		for (int i = 0; i < trainingImages.length; i++) {
			temp[i]= data[i+16]&0xFF;
			trainingImages [i] =temp[i]/255.0;
		}
		System.out.println("Trainingimages Loaded!");
		return trainingImages;
	}
	
	public byte[] loadTestNumbers(){
		byte[] testNumbers={};
		Path testNum = Paths.get(path+"t10k-labels.idx1-ubyte");
		try {
			testNumbers = Files.readAllBytes(testNum);
		} catch (IOException e) {
			e.printStackTrace();
		}
		byte[] nums = new byte[testNumbers.length-8];
		
		for (int i = 0; i < nums.length; i++) {
			nums[i]=testNumbers[i+8];
		}
		testNumbers = nums;
		System.out.println("Testlabels Loaded!");
		return testNumbers;
	}
	
	public double[] loadTestImages(){
		double[] testImages;
		Path testImg = Paths.get(path+"t10k-images.idx3-ubyte");
		byte[] data={};
		try {
			data = Files.readAllBytes(testImg);
		} catch (IOException e) {
			e.printStackTrace();
		}
		int[] temp = new int[data.length];
		testImages = new double[data.length-16];
		for (int i = 0; i < testImages.length; i++) {
			temp[i]= data[i+16]&0xFF;
			testImages [i] =temp[i]/*/255.0*/;
		}
		System.out.println("Testimages Loaded!");
		return testImages;
	}
	
	public static void main(String[] args) {
		
		int layers[] = {3,25,25,1};
		Network2 n = new Network2(layers);
		for (int i = 0; i < 5000; i++) {
			n.feedForward(new double[]{0,0,0});
			n.backProp(new double[]{0});
			
			n.feedForward(new double[]{0,0,1});
			n.backProp(new double[]{1});
			
			n.feedForward(new double[]{0,1,0});
			n.backProp(new double[]{1});
			
			n.feedForward(new double[]{0,1,1});
			n.backProp(new double[]{0});
			
			n.feedForward(new double[]{1,0,0});
			n.backProp(new double[]{1});
			
			n.feedForward(new double[]{1,0,1});
			n.backProp(new double[]{0});
			
			n.feedForward(new double[]{1,1,0});
			n.backProp(new double[]{0});
			
			n.feedForward(new double[]{1,1,1});
			n.backProp(new double[]{1});
		}
		System.out.println(n.feedForward(new double[]{0,0,0})[0]);
		System.out.println(n.feedForward(new double[]{0,0,1})[0]);
		System.out.println(n.feedForward(new double[]{0,1,0})[0]);
		System.out.println(n.feedForward(new double[]{0,1,1})[0]);
		System.out.println(n.feedForward(new double[]{1,0,0})[0]);
		System.out.println(n.feedForward(new double[]{1,0,1})[0]);
		System.out.println(n.feedForward(new double[]{1,1,0})[0]);
		System.out.println(n.feedForward(new double[]{1,1,1})[0]);
//		n.saveANN();
		int layers2 [] = {28*28,1000,1000,10};
		boolean b = false;
		while(!b) {
		n = new Network2(layers2);
		b = n.train();
		n.learningrate*=0.95;
		}
	}

}
