package neuralnetworks;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class RecurrentNetwork2 {
	String path = "C:\\workspace\\KNN\\src\\knn\\shakespear.txt";
	int[] layer;
	OutputLayer[] layers;
	double learningrate;
	int times;
	
	boolean gradientClipping = true;
	double gradientClippingThreshold = 0.5;
	
	boolean batchLearning = true;
	int batchSize = 15;
	int count = 0;
	
	public RecurrentNetwork2(int[]layer,int times,double learningrate){
		this.layer = layer;
		this.times=times;
		this.learningrate = learningrate;
		System.out.println(learningrate);
		layers = new OutputLayer[layer.length-1];
		for (int i = 0; i < layers.length-1; i++) {
			layers[i]= new HiddenLayer(layer[i],layer[i+1],learningrate,times);
			layers[i].setGradientClipping(gradientClipping,gradientClippingThreshold);
		}
		layers[layers.length-1]=new OutputLayer(layer[layer.length-2],layer[layer.length-1],learningrate,times);
		layers[layers.length-1].setGradientClipping(gradientClipping,gradientClippingThreshold);
	}

	public double[] feedForward(double[] inputs) {
		layers[0].feedForward(inputs,0);
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i-1].outputs[0],0);
		}
		return layers[layers.length-1].outputs[0];
	}
	public double[] feedForward(double[] inputs,int time) {
		layers[0].feedForward(inputs,time);
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i-1].outputs[time+1],time);
		}
		return layers[layers.length-1].outputs[time];
	}
	
	public double[][] feedForward(double[][]inputs){
		for (int h = 0; h < times; h++) {
			layers[0].feedForward(inputs[h],h);
			for (int i = 1; i < layers.length; i++) {
				layers[i].feedForward(layers[i-1].outputs[h+1],h);
			}
		}
		
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double [][]expected){
		if(batchLearning) {
			if(count < batchSize) {
				count++;
			}else {
				count = 0;
			}
			for (int i = 0; i < times; i++) {
				layers[layers.length-1].backPropOutput(expected[i],i);
				for (int j = layers.length-2; j > -1; j--) {
						layers[j].backPropHidden(layers[j+1].gamma,layers[j+1].weights,i);
				}
			}
			if(count==batchSize) {
				 System.out.println("update");
				for (int j = 0; j < layers.length; j++) {
					layers[j].update();
					layers[j].reset();
				}
			}
		}else {
			for (int i = 0; i < times; i++) {
				layers[layers.length-1].backPropOutput(expected[i],i);
				for (int j = layers.length-2; j > -1; j--) {
						layers[j].backPropHidden(layers[j+1].gamma,layers[j+1].weights,i);
				}
			}
			for (int j = 0; j < layers.length; j++) {
				layers[j].update();
				layers[j].reset();
			}
		}
	}
	
	public void setPast() {
		for (int i = 0; i < layers.length-1; i++) {
			layers[i].setPast(layers[i].outputs[layers[i].outputs.length-1]);
		}
	}
	
	public void saveANN(){// recurrent rein
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
	
	public char[] getWB(){
		List <Character> chars  = new ArrayList <Character>();
		File f = new File(path);
		BufferedReader bf;
		try {
			bf = new BufferedReader(new FileReader(f));
			int c = bf.read();
			while(c!=-1) {
				if(!chars.contains((char)c)) {
					chars.add((char)c);
				}
				c=bf.read();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		char[]out = new char[chars.size()];
		for (int i = 0; i < out.length; i++) {
			out[i]=chars.get(i);
		}
		return out;
	}
	
	public double[] getArray(char c, char[]wb) {
		double[] out = new double[wb.length];
		for (int i = 0; i < wb.length; i++) {
			if(wb[i]==c) {
				out[i]=1;
			}else {
				out[i]=0;
			}
		}
		return out;
	}
	
	public double[] getOneHot(int c, int length) {
		double[] out = new double[length];
		for (int i = 0; i < out.length; i++) {
			if(i==c) {
				out[i]=1;
			}else {
				out[i]=0;
			}
		}
		return out;
	}
	
	public int getOutput(double[]in) {
		int out = 0;
		double v = 0;
		for (int i = 0; i < in.length; i++) {
			if(in[i]>v) {
				out = i;
				v=in[i];
			}
		}
		return out;
	}
	
	public double[][] getInputs(){
		List <Character> chars  = new ArrayList <Character>();
		File f = new File(path);
		BufferedReader bf;
		List <Integer> a  = new ArrayList <Integer>();
		try {
			bf = new BufferedReader(new FileReader(f));
			int c = bf.read();
			while(c!=-1) {
				if(!chars.contains((char)c)) {
					chars.add((char)c);
				}
				a.add(c);
				c=bf.read();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		double [][] inputs = new double[(int) f.length()][chars.size()];
		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				int v =a.get(i) ;
				int v2=(int)chars.get(j);
				if(v==v2) {
					inputs[i][j]=1;
				}else {
					inputs[i][j]=0;
				}
			}
		}
		return inputs;
		
	}
	
	public String trans(char[] wb,double[][]out) {
		String s = "";
		for (int i = 0; i < out.length; i++) {
			for (int j = 0; j < out[i].length; j++) {
				if(out[i][j]==1) {
					s+=wb[j];
					break;
				}
			}
		}
		return s;
	}
	
	public String trans(char[] wb,double[]out) {
		String s = "";
		for (int i = 0; i < out.length; i++) {
			if(out[i]==1) {
				s+=wb[i];
				break;
			}
		}
		return s;
	}
	
	public class OutputLayer{
		public String type ="Output";

		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of nurons in curent layer
		double learningrate;
		double [][] outputs;//{time}{ouputs}
		double [][] z;
		double [][] inputs;//{time}{inputs}
		double [][] weights;
		double [][] weightsDelta;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		int times;
		
		boolean gradientClipping = false;
		double gradientClippingThreshold = 0.5;
		
		public void setGradientClipping(boolean b, double threshold) {
			gradientClipping = b;
			gradientClippingThreshold = threshold;
			
		}
		
		public OutputLayer(int numberOfInputs, int numberOfOutputs,double rate, int times){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			this.times = times;
			
			outputs = new double[times][numberOfOutputs];
			z = new double[times][numberOfOutputs];
			inputs = new double [times][numberOfInputs];
			
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
					weights[i][j] = (Math.random()-0.5)/1;
					weightsDelta[i][j]=0;
				}
				bias[i]= Math.random()-0.5;
				biasDelta[i]=0;
			}
		}
		public void reset(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weightsDelta[i][j]=0;
				}
				biasDelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				z[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {
					z[time][i]+= inputs[j]*weights[i][j];
				}
				z[time][i]+=bias[i];
				outputs[time][i] = sigmoid(z[time][i]);
			}
			outputs[time] = softmax(outputs[time]);
			return outputs[time];
		}
		
		public void backPropOutput(double[] expected,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
//				if(expected[i]==0) {
//					error[i] = (outputs[time][i]+0.5)*(outputs[time][i]+0.5);
//					gamma[i] = error[i]*sigmoidPrime(z[time][i]);
//				}else {
					error[i] = (outputs[time][i]-expected[i]);
					gamma[i] = error[i]*sigmoidPrime(z[time][i]);
//				}
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]+= gamma[i]*inputs[time][j];
				}
				biasDelta[i]+=gamma[i];
			}
		}
		
		public void update(){
			if(gradientClipping) {
				for (int i = 0; i < weights.length; i++) {
					for (int j = 0; j < weights[i].length; j++) {
						if(weightsDelta[i][j]*learningrate>gradientClippingThreshold) {
							weights[i][j] -= gradientClippingThreshold;
						}else if(weightsDelta[i][j]*learningrate<(-1)*gradientClippingThreshold) {
							weights[i][j] += gradientClippingThreshold;
						}else {
							weights[i][j] -= weightsDelta[i][j]*learningrate;///times;
						}
					}
					if(biasDelta[i]*learningrate>gradientClippingThreshold) {
						bias[i]-=gradientClippingThreshold;
					}else if(biasDelta[i]*learningrate<(-1)*gradientClippingThreshold){
						bias[i]+=gradientClippingThreshold;
					}else {
						bias[i]-=biasDelta[i]*learningrate;///times;
					}
				}
			}else {
				for (int i = 0; i < weights.length; i++) {
					for (int j = 0; j < weights[i].length; j++) {
						weights[i][j] -= weightsDelta[i][j]*learningrate;///times;
					}
					bias[i]-=biasDelta[i]*learningrate;///times;
				}
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

		public void setPast(double[] past) {
			System.out.println("parentcall setPast()");
		}
		public void backPropHidden(double[] gamma2,double [][] weightsForward, int i) {
			System.out.println("parentcall backPropHidden()");
		}
		public double[] softmax(double []x) {
			double sum = 0;
			double[] out = new double[x.length];
			for (int i = 0; i < x.length; i++) {
				sum+=x[i];
			}
			for (int i = 0; i < out.length; i++) {
				out[i]=x[i]/sum;
			}
			return out;
		}
	}
	
	public class HiddenLayer extends OutputLayer{
		
		public String type ="Hidden";
		
		public double [][] recurrentWeights = new double[numberOfOutputs][numberOfOutputs];//[Outputs from past ff][Outputs from this ff]
		
		public double [][] recurrentWeightsDelta= new double[numberOfOutputs][numberOfOutputs];
		
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs, double rate,int times) {
			super(numberOfInputs, numberOfOutputs, rate,times);
			outputs = new double[times+1][numberOfOutputs];
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < recurrentWeights[i].length; j++) {
					recurrentWeights[i][j] = (Math.random()-0.5)/1;
					recurrentWeightsDelta[i][j]=0;
				}
			}
		}
		
		public void reset(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weightsDelta[i][j]=0;
				}
				for (int j = 0; j < numberOfOutputs; j++) {
					recurrentWeightsDelta[i][j]=0;
				}
				biasDelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
			for (int i = 0; i < numberOfOutputs; i++) {//Iterate over outputs from this time
				z[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {//Iterate over the inputs
					z[time][i]+= inputs[j]*weights[i][j];
				}
				for (int j = 0; j < numberOfOutputs; j++) {//Iterate over the Outputs from past
					z[time][i]+=outputs[time][j]*recurrentWeights[i][j];
				}
				z[time][i]+=bias[i];
				outputs[time+1][i] = Math.tanh(z[time][i]);
			}
			return outputs[time+1];
		}
		public void backPropHidden(double[]gammaForward, double [][] weightsForward,int time){
//			System.out.println("Gamma from Output at time "+time+ ": "+ Arrays.toString(gammaForward));
			//normal
			for (int i = 0; i < numberOfOutputs; i++) {
				gamma[i] = 0;
				for (int j = 0; j < gammaForward.length; j++) {
					gamma[i] += gammaForward[j]*weightsForward[j][i];//Vektor * Matrix= Vektor +
				}
				gamma[i]*= tanhPrime(z[time][i]);//Vektor * Vektor = Vektor *
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]+= gamma[i]*inputs[time][j];//Vektor * Vektor = Matrix
				}
				biasDelta[i]+= gamma[i];//Vektor = Vektor
			}
			
			//Time
			double [][][]temp= new double[time+1][numberOfOutputs][numberOfOutputs];
			for (int i = 0; i < temp.length; i++) {
				for (int j = 0; j < temp[i].length; j++) {
					for (int j2 = 0; j2 < temp[i][j].length; j2++) {
						temp[i][j][j2]=0;
					}
				}
			}
			double [] tempGamma = new double[numberOfOutputs];
			for (int i = 0; i < time+1; i++) {//Iterate over all times till now
				if(i>0) {
					for (int j2 = 0; j2 < numberOfOutputs; j2++) {
						tempGamma[j2]=0;
						for (int k = 0; k < numberOfOutputs; k++) {
							tempGamma[j2]+=gamma[k]*recurrentWeights[k][j2];
						}
						tempGamma[j2]*=tanhPrime(z[time-i][j2]);
					}
				}
				gamma = tempGamma.clone();
				for (int j = 0; j < numberOfOutputs; j++) {
					for (int j2 = 0; j2 < numberOfOutputs; j2++) {
						temp[i][j][j2]=tempGamma[j]*outputs[time-i][j2];
					}
				}
			}
			for (int i = 0; i < recurrentWeightsDelta.length; i++) {
				for (int j = 0; j < recurrentWeightsDelta[i].length; j++) {
					for (int j2 = 0; j2 < temp.length; j2++) {
						recurrentWeightsDelta[i][j]+=temp[j2][i][j];
					}
				}
			}
		}
		
		public void update(){
			if(gradientClipping) {
				for (int i = 0; i < weights.length; i++) {
					for (int j = 0; j < weights[i].length; j++) {
						if(weightsDelta[i][j]*learningrate>gradientClippingThreshold) {
							weights[i][j] -= gradientClippingThreshold;///times;
						}else if(weightsDelta[i][j]*learningrate<(-1)*gradientClippingThreshold) {
							weights[i][j] += gradientClippingThreshold;///times;
						}else {
							weights[i][j] -= weightsDelta[i][j]*learningrate;
						}
					}
					for (int j = 0; j < recurrentWeights[i].length; j++) {
						if(recurrentWeightsDelta[i][j]*learningrate>gradientClippingThreshold) {
							recurrentWeights[i][j] -= gradientClippingThreshold;///times;
						}else if(recurrentWeightsDelta[i][j]*learningrate<(-1)*gradientClippingThreshold) {
							recurrentWeights[i][j] += gradientClippingThreshold;///times;
						}else {
							recurrentWeights[i][j] -= recurrentWeightsDelta[i][j]*learningrate;
						}
						recurrentWeights[i][j]-=recurrentWeightsDelta[i][j]*learningrate;///times;
					}
					if(biasDelta[i]*learningrate>gradientClippingThreshold) {
						bias[i]-=gradientClippingThreshold;
					}else if(biasDelta[i]*learningrate<(-1)*gradientClippingThreshold) {
						bias[i]+=gradientClippingThreshold;
					}else {
						bias[i]-=biasDelta[i]*learningrate;
					}
				}
			}else {
				for (int i = 0; i < weights.length; i++) {
					for (int j = 0; j < weights[i].length; j++) {
						weights[i][j] -= weightsDelta[i][j]*learningrate;///times;
					}
					for (int j = 0; j < recurrentWeights[i].length; j++) {
						recurrentWeights[i][j]-=recurrentWeightsDelta[i][j]*learningrate;///times;
					}
					bias[i]-=biasDelta[i]*learningrate;///times;
				}
			}
			
		}
		
		public void setPast(double[] past) {
			outputs[0]=past;
		}
		
		public double sigmoid(double x){
			return(1/(1+Math.exp(-x)));
		}
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
		public double[] softmax(double []x) {
			double sum = 0;
			double[] out = new double[x.length];
			for (int i = 0; i < x.length; i++) {
				sum+=x[i];
			}
			for (int i = 0; i < out.length; i++) {
				out[i]=x[i]/sum;
			}
			return out;
		}
	}
	
	public static void main(String[] args) {
		int[] a = {2,100,200,4};
//		double [][]input = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,1,0}};
//		double [][]expected = {{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,0,1}};
		RecurrentNetwork2 rn2 = new RecurrentNetwork2(a, 4,0.0001);
//		System.out.println("input: "+Arrays.toString(input));
//		double[][] test = rn2.feedForward(input);
//		for (int i = 0; i < test.length; i++) {
//			System.out.println("output"+Arrays.toString(test[i]));	
//		}
//		System.out.println();
//		rn2.layers[0].setPast(rn2.layers[0].outputs[rn2.layers[0].outputs.length-1]);
//		System.out.println("set Past to: "+Arrays.toString(rn2.layers[0].outputs[0]));
//		test = rn2.feedForward(input);
//		for (int i = 0; i < test.length; i++) {
//			System.out.println("output"+Arrays.toString(test[i]));	
//		}
//		System.out.println();
//		System.out.println("---------------------------------------------------------------------------------------");
//		for (int i = 0; i < 2000; i++) {
//			rn2.feedForward(input);
//			rn2.backProp(expected);
//		}
//		test = rn2.feedForward(input);
//		for (int j = 0; j < test.length; j++) {
//			System.out.println("output test"+Arrays.toString(test[j]));	
//		}
		double[][] in = rn2.getInputs();
		for (int i = 0; i < 20; i++) {
			System.out.println(Arrays.toString(in[i]));
		}
		int times=25;
		System.out.println(in.length);
		System.out.println(in[1].length);
		double[][][] inputs = new double[in.length/times][times][in[1].length];//for mit length -1
		double [][][] expected2 = new double[in.length/times][times][in[1].length];
		for (int i = 1; i < inputs.length-2; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				inputs[i-1][j]=in[j+j*i];
				expected2[i-1][j]=in[j+j*i+1];
			}
		}
		char [] wb = rn2.getWB();
		System.out.println("wb: "+wb.length);
		double[] start = rn2.getArray('a', wb);
		System.out.println("start: "+Arrays.toString(start));
		System.out.println("start: "+start.length);
		System.out.println("input: "+inputs[0][0].length);
		System.out.println("in: "+in[1].length);
		double[] ff = null;
		int high;
		int []b = {in[1].length,1000,2000,in[1].length};
		rn2 = new RecurrentNetwork2(b, times,0.006);
		System.out.println(rn2.gradientClipping);
		System.out.println(rn2.batchLearning);
		for (int i = 0; i < 5000; i++) {
			for (int j = 0; j < inputs.length/rn2.batchSize; j+=rn2.batchSize) {
				System.out.println("------------------------------------------------------------------------------------------------------------------------------------------------------");
				for (int j2 = 0; j2 <= rn2.batchSize; j2++) {
					System.out.println("----------------------"+j2+"----------------------");
					rn2.feedForward(inputs[j*j2+j2]);
					rn2.backProp(expected2[j*j2+j2]);
				}
//				rn2.setPast();
				for (int h = 0; h < 4; h++) {
					for (int k = 0; k < times; k++) {
						ff = rn2.feedForward(start,k);
						high = rn2.getOutput(ff);
						start = rn2.getOneHot(high, ff.length);
						System.out.print(rn2.trans(wb, start));
					}
					rn2.setPast();
				}
				double error = 0;
				for (int k = 0; k < ff.length; k++) {
					error+=ff[k];
				}
				System.out.println(Arrays.toString(ff));
				System.out.println(error);
			}
		}
	}

}
