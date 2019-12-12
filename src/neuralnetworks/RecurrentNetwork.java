package neuralnetworks;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class RecurrentNetwork {//backprop fehlt, noch nicht fertig, wirklich nicht aber dann doch irgendwie
	// Bei Layer initialisierung noch die Länge von den Weights, die den Past machen angleichen

	String path = "C:/Users/Cybren/Desktop/HW/";
	int[] layer;
	OutputLayer[] layers;
	double learningrate = 0.006;
	int times;
	
	public RecurrentNetwork(int[]layer,int times){
		this.layer = layer;
		this.times=times;
		layers = new OutputLayer[layer.length-1];
		for (int i = 0; i < layers.length-1; i++) {
			layers[i]= new HiddenLayer(layer[i],layer[i+1],learningrate,times);
		}
		layers[layers.length-1]= new OutputLayer(layer[layer.length-2],layer[layer.length-1],learningrate,times);
	}
	
	public double[] feedForward(double[]inputs,int time){
		layers[0].feedForward(inputs,time);
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i-1].outputs[time],time);
		}
		return layers[layers.length-1].outputs[time];
	}
	
	public void backProp(double []expected,int time){
		layers[layers.length-1].backProp(expected,time);
		for (int i = layers.length-2; i > -1; i--) {
				layers[i].backProp(layers[i+1].gamma, layers[i+1].weights,time);
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
	
	public double[]zero(double[]in){
		for (int i = 0; i < in.length; i++) {
			in[i]=0;
		}
		return in;
	}
	public double[][]zero(double[][]in){
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[i].length; j++) {
				in[i][j]=0;
			}
		}
		return in;
	}
	public double [][][]zero(double[][][]in){
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[i].length; j++) {
				for (int j2 = 0; j2 < in[i][j].length; j2++) {
					in[i][j][j2]=0;
				}
			}
		}
		return in;
	}
	
	public class HiddenLayer extends OutputLayer{
		double [][] weightsH;//weights für recursion;
		double [][] weightsHDelta;
		double [] biasH;
		double [] biasHDelta;
		double [][] past;
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs,double rate,int times){
			super(numberOfInputs,numberOfOutputs,rate,times);
			
			weightsH = new double [numberOfOutputs][numberOfOutputs];
			weightsHDelta = new double [numberOfOutputs][numberOfOutputs];
			
			biasH = new double[numberOfOutputs];
			biasHDelta = new double[numberOfOutputs];
			past = new double[times+1][numberOfOutputs];
			iniH();
		}
		public void iniH(){
			for (int i = 0; i < weightsH.length; i++) {
				for (int j = 0; j < weightsH[i].length; j++) {
					weightsH[i][j]=Math.random()-0.5;
					weightsHDelta[i][j]=0;
				}
				biasH[i]= Math.random()-0.5;
				biasHDelta[i]=0;
				past[0][i]=1;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
//			System.out.println("past at time "+time+" :"+Arrays.toString(past[time]));
			this.inputs[time] = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				outputs[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {
					outputs[time][i]+= inputs[j]*weights[i][j];
					outputs[time][i]+= past[time][i]*weightsH[i][j];
				}
				outputs[time][i]+=bias[i];
				outputs[time][i] = Math.tanh(outputs[time][i]);
			}
			past[time+1]=outputs[time];
			return outputs[time];
		}
		public void backProp(double[]gammaForward, double [][] weightsForward,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
				gamma[i] = 0;
				for (int j = 0; j < gammaForward.length; j++) {
					gamma[i] += gammaForward[j]/**weightsForward[j][i]*/;
				}
				gamma[i]*= tanhPrime(outputs[time][i]);
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]+= gamma[i]*inputs[time][j];
					weightsHDelta[i][j]+= gamma[i]*past[time][j];
				}
				biasDelta[i] += gamma[i];
				biasHDelta[i]+= gamma[i];
			}
			for (int h = 0; h < time; h++) {
				for (int i = 0; i < numberOfOutputs; i++) {
					for (int j = 0; j < numberOfInputs; j++) {
						weightsDelta[i][j]+= gamma[i]*inputs[h][j]*Math.pow(weightsH[i][j], time-h);
						weightsHDelta[i][j]+= gamma[i]*past[h][j]*Math.pow(weightsH[i][j], time-h);
					}
					biasDelta[i] += gamma[i];// hier eventeuell das + rausnehmen bzw einfach beide bias, so dass bias nur noch oben einmal vorkommt;
					biasHDelta[i]+= gamma[i];
				}
			}
		}
		
		public void update(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] -= weightsDelta[i][j]*learningrate;
					weightsH[i][j] -=weightsHDelta[i][j]*learningrate;
				}
				bias[i]-=biasDelta[i];
				biasH[i]-=biasHDelta[i];
			}
		}
		
	}
public class OutputLayer{
		
		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of neurons in curent layer
		double learningrate;
		double [][] outputs;
		double [][] inputs;
		double [][] weights;
		double [][] weightsDelta;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		
		public OutputLayer(int numberOfInputs, int numberOfOutputs,double rate,int times){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			
			outputs = new double[times][numberOfOutputs];
			inputs = new double [times][numberOfInputs];
			
			weights = new double[numberOfOutputs][numberOfInputs];
			weightsDelta = new double[numberOfOutputs][numberOfInputs];
			
			gamma = new double[numberOfOutputs];
			error = new double[numberOfOutputs];
			
			bias = new double[numberOfOutputs];
			biasDelta = new double[numberOfOutputs];
			
			ini();
		}
		
		public void backProp(double[] gamma2, double[][] weights2, int time) {
			System.out.println("thats the prototype!");
		}

		public void ini(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Math.random()-0.5;
					weightsDelta[i][j] =0;
				}
				bias[i]= Math.random()-0.5;
				biasDelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				outputs[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {
					outputs[time][i]+= inputs[j]*weights[i][j];
				}
				outputs[time][i]+=bias[i];
			}
			outputs[time] = softmax(outputs[time]);
			return outputs[time];
		}
		
		public void backProp(double[] expected,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
				error[i] = outputs[time][i]-expected[i];
				gamma[i] = error[i];
			}
			for (int i = 0; i < weightsDelta.length; i++) {
				for (int j = 0; j < weightsDelta[i].length; j++) {
					weightsDelta[i][j]+= error[i]*inputs[time][j];
				}
				biasDelta[i]+=error[i];
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
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}
		public double[] softmax (double[] x) {
			double [] back = new double[x.length];
			double sum =0;
			for (int i = 0; i < x.length; i++) {
				sum+=Math.exp(x[i]);
			}
			for (int i = 0; i < back.length; i++) {
				back[i]=Math.exp(x[i])/sum;
			}
			return back;
		}
	}
	public static void main(String[]args){
		//h:0 e:1 l:2 o:3;
		int []layer = {4,100,200,100,4};
		RecurrentNetwork r = new RecurrentNetwork(layer, 4);
		double [][]input = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,1,0}};
		double [][]expected = {{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,0,1}};
		double []out={};
		for (int i = 0; i < 50; i++) {
			for (int j = 0; j < input.length; j++) {
				out=r.feedForward(input[j], j);
				System.out.println("output at time "+j+" in epoch "+i+": "+Arrays.toString(out));
			}
			
			for (int j = input.length-1; j > -1 ; j--) {
				r.backProp(expected[j], j);
			}
			for (int j = 0; j < r.layers.length; j++) {
				r.layers[j].update();
			}
			System.out.println();
		}
		for (int i = 0; i < input.length; i++) {
			out=r.feedForward(input[i], i);
			System.out.println();
			System.out.println("output at time "+i+": "+Arrays.toString(out));
		}
		System.out.println(out[1]+out[2]+out[3]+out[0]);
	}
}
