package neuralnetworks;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class LSTMNetwork_old {//Cellstate safen und an next Batch weiter geben

	String path = "C:/Users/Cybren/Desktop/HW/";
	int[] layer;//structure {{5},{100},{2}}
	OutputLayer[] layers;//layers 2 hidden 1 output
	double learningrate = 0.1;
	int times;// how many times the Network does remember
	
	public LSTMNetwork_old(int[]layer,int times,double learningrate){
		this.layer = layer;
		this.times=times;
		this.learningrate = learningrate;
		layers = new OutputLayer[layer.length-1];
		for (int i = 0; i < layers.length-1; i++) {
			layers[i]= new HiddenLayer(layer[i],layer[i+1],learningrate,times);
		}
		layers[layers.length-1]= new OutputLayer(layer[layer.length-2],layer[layer.length-1],learningrate,times);
	}
	
	public double[][] feedForward(double[][]inputs){
		for (int i = 0; i < times; i++) {
			layers[0].feedForward(inputs[i],i);
			for (int j = 1; j < layers.length; j++) {
				layers[j].feedForward(layers[j-1].outputs[i],i);
			}
		}
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double [][]expected){
		for (int i = 0; i < times; i++) {
			layers[layers.length-1].backProp(expected[i],i);//outputlayer
			//hiddenlayer under outputlayer
			layers[layers.length-2].backProp(layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,i);
			layers[layers.length-2].backPropTime(layers[layers.length-1].gamma,layers[layers.length-1].gamma,layers[layers.length-1].gamma,layers[layers.length-1].gamma,i);
			//the hiddenlayers below the last hiddenlayer
			for (int j = layers.length-3; j > -1; j--) {
				layers[j].backProp(layers[j+1].gammaF,layers[j+1].weightsF,layers[j+1].gammaA,layers[j+1].weightsA,layers[j+1].gammaI,layers[j+1].weightsI,layers[j+1].gammaO,layers[j+1].weightsO,i);
				layers[j].backPropTime(layers[j].gammaFR,layers[j].gammaAR,layers[j].gammaIR,layers[j].gammaOR,i);
			}
		}
		for (int i = 0; i < layers.length; i++) {
			layers[i].update();
		}
	}
	
	public void saveANN(){//doesnt work yet
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
		Path p = FileSystems.getDefault().getPath(path,"Handwrite0055.lstm");
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
		double [][][] weightsFDelta;
		double [][][] weightsIDelta;
		double [][][] weightsADelta;
		double [][][] weightsODelta;
		
		double [][] rF;//recurrents für forget;
		double [][] rFDelta;
		double [][] rI;//recurrents für input;
		double [][] rIDelta;
		double [][] rA;//recurrents für Add;
		double [][] rADelta;
		double [][] rO;//recurrents für Output;
		double [][] rODelta;
		
		double [] biasF;//bias für forget;
		double [][] biasFDelta;
		double [] biasI;//bias für input;
		double [][] biasIDelta;
		double [] biasA;//bias für Add;
		double [][] biasADelta;
		double [] biasO;//bias für Output;
		double [][] biasODelta;
		
		double [][] forget;
		double [][] ingate;
		double [][] add;
		double [][] outgate;
		
		double [][] past;
		double [][] cellstate;
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs,double rate,int times){
			super(numberOfInputs,numberOfOutputs,rate,times);
			type = "hidden";
			weightsF = new double [numberOfOutputs][numberOfInputs];
			weightsFDelta = new double [times][numberOfOutputs][numberOfInputs];
			weightsI = new double [numberOfOutputs][numberOfInputs];
			weightsIDelta = new double[times] [numberOfOutputs][numberOfInputs];
			weightsA = new double [numberOfOutputs][numberOfInputs];
			weightsADelta = new double[times] [numberOfOutputs][numberOfInputs];
			weightsO = new double [numberOfOutputs][numberOfInputs];
			weightsODelta = new double[times] [numberOfOutputs][numberOfInputs];
			
			rF = new double [numberOfOutputs][numberOfOutputs];
			rFDelta = new double [numberOfOutputs][numberOfOutputs];
			rI = new double [numberOfOutputs][numberOfOutputs];
			rIDelta = new double [numberOfOutputs][numberOfOutputs];
			rA = new double [numberOfOutputs][numberOfOutputs];
			rADelta = new double [numberOfOutputs][numberOfOutputs];
			rO = new double [numberOfOutputs][numberOfOutputs];
			rODelta = new double [numberOfOutputs][numberOfOutputs];
			
			biasF = new double[numberOfOutputs];
			biasFDelta = new double[times][numberOfOutputs];
			biasI = new double[numberOfOutputs];
			biasIDelta = new double[times][numberOfOutputs];
			biasA = new double[numberOfOutputs];
			biasADelta = new double[times][numberOfOutputs];
			biasO = new double[numberOfOutputs];
			biasODelta = new double[times][numberOfOutputs];
			
			gammaF=new double[numberOfOutputs];
			gammaI=new double[numberOfOutputs];
			gammaA=new double[numberOfOutputs];
			gammaO=new double[numberOfOutputs];
			
			gammaFR=new double[numberOfOutputs];
			gammaIR=new double[numberOfOutputs];
			gammaAR=new double[numberOfOutputs];
			gammaOR=new double[numberOfOutputs];
			
			forget = new double[times][numberOfOutputs];
			ingate = new double[times][numberOfOutputs];
			add = new double[times][numberOfOutputs];
			outgate = new double[times][numberOfOutputs];
			
			past = new double[times+1][numberOfOutputs];
			cellstate = new double[times+1][numberOfOutputs];//warum +1, weil 0 der intial ist und danach immer time+1 für speicher
			iniH();
		}
		
		public void iniH(){
			for (int i = 0; i < weightsF.length; i++) {
				for (int j = 0; j < weightsF[i].length; j++) {
					weightsF[i][j]=Math.random()-0.5;
					weightsI[i][j]=Math.random()-0.5;
					weightsA[i][j]=Math.random()-0.5;
					weightsO[i][j]=Math.random()-0.5;
					
					rF[i][j]=Math.random()-0.5;
					rI[i][j]=Math.random()-0.5;
					rA[i][j]=Math.random()-0.5;
					rO[i][j]=Math.random()-0.5;
				}
				biasF[i]= Math.random()-0.5;
				biasI[i]= Math.random()-0.5;
				biasA[i]= Math.random()-0.5;
				biasO[i]= Math.random()-0.5;
				
				past[0][i]=1;
				cellstate[0][i]=1;
			}
			for (int h = 0; h < times; h++) {
				for (int i = 0; i < weightsF.length; i++) {
					for (int j = 0; j < weightsF[i].length; j++) {
						weightsFDelta[h][i][j]=0;
						weightsIDelta[h][i][j]=0;
						weightsADelta[h][i][j]=0;
						weightsODelta[h][i][j]=0;

						rFDelta[i][j]=0;
						rIDelta[i][j]=0;
						rADelta[i][j]=0;
						rODelta[i][j]=0;
					}
					biasFDelta[h][i]=0;
					biasIDelta[h][i]=0;
					biasADelta[h][i]=0;
					biasODelta[h][i]=0;
				}
			}
		}
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs,double rate,int times, double[]cellstate, double[] past){//to transfer the last cellstate and output to the present
			super(numberOfInputs,numberOfOutputs,rate,times);
			
			weightsF = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsFDelta = new double [times][numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsI = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsIDelta = new double [times][numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsA = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsADelta = new double [times][numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsO = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsODelta = new double [times][numberOfOutputs][numberOfInputs+numberOfOutputs];
			
			rF = new double [numberOfOutputs][numberOfOutputs];
			rFDelta = new double [numberOfOutputs][numberOfOutputs];
			rI = new double [numberOfOutputs][numberOfOutputs];
			rIDelta = new double [numberOfOutputs][numberOfOutputs];
			rA = new double [numberOfOutputs][numberOfOutputs];
			rADelta = new double [numberOfOutputs][numberOfOutputs];
			rO = new double [numberOfOutputs][numberOfOutputs];
			rODelta = new double [numberOfOutputs][numberOfOutputs];
			
			biasF = new double[numberOfOutputs];
			biasFDelta = new double[times][numberOfOutputs];
			biasI = new double[numberOfOutputs];
			biasIDelta = new double[times][numberOfOutputs];
			biasA = new double[numberOfOutputs];
			biasADelta = new double[times][numberOfOutputs];
			biasO = new double[numberOfOutputs];
			biasODelta = new double[times][numberOfOutputs];
			
			gammaF=new double[numberOfOutputs];
			gammaI=new double[numberOfOutputs];
			gammaA=new double[numberOfOutputs];
			gammaO=new double[numberOfOutputs];
			
			gammaFR=new double[numberOfOutputs];
			gammaIR=new double[numberOfOutputs];
			gammaAR=new double[numberOfOutputs];
			gammaOR=new double[numberOfOutputs];
			
			forget = new double[times][numberOfOutputs];
			ingate = new double[times][numberOfOutputs];
			add = new double[times][numberOfOutputs];
			outgate = new double[times][numberOfOutputs];
			
			this.past = new double[times+1][numberOfOutputs];
			this.cellstate = new double[times+1][numberOfOutputs];
			iniH(cellstate,past);
		}
		
		public void iniH(double[]cellstate,double[]past){//to set the Cellstate
			for (int i = 0; i < weightsF.length; i++) {
				for (int j = 0; j < weightsF[i].length; j++) {
					weightsF[i][j]=Math.random()-0.5;
					weightsI[i][j]=Math.random()-0.5;
					weightsA[i][j]=Math.random()-0.5;
					weightsO[i][j]=Math.random()-0.5;
					
					rF[i][j]=Math.random()-0.5;
					rI[i][j]=Math.random()-0.5;
					rA[i][j]=Math.random()-0.5;
					rO[i][j]=Math.random()-0.5;
				}
				biasF[i]= Math.random()-0.5;
				biasI[i]= Math.random()-0.5;
				biasA[i]= Math.random()-0.5;
				biasO[i]= Math.random()-0.5;
			}
			for (int h = 0; h < times; h++) {
				for (int i = 0; i < weightsF.length; i++) {
					for (int j = 0; j < weightsF[i].length; j++) {
						weightsFDelta[h][i][j]=0;
						weightsIDelta[h][i][j]=0;
						weightsADelta[h][i][j]=0;
						weightsODelta[h][i][j]=0;

						rFDelta[i][j]=0;
						rIDelta[i][j]=0;
						rADelta[i][j]=0;
						rODelta[i][j]=0;
					}
					biasFDelta[h][i]=0;
					biasIDelta[h][i]=0;
					biasADelta[h][i]=0;
					biasODelta[h][i]=0;
				}
			}
			this.past[0]=past;
			this.cellstate[0]=cellstate;
		}
		
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time]=inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {//start feedforward
				forget[time][i]=0;
				ingate[time][i]=0;
				add[time][i]=0;
				outgate[time][i]=0;
				for (int j = 0; j < inputs.length; j++) {
					forget [time][i]+=inputs[j]*weightsF[i][j]+past[time][j]*rF[i][j];
					ingate [time][i]+=inputs[j]*weightsI[i][j]+past[time][j]*rI[i][j];
					add    [time][i]+=inputs[j]*weightsA[i][j]+past[time][j]*rA[i][j];
					outgate[time][i]+=inputs[j]*weightsO[i][j]+past[time][j]*rO[i][j];
				}
				forget[time][i]+=biasF[i];
				ingate[time][i]+=biasI[i];
				add[time][i]+=biasA[i];
				outgate[time][i]+=biasO[i];
				
				cellstate[time+1][i]=cellstate[time][i]*sigmoid(forget[time][i])+(sigmoid(ingate[time][i])*Math.tanh(add[time][i]));
				
				outputs[time][i]=sigmoid(outgate[time][i])*Math.tanh(cellstate[time+1][i]);
				past[time+1][i]=outputs[time][i];
			}
			return outputs[time];
		}
		public void backProp(double[]gammaFForward,double[][]weightsFForward,double[]gammaAForward,double[][]weightsAForward,double[]gammaIForward,double[][]weightsIForward,double[]gammaOForward,double[][]weightsOForward,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
				gammaF[i] = 0;
				gammaA[i] = 0;
				gammaI[i] = 0;
				gammaO[i] = 0;
				for (int j = 0; j < weightsFForward.length; j++) {
					gammaF[i] += gammaFForward[j]*weightsFForward[j][i];
					gammaA[i] += gammaAForward[j]*weightsAForward[j][i];
					gammaI[i] += gammaIForward[j]*weightsIForward[j][i];
					gammaO[i] += gammaOForward[j]*weightsOForward[j][i];
				}
				
				gammaF[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoidPrime(forget[time][i])*cellstate[time][i];
				gammaA[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoid(ingate[time][i])*tanhPrime(add[time][i]);
				gammaI[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*Math.tanh(add[time][i])*sigmoidPrime(ingate[time][i]);
				gammaO[i]*= Math.tanh(cellstate[time+1][i])*sigmoidPrime(outgate[time][i]);
			}
			
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsFDelta[time][i][j]= gammaF[i]*inputs[time][j];
					weightsADelta[time][i][j]= gammaA[i]*inputs[time][j];
					weightsIDelta[time][i][j]= gammaI[i]*inputs[time][j];
					weightsODelta[time][i][j]= gammaO[i]*inputs[time][j];
				}
				biasFDelta[time][i]= gammaF[i];
				biasADelta[time][i]= gammaA[i];
				biasIDelta[time][i]= gammaI[i];
				biasODelta[time][i]= gammaO[i];
			}
		}
		
		public void backPropTime(double[] gammaFRForward,double[] gammaARForward,double[] gammaIRForward,double[] gammaORForward,int time) {
			for (int i = 0; i < numberOfOutputs; i++) {
				gammaFR[i]=0;
				gammaAR[i]=0;
				gammaIR[i]=0;
				gammaOR[i]=0;
				for (int j = 0; j < gammaFRForward.length; j++) {
					gammaFR[i]+=gammaFRForward[j]*rF[j][i];
					gammaAR[i]+=gammaARForward[j]*rF[j][i];
					gammaIR[j]+=gammaIRForward[j]*rF[j][i];
					gammaOR[j]+=gammaORForward[j]*rF[j][i];
				}
				gammaFR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoidPrime(forget[time][i])*cellstate[time][i];
				gammaAR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoid(ingate[time][i])*tanhPrime(add[time][i]);
				gammaIR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*Math.tanh(add[time][i])*sigmoidPrime(ingate[time][i]);
				gammaOR[i]*= Math.tanh(cellstate[time+1][i])*sigmoidPrime(outgate[time][i]);
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfOutputs; j++) {
					rFDelta[i][j]= gammaFR[i]*past[time][j];
					rADelta[i][j]= gammaAR[i]*past[time][j];
					rIDelta[i][j]= gammaIR[i]*past[time][j];
					rODelta[i][j]= gammaOR[i]*past[time][j];
				}
			}
		}
		
		public void update(){
			//update weights
			double [][]updateF= new double[numberOfOutputs][numberOfInputs];
			double [][]updateA= new double[numberOfOutputs][numberOfInputs];
			double [][]updateI= new double[numberOfOutputs][numberOfInputs];
			double [][]updateO= new double[numberOfOutputs][numberOfInputs];
			updateF=zero(updateF);
			updateA=zero(updateA);
			updateI=zero(updateI);
			updateO=zero(updateO);
			double []updateFB= new double[numberOfOutputs];
			double []updateAB= new double[numberOfOutputs];
			double []updateIB= new double[numberOfOutputs];
			double []updateOB= new double[numberOfOutputs];
			updateFB=zero(updateFB);
			updateAB=zero(updateAB);
			updateIB=zero(updateIB);
			updateOB=zero(updateOB);
			for (int i = 0; i < weightsF.length; i++) {
				for (int j = 0; j < weightsF[i].length; j++) {
					for (int h = 0; h < times; h++) {
						updateF[i][j]+=weightsFDelta[h][i][j];
						updateA[i][j]+=weightsADelta[h][i][j];
						updateI[i][j]+=weightsIDelta[h][i][j];
						updateO[i][j]+=weightsODelta[h][i][j];
					}
					updateF[i][j]/=times;
					updateA[i][j]/=times;
					updateI[i][j]/=times;
					updateO[i][j]/=times;
				}	
			}
			for (int j = 0; j < biasF.length; j++) {
				for (int h = 0; h < times; h++) {
					updateFB[j]+=biasFDelta[h][j];
					updateAB[j]+=biasADelta[h][j];
					updateIB[j]+=biasIDelta[h][j];
					updateOB[j]+=biasODelta[h][j];
				}
				updateFB[j]/=times;
				updateAB[j]/=times;
				updateIB[j]/=times;
				updateOB[j]/=times;
			}
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weightsF[i][j] -= updateF[i][j]*learningrate;
					weightsA[i][j] -= updateA[i][j]*learningrate;
					weightsI[i][j] -= updateI[i][j]*learningrate;
					weightsO[i][j] -= updateO[i][j]*learningrate;
				}
				biasF[i]-=updateFB[i];
				biasA[i]-=updateAB[i];
				biasI[i]-=updateIB[i];
				biasO[i]-=updateOB[i];
			}
			//update recursive
			for (int i = 0; i < rF.length; i++) {
				for (int j = 0; j < rF[i].length; j++) {
					rF[i][j] -= rFDelta[i][j]*learningrate;
					rA[i][j] -= rADelta[i][j]*learningrate;
					rI[i][j] -= rIDelta[i][j]*learningrate;
					rO[i][j] -= rODelta[i][j]*learningrate;
				}
			}
			
		}
	}
public class OutputLayer{
		String type = "output";
		public double[] gammaFR;
		public double[] gammaAR;
		public double[] gammaIR;
		public double[] gammaOR;
		public double[][] weightsO;
		public double[] gammaO;
		public double[][] weightsI;
		public double[] gammaI;
		public double[][] weightsA;
		public double[] gammaA;
		public double[][] weightsF;
		public double[] gammaF;
		
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
		
		public void backPropTime(double[] gammaFR2, double[] gammaAR2, double[] gammaIR2, double[] gammaOR2, int i) {
			System.out.println("This is the prototype"+i);	
		}

		public void backProp(double[] gamma2, double[][] weights2, double[] gamma3, double[][] weights3,double[] gamma4, double[][] weights4, double[] gamma5, double[][] weights5, int i) {
			System.out.println("This is the prototype"+i);
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
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
		
		public double tanhPrime(double x){
			return 1-(x*x);
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
		int []layer = {4,20,30,4};
		double [][]input = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,1,0},{0,0,1,0}};
		double [][]expected = {{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,0,1},{0,0,0,1},{0,0,0,1}};
		LSTMNetwork_old r = new LSTMNetwork_old(layer, 6,0.1);
		double [][]out= {{}};
		for (int i = 0; i < 20; i++) {
				out=r.feedForward(input);
				r.backProp(expected);
			for (int j = 0; j < r.layers.length; j++) {
				r.layers[j].update();
			}
			for (int h = 0; h < out.length; h++) {
				System.out.println(Arrays.toString(out[h]));
			}
//			System.out.println();
		}
			out=r.feedForward(input);
			for (int i = 0; i < out.length; i++) {
				System.out.println(Arrays.toString(out[i]));
			}
	}
}
