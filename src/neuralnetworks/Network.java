package neuralnetworks;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Network {
	
	double biases[][];//[Number of the Layer -1] because first Layer has no bias [Number of the Neutron]
	int numLayers;
	double weights [][][];//weights[numLayers-1][numNeurons in that Layer|to] [NumNeruons in previous layer|from]
	int layers[];
	int numNeurons;
	double learningrate = 6; 
	String path = "C:/Users/Cybren/Desktop/HW/";
	
	public Network(int layers[]){
		numLayers=layers.length;
		biases = new double[layers.length-1][];
		for (int i = 0; i < layers.length; i++) {
			numNeurons+=layers[i];
		}
		for (int i = 0; i < layers.length-1; i++) {
			biases [i] = new double [layers[i+1]];
			for (int j = 0; j < biases[i].length; j++) {
				//biases[i][j]=Math.random();
				biases[i][j]=0;
			}
		}
		weights= new double [numLayers-1][][];
		for (int i = 0; i < weights.length; i++) {
			weights[i]=new double[layers[i+1]][];
			
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j]= new double[layers[i]];
				
				for (int j2 = 0; j2 < weights[i][j].length; j2++) {// ...dom()/number cause many inputs sum up to a high sum (xD) that destroys the algortihm
					weights[i][j][j2]=Math.random()/100-0.5;	
				}
				//System.out.println("weights "+i+";"+j+": "+Arrays.toString(weights[i][j]));
			}
		}
		this.layers=layers;
	}
	public Network(int[]layers,double[][]biases,double[][][]weights){
		numLayers=layers.length;
		for (int i = 0; i < layers.length; i++) {
			numNeurons+=layers[i];
		}
		this.biases = biases;
		this.weights=weights;
		this.layers=layers;
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
	
	/*layermin1: the layer the neuron is layer-1 because the first layer is the inputlayer so layer one is array[0];
	 * neuron: number of the neuron we want the output from e.g. second neuron in second layer would be layermin1=1 neutron=2
	 * a.length = numNeurons in Layer -1
	 * a: the output from the neutron from the past layer*/
	
	public double output(double a[],int layermin1,int neuron){
		double out=0;
		
		for (int i = 0; i < a.length; i++) {//sum up all w*x; all inputs times his weight
				out+=weights[layermin1][neuron][i]*a[i];
		}
		out+=biases[layermin1][neuron];
		
		out= Math.tanh(out);
		return out;
	}
	
	public double z(double a[],int layermin1,int neuron){//output without sigmoid
		double out=0;
		
		for (int i = 0; i < a.length; i++) {//sum up all w*x; all inputs times his weight
				out+=weights[layermin1][neuron][i]*a[i];
		}
		out+=biases[layermin1][neuron];
		
		return out;
	}
	
	public double[][][]forward(double input []){
		double[][][]a = new double[2][layers.length-1][];//[Layer][neuron in that layer]
		
		for (int i = 0; i < a[0].length; i++) {
			a[0][i] = new double[layers[i+1]];
			a[1][i] = new double[layers[i+1]];
			for (int j = 0; j < a[0][i].length; j++) {//Matrix weigths[i]* Matrix input
				a[0][i][j]=output(input,i,j);
				a[1][i][j]=z(input,i,j);
			}
			input = a[0][i];
//			System.out.println("Forward(a): "+Arrays.toString(a[0][i]));
//			System.out.println("Forward(z): "+Arrays.toString(a[1][i]));
//			System.out.println();
		}
//		System.out.println();
		return a;
	}
	
	public void train(){//umschreiben, das hier die epochs etc erstellt werden und dann x mal train durchgeführt wird
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
		double[][][] newTP= new double[tPictures.length/n][n][];
		for (int i = 0; i < newTP.length; i++) {
			for (int j = 0; j < newTP[i].length; j++) {
				newTP[i][j]=tPictures[newTP.length*i+j];
			}
		}
		byte[][] newTN = new byte[testNumbers.length/n][n];
		for (int i = 0; i < newTN.length; i++) {
			for (int j = 0; j < newTN[i].length; j++) {
				newTN[i][j]=testNumbers[n*i+j];
			}
		}
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				System.out.print(Math.round(newTP[0][e][i*28+j]*1000.0)/1000.0+" ");
			}
			System.out.println();
		}
		System.out.println(newTN[0][e]);
		//System.out.println(Arrays.toString(newPictures[0]));
		//------------------------------train--------------------------------------
		while(error>0.05){
		byte o = 0;
			for (int i = 0; i < newPictures.length/*/epochsize*/; i++) {//train the network
				double [][][][] temp = new double[epochsize][][][];
				for (int j = 0; j < temp.length; j++) {// get the set of deltas for the epoch
					temp[j]= train(newPictures[j],newNumbers[j]);
				}
				
				double [][][] meanDelta = new double[weights.length][][];
				for (int j2 = 0; j2 < meanDelta.length; j2++) {
					meanDelta[j2] = new double [weights[j2].length][];
					for (int k = 0; k < meanDelta[j2].length; k++) {
						meanDelta[j2] [k]= new double [weights[j2][k].length];
						for (int k2 = 0; k2 < meanDelta[j2][k].length; k2++) {
							
						meanDelta[j2][k][k2]=0;
							for (int j = 0; j < temp.length; j++) {
//							System.out.println(temp[0][0][0][0]);
//							System.out.println("j:"+j+" j2:"+j2+" k:"+k+" k2:"+k2+";");
								meanDelta[j2][k][k2]+=temp[j][j2][k][k2];
							}
						meanDelta[j2][k][k2]/=epochsize;						
						}
					}
				}
				adapt(meanDelta);
				System.out.println("Last Weight"+ weights[weights.length-1][weights[weights.length-1].length-1][weights[weights.length-1][weights[weights.length-1].length-1].length-1]);
				int right=0;
				for (int h = 0; h < newTP[o].length; h++) {
					double high = -1.0;
					int at = -1;
					double[][][] out = forward(newTP[o][h]);
					for (int j = 0; j < out[0][out[0].length-1].length; j++) {
						if(out[0][out[0].length-1][j]>high){
							high = out[0][out[0].length-1][j];
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
				error = (n2-right)/n2;
				System.out.println("error: "+error);
			}
		}	
	}
	
	public double[][][]train(double[]trainin,byte trainnum){//eventuell einfacher durchlauf z.B. nur eine Epoche
		double[][]fa=new double[numLayers][];
		double[][]fz;
		double [][][]deltaw={};
		
			byte[]train = new byte[layers[layers.length-1]];//create Array for the desired result eg [0,0,0,0,1,0,0,0,0,0] for 4
			for (int j = 0; j < train.length; j++) {
				if(j==trainnum){
					train[j]=1;
				}else{
					train[j]=0;
				}
			}
			double[][][]temp;
			fa[0]= new double[trainin.length];
			for (int i = 0; i < trainin.length; i++) {
				fa[0][i]=Math.tanh(trainin[i]);
			}
			temp = forward(trainin/*fa[0]*/);//calculate the a's and z's for every neuron in the Network
			for (int i = 1; i < fa.length; i++) {
				fa[i]=temp[0][i-1];
			}
			fz = temp[1];
			double[]cost = cost(fa[fa.length-1],train);//calculate the costarry a-y
//			System.out.println("cost: "+Arrays.toString(cost));
			deltaw = backward(fa,fz,cost);
		return deltaw;
		
	}
	
	public void test(double[][]trainin,byte[] trainnum){
		for (int i = 0; i < 100000; i++) {
			for (int h = 0; h < trainin.length; h++) {
				double[][][] newW = train(trainin[h],trainnum[h]);
				adapt(newW);
			}
		}
	}
	
	public double [][][]backward(double[][]a,double[][]z,double[]y){
		//initialize sums
		double [][]sums = createSums(a,z,y);
		for (int i = 0; i < sums.length; i++) {
//			System.out.println("sums "+i+": "+Arrays.toString(sums[i]));
		}
		//initialize weightsnew for the last layer
		double[][][]weightsnew= new double [numLayers-1][][];//[Layer][numNeurons in that layer][numneuron in prev layer]
		weightsnew[weightsnew.length-1] = initialize(a, z, y);
		//
		for (int i = numLayers-3; i > -1; i--) {//from last to first layer begin at second last cause last is already computed above i=thirdlast layer
			weightsnew[i]=new double[layers[i+1]][];
			
			for (int j = 0; j < weightsnew[i].length; j++) {//neurons in that layer begin at secondlastlayer j: 0,1,2
				weightsnew[i][j]= new double[layers[i]];
				
				for (int j2 = 0; j2 < weightsnew[i][j].length; j2++) {//j2: 0,1//neuron in prev layer
					weightsnew[i][j][j2]=(-learningrate)*a[i][j2]*sigmoidPrime(z[i][j])*sums[i][j];
//					System.out.println("weigthtsnew "+j2+": "+a[i][j2]+"*"+sigmoidPrime(z[i][j])+"("+z[i][j]+")*"+sums[i][j]);
				}
//				System.out.println("wn"+i+","+j+": "+Arrays.toString(weightsnew[i][j]));
//				System.out.println();
			}
		}
		return weightsnew;
	}

	public double[][] createSums(double[][]a,double[][]z,double[]e){//e: mse/cost and after first iteration the last sum; checked for 3 layers
		double[][]sums = new double[numLayers-2][];					//[Layer][Number of neuron]
		for (int h = sums.length-1; h > -1; h--) {
			sums[h] = new double[layers[h+1]];
			for (int i = 0; i < sums[h].length; i++) {//begin at second last layer; neurons in that layer
				sums[h][i]=0;
				for (int j = 0; j < layers[h+2]; j++) {//begin at last layer; neurons in next layer
					sums[h][i]+=weights[h+1][j][i]*sigmoidPrime(z[h+1][j])*e[j];
				//System.out.println("sums "+i+": "+weights[h+1][j][i]+"*"+sigmoidPrime(z[h+1][j])+"("+z[h+1][j]+")*"+e[j]);
				}
			}
			e=sums[h];
		}
		return sums;
	}
	
	public double[][] initialize(double[][]a,double[][]z,double[]y){//checked; y:cost; first iteration for the last layer of weights
		double [][]back= new double[layers[layers.length-1]][layers[layers.length-2]];//[num neurons in last layer][num neurons in secondlast layer] 
		for (int i = 0; i < back.length; i++) {//go over all neurons in last layer i:lastlayer 
			for (int j = 0; j < back[i].length; j++) {//to give all connections from the last layer to the secondlast j: secondlast 
				back[i][j]=(-learningrate)*a[a.length-2][j]*sigmoidPrime(z[z.length-1][i])*y[i];//a from secondlast layer*sigmoidprime from z from lastlayer*2*cost
				//System.out.println("back "+i+";"+j+": "+a[a.length-2][j]+"*"+sigmoidPrime(z[z.length-1][i])+"("+z[z.length-1][i]+")*"+y[i]+";");
			}
//			System.out.println("init"+i+": "+Arrays.toString(back[i]));
		}
//		System.out.println();
		return back;
	}
	
	public double mse (double[] cost){
		double mse = 0;
		
		for (int i = 0; i < cost.length; i++) {
			mse+=Math.pow(cost[i], 2.0);
		}
		
		return mse;
	}
	
	public double[] cost(double[] out, byte train[]){
		double[] cost = new double [out.length];
		
		for (int i = 0; i < cost.length; i++) {
			cost[i]=out[i]-train[i];
		}
		
		return cost;
	}
	
	public void adapt(double[][][]wnew){
		
		for (int i = 0; i < wnew.length; i++) {
			for (int j = 0; j < wnew[i].length; j++) {
//				System.out.println("wnew"+Arrays.toString(wnew[i][j]));
				for (int j2 = 0; j2 < wnew[i][j].length; j2++) {
					if(i==wnew.length-1){
//						System.out.println("weights"+weights[i][j][j2]+"+wnew"+wnew[i][j][j2]+"");
					}
					weights[i][j][j2]+=wnew[i][j][j2];
					if(i==wnew.length-1){
//						System.out.println("weights"+weights[i][j][j2]);
					}
				}
			}
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
	
	public static void main(String[]args){
		Network n;

		int test[] = {3,25,25,2};
//		n = new Network(test);
//		double[][][]w =new double[test[0]][][];
//		for (int i = 0; i < w.length; i++) {
//			w[i]=new double[test[i+1]][];
//			
//			for (int j = 0; j < w[i].length; j++) {
//				w[i][j]= new double[test[i]];
//				
//				for (int j2 = 0; j2 < w[i][j].length; j2++) {
//					w[i][j][j2]=0;
//				}
//			}
//		}
//
//		w[0][0][0]=0.15;
//		w[0][0][1]=0.25;
//		
//		w[0][1][0]=0.2;
//		w[0][1][1]=0.3;
//		
//		w[1][0][0]=0.4;
//		w[1][0][1]=0.5;
//		
//		w[1][1][0]=0.45;
//		w[1][1][1]=0.55;
//		
//		double[] []b= new double[2][];
//		for (int i = 0; i < test.length-1; i++) {
//			b [i] = new double [test[i+1]];
//			for (int j = 0; j < b[i].length; j++) {
//				b[i][j]=0;
//			}
//		}
//		b[0][0]=0.35;
//		b[0][1]=0.35;
//		
//		b[1][0]=0.6;
//		b[1][1]=0.6;
		
		double[][]input = {{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
		
//		n= new Network(test,b,w);
		n= new Network(test);
		byte []h = {0,1,1,0,1,0,0,1};
		n.test(input, h);
		for (int i = 0; i < h.length; i++) {
			double [][][] r = n.forward(input[i]);
			System.out.println(Arrays.toString(r[0][r[0].length-1]));
		}
//		r = n.forward(test2);
//		System.out.println(Arrays.toString(r[0][r[0].length-1]));
		
		
		int testn[] = {28*28,30,30,10};
		n = new Network(testn);
//		n.train();
		
	}
}
