package neuralnetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class KMeans {
	
	int k = 0;//number of clusters
	double data[][];//number of vektors, length of each vektor
	double prototypes[][];//k, length of each vektor
	ArrayList<Integer>[] clusters;//k,
	double max=0, min=0;
	double [] maxV,minV;
	int vLength;
	
	public KMeans(int k, double [][]data) {
		this.k=k;
		this.data=data;
		prototypes = new double[k][data[0].length];
		vLength = data[0].length;
		minV=data[0].clone();
		maxV=data[0].clone();
		System.out.println(data[0].length);
		System.out.println(Arrays.toString(minV));
		System.out.println(Arrays.toString(maxV));
		for (int i = 1; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				System.out.println("data "+i+","+j+" :"+data[i][j]);
				if(data[i][j]<minV[j]) {
					minV[j]=data[i][j];
					System.out.println("new min "+i+","+j+" : "+data[i][j]);
				}else if(data[i][j]>maxV[j]){
					maxV[j]=data[i][j];
					System.out.println("new max "+i+","+j+" : "+data[i][j]);
				}
			}
			System.out.println("max: "+Arrays.toString(maxV));
			System.out.println("min: "+Arrays.toString(minV));
		}
		
		for (int i = 0; i < k; i++) {//ini protos
			for (int x = 0; x < maxV.length; x++) {
				prototypes[i][x]=ThreadLocalRandom.current().nextDouble(minV[x], maxV[x]);
			}
			System.out.println(Arrays.toString(prototypes[i]));
		}
	}
	
	public KMeans(int k, double [][]data,double[][]prototypes) {
		this.k=k;
		this.data=data;
		this.prototypes = prototypes;
	}
	
	@SuppressWarnings("unchecked")
	public int[][] cluster(int times) {//muss noch die neuen prototypes setzen und times benutzen
		int[][] ret = new int[k][0];
		double[] temp;
		int add=0;
		
		
		for (int h = 0; h < times; h++) {
			clusters = (ArrayList<Integer>[])new ArrayList[k];// ini clusters
			for (int i = 0; i < k; i++) {
				clusters[i]=new ArrayList<Integer>();
			}
			for (int i = 0; i < data.length; i++) {//determine the closest proto and add the datapoint to the coresponding cluster
				temp = new double[k];
				for (int j = 0; j < prototypes.length; j++) {
					temp[j]=0;
					for (int j2 = 0; j2 < prototypes[j].length; j2++) {
						temp[j]+=(prototypes[j][j2]-data[i][j2])*(prototypes[j][j2]-data[i][j2]);
					}
					temp[j]=Math.sqrt(temp[j]);
				}
				System.out.println("temp: "+Arrays.toString(temp));
				add = 0;
				min=temp[0];
				for (int j = 1; j < temp.length; j++) {
					if(temp[j]<min) {
						add = j;
					}
				}
				System.out.println(add);
				clusters[add].add(i);
			}
			//set new protos
			for (int i = 0; i < clusters.length; i++) {//k
				if(clusters[i].size()==0) {
					for (int x = 0; x < maxV.length; x++) {
						prototypes[i][x]=ThreadLocalRandom.current().nextDouble(minV[x], maxV[x]);
					}
					System.out.println("random");
					h-=1;
				}else {
					double vtemp[]=new double[data[0].length];//temp für die neuen koordinaten von prototyp i
					for (int j = 0; j < clusters[i].size(); j++) {//iteriere über alle Elemente in Cluster i 
						for (int j2 = 0; j2 < vtemp.length; j2++) {
							vtemp[j2]=0;
						}
						for (int j2 = 0; j2 < vtemp.length; j2++) {//iteriere über alle Koordinaten j2 von Element j aus Cluster i
							vtemp[j2]-=prototypes[i][j2]-data[clusters[i].get(j)][j2];
						}
					}
					System.out.println("cluster "+i+": "+clusters[i]);
					for (int j = 0; j < vtemp.length; j++) {//set new koordiantes from prototype i
						prototypes[i][j]+=vtemp[j]/clusters[i].size();// immer näher annähern, nicht absolut
					}
				}
				System.out.println("prototype "+i+": "+Arrays.toString(prototypes[i]));
			}
		}
		
		for(int i=0;i<clusters.length;i++) {
			System.out.println("size: "+clusters[i].size());
			ret[i]=new int[clusters[i].size()];
			for(int x=0;x<ret[i].length;x++) {
				ret[i][x]=clusters[i].get(x);
			}
		}
		return ret;
	}
	
	
	public static void main(String[] args) {
		//double [][] data= {{1,1},{0,1},{1,0},{11,12},{11,13},{13,13},{12,8.5},{13,8},{13,9},{13,7},{11,7},{8,2},{9,2},{10,1},{7,13},{5,9},{16,16},{11.5,8},{13,10},{12,13},{14.5,12.5},{14.5,11.5},{15,10.5},{15,9.5},{12,9.5},{10.5,11},{10,10.5},{9,3},{9,4},{9,5}};
		double [][] data= {{1,1},{0,1},{1,0},{0,0},{4,4},{4,5},{5,4},{5,5},{8,8},{9,8},{8,9},{9,9}};
		KMeans k = new KMeans(3,data);
		int[][]cluster = k.cluster(100);
		for (int i = 0; i < cluster.length; i++) {
			System.out.println(Arrays.toString(cluster[i]));
		}
	}

}
