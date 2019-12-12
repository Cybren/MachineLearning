package visuals;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class Graph extends JPanel{
	private static final long serialVersionUID = 559846568445826941L;
	Dimension d = new Dimension(500,500);
	int ax=20,ay=20;
	int draws [];
	int vmax=0,vmin=0,vlength=0;
	Color active=Color.BLACK;
	
	public Graph(int []a){
		this.setPreferredSize(d);
		draws=a;
		vlength=draws.length;
		getMaxMin();
		System.out.println(vmin+" "+vmax);
	}
	
	public void getMaxMin(){
		for (int i = 0; i < draws.length; i++) {
			if(draws[i]>vmax){
				vmax=draws[i];
			}
			if(draws[i]<vmin){
				vmin=draws[i];
			}
		}
	}
	
	public void makegui(){
		JFrame f1 = new JFrame("Graph");
		f1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f1.add(this);
		f1.pack();
		f1.setLocationRelativeTo(null);
		f1.setVisible(true);
		
	}
	public void setColor(Color c){
		active = c;
	}
	public void paint (Graphics g){//2 fors
		System.out.println("draw");
		int ly = this.getHeight()-ay, lx = this.getWidth()-ax;

		g.setColor(Color.WHITE);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());
		g.setColor(Color.BLACK);
		g.drawLine(ax, ay, ax, ly+5);
		g.drawLine(ax-5, ly, lx, ly);
		if(vlength!=0) {
		double sy= (double)ly/vmax;
		double sx= (double)lx/vlength;
		System.out.println(sx);
		System.out.println(lx/vlength);
		System.out.println(lx+" "+vlength);
		g.drawString(Integer.toString(vmax), 0, 15);
		g.drawString(Integer.toString(vlength), lx, ly+15);
		g.setColor(active);
			if(sx>0&&sy>0){
				for (int i = 0; i < draws.length-1; i++) {
					g.drawLine((int)(ax+sx*i), (int)(ly-draws[i]*sy), (int)(ax+sx*(i+1)), (int)(ly-draws[i+1]*sy));
					
					}
				for (int i = 0; i < draws.length; i++) {
					g.fillOval((int)(ax+sx*i-2), (int) (ly-draws[i]*sy-2), 4, 4);
				}
			}else if(sx>0&&sy<0){
				if(vmax>100000){ devide(10000);}else if(vmax>10000){devide(1000);}else if(vmax>1000){devide(100);}
				getMaxMin();
				sy=(double)ly/vmax;
				for (int i = 0; i < draws.length-1; i++) {
					g.drawLine((int)(ax+sx*i), (int)(ly-draws[i]*sy), (int)(ax+sx*(i+1)), (int)(ly-draws[i+1]*sy));
				}
				for (int i = 0; i < draws.length; i++) {
					g.fillOval((int)(ax+sx*i-2), (int) (ly-draws[i]*sy-2), 4, 4);
				}
			}else if(sx<0&&sy>0){
				sx=1/sx;
				System.out.println(sx);
				for (int i = 0; i < draws.length/sx; i++) {
					for (int j = 0; j < sx; j++) {
						g.drawLine((int)(ax+sx*i), (int)(ly-draws[i]*sy), (int)(ax+sx*(i+1)), (int)(ly-draws[i+1]*sy));
					}
					for (int j = 0; j < draws.length; j++) {
						g.fillOval((int)(ax+sx*i-2), (int) (ly-draws[i]*sy-2), 4, 4);
					}
					
				}
			}else{
				sx=1/sx;
				if(vmax>100000){ devide(10000);}else if(vmax>10000){devide(1000);}else if(vmax>1000){devide(100);}
				sy=(double)ly/vmax;
				for (int i = 0; i < draws.length/sx; i++) {
					for (int j = 0; j < sx; j++) {
						g.drawLine((int)(ax+sx*i), (int)(ly-draws[i]*sy), (int)(ax+sx*(i+1)), (int)(ly-draws[i+1]*sy));
					}
					for (int j = 0; j < draws.length; j++) {
						g.fillOval((int)(ax+sx*i-2), (int) (ly-draws[i]*sy-2), 4, 4);
					}
				}
			}
		}
	}
	public void devide(int a){
		for (int i = 0; i < draws.length; i++) {
			draws[i]=draws[i]/a;
		}
	}
	public static void main(String[] args) {
//		int [] a ={0,10,20,30,40,50,60,70,50,80,30,90,100,120,20,40,0};
		int []a= new int[1000];
		for (int i = 0; i < a.length; i++) {
			a[i]=(int)(Math.random()*1000);
		}
		Graph m = new Graph(a);
		m.setColor(Color.BLUE);
		m.makegui();
	}

}
