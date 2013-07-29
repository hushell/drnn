/* 

generate_features.cpp

Code to generate color/texture features.

Originally written by Gary Huang.  Modified by Andrew Kae

*/

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <math.h>
#include <sys/stat.h>

using namespace std;

// specify dimensions of images
int iheight = 250, iwidth = 250;

// specify number of color clusters and texture clusters
int numClusters = 64;
int numTextClusters = 64;

// define directories, to be set later

//LFW Funneled Images directory
string lfw_dir;

// superpixels directory
string sp_dir;
string spseg_features_dir;
string label_dir;
string gt_dir;
string spmat_dir;
string texton_dir;
string pb_dir;

// Specify subroutines

// Compute Node Features
void computeNodeFeatures(string s, int hh, int ww, int **mat, int numSP, 
			 vector<vector<float> > &meanLab, vector<vector<float> > &texthists,
			 ofstream *outfile, float **clusters);
// Compute Edge features
void computeEdgeFeatures(string s, int n, int **mat, int numSP, 
			 vector<vector<float> > &meanLab, vector<vector<float> > &texthists,
			 ofstream *outfile);

// Read the labels from the ppm file
int readppmtomat(char *fn, int **&mat);

// read superpixel indices
int readSuperpixel(char *fn, int **&mat, int hh, int ww);

// Read the probability of boundary data 
void readpb(char *fn, float **&vals);

// Compute distance between 2 vectors
float dist(vector<float> a, vector<float> b);

// Compute distance between 2 vectors
void rgb2lab(int ir, int ig, int ib, float &l, float &a, float &b);

// Helper function to convert RGB to LAB
float labf(float t);

// Generate an alternative form of the ground truth
void computeGTppm(string s, int n, int **mat, int numSP);

// Determine which bin to put the color value 
int getBin(float l, float a, float b, float **clusters);

// Determine distance of color to a particular cluster
float dist(float l, float a, float b, float *cluster);

// Compute chi-squared distance between superpixels
float chisquared(vector<float> a, vector<float> b);

/////////////////////////////////////

// Main routine START
int main(int argc, char* argv[])
{

	if (argc != 11)
	{
		cout << "usage: " << argv[0] << " <File List> <LAB Cluster File> <LFW Directory> <Superpixel Directory> <Features Directory> <Label Directory> <Label Directory (Generate)> <Superpixel Directory (Generate)> <Texture Directory> <PB Directory>" << endl;
		exit(-1);
	}  

	//Specify directories
	string list = argv[1];		
	string lab_file = argv[2];

	lfw_dir = argv[3];
	sp_dir = argv[4];
	spseg_features_dir = argv[5];
	label_dir = argv[6];
	gt_dir = argv[7];
	spmat_dir = argv[8];
	texton_dir = argv[9];
	pb_dir = argv[10];		

	//check if LAB clusters exist already.  If they exist, load them.  If they don't then create it.
	float **clusters = new float*[numClusters];	
	
	for(int i=0; i<numClusters; i++) {
		clusters[i] = new float[3];
	}
			
	ifstream kmin(lab_file.c_str());
	if(kmin.is_open()) 
	{
		cout << "Loading LAB cluster file : " << lab_file << endl;
		
		for(int i=0; i<numClusters; i++)
		{
			for(int j=0; j<3; j++)
				kmin >> clusters[i][j];
		}		
	} else
	{
		cout << "Could not open LAB cluster file : " << lab_file << endl;
		cout << "Recreating LAB clusters..." << endl;		

		//Compute LAB clusters		
		//Read in the list of images
		ifstream infile(list.c_str());
		if(!infile.is_open())
		{
			cout << "could not open " << list << endl;
			exit(-1);
		}

		int th = 0;
		int tw = 0;
		string s;
		vector<string> fns;
		int nSamples = 0;
		while(true)
		{
			int hh, ww;
			infile >> s >> hh >> ww;
			if(infile.eof())
				break;

			//store the name and id 
			if (nSamples++ % 10 == 0)
			{
				cout << nSamples-1 << " " << s << " pushed." << endl;
				fns.push_back(s);

				th += hh;
				tw += ww;
			}
		}
		
		//process the images
		cout << "Processing " << fns.size() << " images." << endl;
		cout << "Need " << th << "*" << tw << " bytes." << endl;
		//CvMat *labs = cvCreateMat(fns.size()*iheight*iwidth/4, 3, CV_32FC1);
		CvMat *labs = cvCreateMat(th*tw, 3, CV_32FC1);
		
		int offset = 0;
		for(int index=0; index<fns.size(); index++)
		{
			cout << index << endl;
			s = fns[index];
			char fn[1024];

			//sprintf(fn, "%s/%s/%s_%04d.jpg", lfw_dir.c_str(), s.c_str(), s.c_str(), n);
			sprintf(fn, "%s/%s.png", lfw_dir.c_str(), s.c_str());

			IplImage *image = cvLoadImage(fn, -1);
			if(image == NULL)
			{
				cout << "could not open " << fn << endl;
				exit(-1);
			}
			
			int step = image->widthStep / sizeof(uchar);

			uchar *data = (uchar*)image->imageData;

			//convert RGB to LAB coordinates, sampling every other pixel.
			for(int i=0; i<image->height; i+=2)
			{
				for(int j=0; j<image->width; j+=2)
				{
					float l, a, b;

					int ir = data[i*step + j*3 + 2];
					int ig = data[i*step + j*3 + 1];
					int ib = data[i*step + j*3 + 0];

					rgb2lab(ir,ig,ib,l,a,b);
					labs->data.fl[offset + 0] = l;
					labs->data.fl[offset + 1] = a;
					labs->data.fl[offset + 2] = b;
					offset += 3;
				}
			}
			cvReleaseImage(&image);
		}

		CvMat *labels = cvCreateMat(labs->height, 1, CV_32SC1);

		cout << "Running kmeans with numClusters: " << numClusters << endl;

		//Cluster the LAB points
		cvKMeans2(labs, numClusters, labels, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, .001));

		//compute centroids

		//vector of 3 floats initialized to 0
		vector<int> numInC(numClusters, 0);
		vector<float> sigmaSq(numClusters, 0);

		for (int ilab = 0; ilab < labels->height; ilab++)
		{
			int label = labels->data.i[ilab];
			for (int ii = 0; ii < 3; ii++)
			{
				clusters[label][ii] += labs->data.fl[ilab*3+ii];
			}
			++numInC[label];
		}

		for(int i=0; i<numClusters; i++)
		{
			for(int j=0; j<3; j++)
				clusters[i][j] /= numInC[i];
		}

		cvReleaseMat(&labs);
		cvReleaseMat(&labels);

		//Write out the centroids.
		ofstream kmout(lab_file.c_str());
		for(int i=0; i<numClusters; i++)
		{
			for(int j=0; j<3; j++)
				kmout << clusters[i][j] << '\t';
			kmout << endl;
		}
	} 

	//Compute features		
	cout << "Computing Features..." << endl;	

	//create the directory if it doesn't already exist
	char spseg_dir[1024];
	struct stat st;

	sprintf(spseg_dir, "%s", spseg_features_dir.c_str());     

	if(stat(spseg_dir,&st) != 0) {
		//create it
		mkdir(spseg_dir, S_IRWXU);
	} 

	//read list
	ifstream infile(list.c_str());

	int nImages = 0;
	while(true)
	{
		string s;
		int hh, ww;
		infile >> s >> hh >> ww;
		cout << nImages++ << " " << s << endl;
		if(infile.eof())
			break;

		//mat is a 2D array of superpixel label ids.  Each pixel is assigned the label of the superpixel.
		int **mat;
		mat = new int*[hh];
		for(int i=0; i<hh; i++)
			mat[i] = new int[ww];

		char fn[1024];
		sprintf(fn, "%s/%s_seg.dat", sp_dir.c_str(), s.c_str());

		//int numSP = readppmtomat(fn, mat);
		int numSP = readSuperpixel(fn, mat, hh, ww);

		sprintf(fn, "%s/%s_spfeat.dat", spseg_features_dir.c_str(), s.c_str());     
		ofstream outfile(fn);
		if(!outfile.is_open())
		{
			cout << "could not create " << fn << endl;
			exit(-1);
		}
		outfile << numSP << endl << endl;

		//mean color of each superpixel
		vector<vector<float> > meanLab;
		
		//texture histogram of each superpixel
		vector<vector<float> > texthists;
		
		computeNodeFeatures(s, hh, ww, mat, numSP, meanLab, texthists, &outfile, clusters);
		//computeEdgeFeatures(s, n, mat, numSP, meanLab, texthists, &outfile);
		//computeGTppm(s, n, mat, numSP);

		for(int i=0; i<hh; i++)
			delete[] mat[i];
		delete[] mat;
	}

	for(int i=0; i<numClusters; i++)
		delete[] clusters[i];
	delete[] clusters;
}

/*
* END Main Routine
*/

/////////////////////////////

/*   
   Compute Node Features
	
   % s,n : specify filename
   % mat : 2D matrix superpixel IDs
   % numSP : number of superpixels
   % meanLab : mean color of each superpixel -> input of edge feat
   % texthists : texture histogram of each superpixel -> input of edge feat
   % outfile : 	feature file 
   % clusters : cluster centroids (64) 
*/
void computeNodeFeatures(string s, int hh, int ww, 
		     int **mat, int numSP, 
			 vector<vector<float> > &meanLab, 
			 vector<vector<float> > &texthists,
			 ofstream *outfile, float **clusters)
{
  //int posFOffset = numClusters;
  //int numGridFeatures = 64;
  int texFOffset = numClusters;
  
  //int numFeatures = numClusters + numGridFeatures;
  int numFeatures = numClusters + numTextClusters;

  vector<float> instanceNFR(numFeatures, 0);
  vector<vector<float> > instanceNF(numSP, instanceNFR);

  vector<float> empty(3, 0);
  meanLab.resize(numSP, empty);
  vector<float> textempty(numTextClusters, 0);
  texthists.resize(numSP, textempty);

  vector<int> numInSP(numSP, 0);

  char fn[1024];
  
  //Funneled LFW images
  sprintf(fn, "%s/%s.png", lfw_dir.c_str(), s.c_str());

  IplImage *image = cvLoadImage(fn, -1);
  if(image == NULL)
    {
      cout << "couldn't open " << fn << endl;
      exit(-1);
    }
  int step = image->widthStep/sizeof(uchar);	  

  uchar *data = (uchar*)image->imageData;

  float minLabL = 100, maxLabL = -100, minLaba = 100, maxLaba = -100,
    minLabb = 100, maxLabb = -100;

  //look for textons
  sprintf(fn, "%s/%s_tex.dat", texton_dir.c_str(), s.c_str());
  ifstream texin(fn);
  if(!texin.is_open())
    {
      cout << "could not open " << fn << endl;
      exit(-1);
    } 
  
  for(int i=0; i<hh; i++)
    {
      for(int j=0; j<ww; j++)
	{
	  int sp = mat[i][j];
	  ++numInSP[sp];

	  ////NOTE: 250/8 = 31.25 ie an 8x8 grid
	  ///*Spatial/Position Feature*/
	  ////get a bin between 1-8
	  //int y = (int)(i / 31.25);
	  //int x = (int)(j / 31.25);
	  //
	  ////put the "vote" into the appropriate bin -- normalize later
	  //++instanceNF[sp][posFOffset+(y<<3)+x];

	  /* Texton Feature */
	  int texcluster;
	  texin >> texcluster;
	  --texcluster;
	  ++instanceNF[sp][texFOffset + texcluster];
	  ++texthists[sp][texcluster];	 
	  
	  /* Color Feature */
	  int r = data[i*step + j*3 + 2];
	  int g = data[i*step + j*3 + 1];
	  int b = data[i*step + j*3 + 0];
	  float LabL, Laba, Labb;
	  rgb2lab(r,g,b,LabL, Laba, Labb);
	  //cout << LabL << '\t' << Laba << '\t' << Labb << endl;
	  if(LabL > maxLabL)
	    maxLabL = LabL;
	  if(LabL < minLabL)
	    minLabL = LabL;
	  if(Laba > maxLaba)
	    maxLaba = Laba;
	  if(Laba < minLaba)
	    minLaba = Laba;
	  if(Labb > maxLabb)
	    maxLabb = Labb;
	  if(Labb < minLabb)
	    minLabb = Labb;

	  ++instanceNF[sp][getBin(LabL, Laba, Labb, clusters)];

	  meanLab[sp][0] += LabL;
	  meanLab[sp][1] += Laba;
	  meanLab[sp][2] += Labb;
	}
    }

  	(*outfile) << numFeatures+1 << endl; // constant features

	for(int i=0; i<numSP; i++)
	{
		(*outfile) << 1 << '\t'; // constant feature

		for(int j=0; j<numFeatures; j++)
		{
			instanceNF[i][j] /= numInSP[i];
			(*outfile) << instanceNF[i][j] << '\t';
		}
		(*outfile) << endl;

		for(int k=0; k<3; k++)
			meanLab[i][k] /= numInSP[i];

		for(int k=0; k<numTextClusters; k++)
			texthists[i][k] /= numInSP[i];      

	}
	(*outfile) << endl;

	texin.close();
}

/*   
   Compute Edge features

   % s,n : specify filename
   % mat : 2D matrix superpixel IDs
   % numSP : number of superpixels
   % meanLab : mean color of each superpixel
   % texthists : texture histogram of each superpixel
   % outfile : 	feature file 
*/
  
void computeEdgeFeatures(string s, int n, int **mat, int numSP, 
			 vector<vector<float> > &meanLab, 
			 vector<vector<float> > &texthists, ofstream *outfile)
{
  map<pair<int, int>, int> foundEdges;
  vector<pair<int, int> > instanceEdges;

  int numFeatures = 1 + 1 + 1;
  vector<float> edgeFeatures;

  //Tracks number of pixels along boundaries of each pair of superpixels
  vector<float> bT;

  //Retrieve the Probability of Boundary feature
  char fn[1024];

  sprintf(fn, "%s/%s/%s_%04d.emag_thick.txt", pb_dir.c_str(), s.c_str(), s.c_str(), n);  
  
  float **vals;
  vals = new float*[iheight];
  for(int i=0; i<iheight; i++)
    vals[i] = new float[iwidth];

  //this returns PB of whether there is a boundary at img[i][j]
  readpb(fn, vals);
  
  for(int i=0; i<iheight-1; i++)
    {
      for(int j=0; j<iwidth-1; j++)
	{
	  if(mat[i][j] != mat[i+1][j])
	    {
	      float val = (vals[i][j] + vals[i+1][j]);

	      //impose an ordering
	      pair<int, int> p(min(mat[i][j], mat[i+1][j]), max(mat[i][j], mat[i+1][j]));
	      if(foundEdges.find(p) == foundEdges.end())
		{
		  foundEdges[p] = instanceEdges.size();
		  instanceEdges.push_back(p);
		  edgeFeatures.push_back(val);
		  bT.push_back(1);
		}
	      else
		{
		  edgeFeatures[foundEdges[p]] += val;
		  bT[foundEdges[p]]++; 
		}
	    }
	  if(mat[i][j] != mat[i][j+1])
	    {
	      float val = (vals[i][j] + vals[i][j+1]);

	      pair<int, int> p(min(mat[i][j], mat[i][j+1]), max(mat[i][j], mat[i][j+1]));
	      if(foundEdges.find(p) == foundEdges.end())
		{
		  foundEdges[p] = instanceEdges.size();
		  instanceEdges.push_back(p);
		  edgeFeatures.push_back(val);
		  bT.push_back(1);
		}
	      else
		{
		  edgeFeatures[foundEdges[p]] += val;
		  bT[foundEdges[p]]++;
		}
	    }
	}
    }

  (*outfile) << instanceEdges.size() << '\t' << numFeatures+1 << endl; // constant feature

  for(int i=0; i<instanceEdges.size(); i++)
    {
      int node1 = instanceEdges[i].first;
      int node2 = instanceEdges[i].second;

      (*outfile) << node1 << '\t' << node2 << '\t';
      
      (*outfile) << 1 << '\t' << edgeFeatures[i] << '\t' 
		 << dist(meanLab[node1],meanLab[node2]) <<  '\t' 
		 << chisquared(texthists[node1], texthists[node2]) << endl; // include constant feature 
    }

  for(int i=0; i<iheight; i++)
    delete[] vals[i];
  delete[] vals;
}

/*
  Read the labels from the ppm file

   % fn : specify filename
   % mat : 2D matrix superpixel IDs
 */

int readppmtomat(char *fn, int **&mat)
{
  map<int, int> ppm2mat;
  ifstream infile(fn);
  if(!infile.is_open())
    {
      cout << "could not open " << fn << endl;
      exit(-1);
    }
  string s;
  infile >> s; // P3
  int r, g, b, n, m, x;
  infile >> n >> m >> x;

  for(int i=0; i<m; i++)
    {
      for(int j=0; j<n; j++)
	{
	  infile >> r >> g >> b;
	  x = (r << 16) + (g << 8) + b;
	  if(ppm2mat.find(x) == ppm2mat.end())
	    {
	      int s = ppm2mat.size();
	      ppm2mat[x] = s;
	      mat[i][j] = ppm2mat[x];
	    }
	  else
	    {
	      mat[i][j] = ppm2mat[x];
	    }
	}
    }
  return ppm2mat.size();
}

int readSuperpixel(char *fn, int **&mat, int hh, int ww)
{
  map<int, int> ppm2mat;
  ifstream infile(fn);
  if(!infile.is_open())
    {
      cout << "could not open " << fn << endl;
	  return -1;
    }
  int n, m, x;
  infile >> m >> n;

  if (m != hh || n != ww)
  {
	cout << "size mismatched!" << endl;
	return -1;
  }

  for(int i=0; i<m; i++)
    {
      for(int j=0; j<n; j++)
	  {
	      infile >> x;
	      if(ppm2mat.find(x) == ppm2mat.end())
	        {
	          int s = ppm2mat.size();
	          ppm2mat[x] = s;
	          mat[i][j] = ppm2mat[x];
	        }
	      else
	        {
	          mat[i][j] = ppm2mat[x];
	        }
	  }
    }
  return ppm2mat.size();
}

/* 
   Read the probability of boundary data 

   % fn : filename
   % vals : matrix of Pb values
*/


void readpb(char *fn, float **&vals)
{
  ifstream infile(fn);
  if(!infile.is_open())
    {
      cout << "could not open " << fn << endl;
      exit(-1);
    }

  for(int i=0; i<iheight; i++)
    {
      for(int j=0; j<iwidth; j++)
	infile >> vals[i][j];
    }
}

/* 
   Compute distance between 2 vectors

   % a,b : vectors
*/

float dist(vector<float> a, vector<float> b)
{
  float r=0;
  for(int i=0; i<a.size(); i++)
    r += (a[i]-b[i])*(a[i]-b[i]);
  return(sqrt(r));
}

/* 
   Convert RGB point to LAB

   % ir,ig,ib : RGB
   % l,a,b : LAB
*/

void rgb2lab(int ir, int ig, int ib, float &l, float &a, float &b)
{
  float fr = ir / 255.0;
  float fg = ig / 255.0;
  float fb = ib / 255.0;

  float X = fr * 0.412453 + fg * 0.357580 + fb * 0.180423;
  float Y = fr * 0.212671 + fg * 0.715160 + fb * 0.072169;
  float Z = fr * 0.019334 + fg * 0.119193 + fb * 0.950227;

  X /= 0.950456;
  Z /= 1.088754;
  
  if(Y>0.008856)
    l = 116*pow(Y,1/3.0f)-16;
  else
    l = 903.3*Y;

  a = 500.0*(labf(X) - labf(Y));
  b = 200.0*(labf(Y) - labf(Z));
}

/* 
   Helper function to convert RGB to LAB
*/

float labf(float t)
{
  if(t > 0.008856)
    return pow(t, 1/3.0f);
  else
    return 7.787*t + 16/116.0;
}

/*
    Generate an alternative form of the ground truth

	% s,n : specify filename
	% mat : 2D matrix superpixel IDs
	% numSP : number of superpixels
*/
void computeGTppm(string s, int n, int **mat, int numSP)
{
  char fn[1024];
  sprintf(fn, "%s/%s_%04d.ppm", label_dir.c_str(),
	  s.c_str(), n);

  IplImage* img=0; 
  img=cvLoadImage(fn);

  vector<int> instanceGT(numSP, 0);

  if(!img)  
    {
      printf("Could not load image file: %s\n", fn);
      
      exit(1);
      
      // leave labels as all background
    } 
  else 
    {
      int step = img->widthStep / sizeof(uchar);  
      uchar *data = (uchar*)img->imageData;

      int label;
      for(int i=0; i<iheight; i++)
	{
	  for(int j=0; j<iwidth; j++)
	    {
	      int r = data[i*step + j*3 + 2];
	      int g = data[i*step + j*3 + 1];
	      int b = data[i*step + j*3 + 0];	      

	      if (r == 255) 
		instanceGT[mat[i][j]] = 0;
	      else if (g == 255)
		instanceGT[mat[i][j]] = 1;
	      else if (b == 255) 
		instanceGT[mat[i][j]] = 2;
	    }
	}
    }

  //create the directory if it doesn't already exist
  char fn_dir[1024];
  sprintf(fn_dir, "%s/%s/", gt_dir.c_str(), s.c_str());     
  
  struct stat st;
  if(stat(fn_dir,&st) != 0) {
    //create it
    mkdir(fn_dir, S_IRWXU);
  } 

  sprintf(fn, "%s/%s/%s_%04d.dat", gt_dir.c_str(), s.c_str(), s.c_str(), n);

  ofstream outfile(fn);
  outfile << numSP << endl;
  for(int i=0; i<numSP; i++)
    outfile << instanceGT[i] << endl;
}

/* 
   Determine which bin to put the color value 

   % l,a,b : color
   % clusters : LAB color centroids	
*/

int getBin(float l, float a, float b, float **clusters)
{
  float bd = dist(l,a,b,clusters[0]);
  int bi = 0;
  for(int i=1; i<numClusters; i++)
    {
      float td = dist(l,a,b,clusters[i]);
      if(td < bd)
	{
	  bd = td;
	  bi = i;
	}
    }
  return bi;
}

/* 
   Determine distance of color to a particular cluster

   % l,a,b : color
   % cluster : LAB color centroid
*/

float dist(float l, float a, float b, float *cluster)
{
  return (l-cluster[0])*(l-cluster[0])+(a-cluster[1])*(a-cluster[1])+(b-cluster[2])*(b-cluster[2]);
}

/* 
   Compute chi-squared distance between superpixels

   % a,b : vectors
*/

float chisquared(vector<float> a, vector<float> b)
{
  float r=0;
  for(int i=0; i<a.size(); i++)
    {
      float denom = a[i] + b[i];
      if(denom > 0.0000001)
	r += ( (a[i] - b[i]) * (a[i] - b[i]) ) / denom;
    }
  return r/2.0;
}
