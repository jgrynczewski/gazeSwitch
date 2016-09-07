/**
   Program presents set of steps of the algorithm used in 'eyeSwitch' program.
   
   To see dependencies read README file. 
   
   To run type:
   cmake .
   make
   ./steps <Image_Path>

   @author Jerzy Grynczewski
   @version 2.0
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <boost/lexical_cast.hpp>

using namespace cv;

int main( int argc, char** argv )
{
  if ( argc != 2)
    {
      std::cout<<"usage: steps.out <Image_Path>\n";
      return -1;
    }
  
  Mat img, gray_image, filtered_image, thresh_pupil_image, contours_pupil_image, contours_pupil_image_BGR, thresh_glint_image, contours_glint_image, contours_glint_image_BGR, pupil_detection, glint_detection, distances_image, decision_image;
  img = imread( argv[1], 1 );
  
  char pressed_button=' ';
  
  if ( !img.data )
    {
      std::cout << "No image data \n";
      return -1;
    }

  // flip(img, img, 1);
  cvtColor(img, gray_image, COLOR_BGR2GRAY);
  GaussianBlur(gray_image, filtered_image, Size(7,7), 0, 0);

  int x_border = 20, y_border = 2*x_border, x_size = 300, y_size = 225;

  // Pupil contours

  int pupil_thresh = 35;
  threshold(filtered_image, thresh_pupil_image, pupil_thresh, 255, THRESH_BINARY_INV);
  
  Scalar pupil_color( 255, 0, 0 );
  Scalar pupil_contours_color( 0, 255, 255 );
  vector<vector<Point> > pupil_contours;
  vector<Vec4i> pupil_hierarchy;
  
  contours_pupil_image = thresh_pupil_image.clone();
  findContours(contours_pupil_image, pupil_contours, pupil_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
  cvtColor(contours_pupil_image, contours_pupil_image_BGR, COLOR_GRAY2BGR);
  drawContours(contours_pupil_image_BGR, pupil_contours, -1, pupil_contours_color, 2);

  // Pupil detection

  int i = 0;

  Moments pupil_mu;
  std::vector<int> pupil_x(pupil_contours.size());
  std::vector<int> pupil_y(pupil_contours.size());
  std::vector<int> pupil_radius(pupil_contours.size());
  double area, pupil_r;
  Rect pupil_bound;
  
  bool dilation_flag = false;
  
  cvtColor(contours_pupil_image, pupil_detection, COLOR_GRAY2BGR);

  vector<vector<Point> > pupil_contours_poly( pupil_contours.size() );
  vector<Rect> pupil_boundRect( pupil_contours.size() );
  vector<Point2f>pupil_center( pupil_contours.size() );
  vector<float>pupil_radius2( pupil_contours.size() );
  
  int nr_dilation_applied = 5;
  double pupil_area_limit=0.5, pupil_rect_limit=0.45, min_pupil_area=900, max_pupil_area=8000;

  for(int idx=0; idx<pupil_contours.size(); idx++)
    {

      pupil_mu = moments(pupil_contours[idx], false);
      area = pupil_mu.m00;

      approxPolyDP( Mat(pupil_contours[idx]), pupil_contours_poly[idx], 3, true );
      pupil_boundRect[idx] = boundingRect( Mat(pupil_contours_poly[idx]) );
      minEnclosingCircle( (Mat)pupil_contours_poly[idx], pupil_center[idx], pupil_radius2[idx] );
      
      if (area>min_pupil_area && area<max_pupil_area && std::abs(1-(area/(3.14*pupil_radius2[idx]*pupil_radius2[idx])))<pupil_area_limit && std::abs(1-1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width)<pupil_rect_limit)
        {
	  	  
	  std::cout << "\nŹrenica nr " << idx+1 << " | R=" << pupil_radius2[idx] << std::endl << min_pupil_area<<"<P<"<<max_pupil_area<<" = " << area << " | P_k=" << 3.14*pupil_radius2[idx]*pupil_radius2[idx] << " | P/P_k="<< area/(3.14*pupil_radius2[idx]*pupil_radius2[idx]) << std::endl << "Warunek1 (1-P/P_k<"<<pupil_area_limit<<"): " << std::abs(1-(area/(3.14*pupil_radius2[idx]*pupil_radius2[idx]))) << std::endl << "height_rect=" << pupil_boundRect[idx].height << " | width_rect=" << pupil_boundRect[idx].width << " | h/w=" << 1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width << std::endl << "Warunek2 (1-h/w<" << pupil_rect_limit<<"): " << std::abs(1-1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width) << std::endl;
	  
      	  drawContours(pupil_detection, pupil_contours, idx, pupil_color, 2);

      	  pupil_x[i]=pupil_mu.m10/area;
      	  pupil_y[i]=pupil_mu.m01/area;
      	  pupil_radius[i]=sqrt(area/M_PI);
	  
      	  line(pupil_detection, Point(pupil_x[i]-5, pupil_y[i]-5), Point(pupil_x[i]+5,pupil_y[i]+5), pupil_color, 2, 8);
      	  line(pupil_detection, Point(pupil_x[i]-5, pupil_y[i]+5), Point(pupil_x[i]+5,pupil_y[i]-5), pupil_color, 2, 8);
      	  i++;
        }
    }    

  if (i==0) //which means pupil not found. Than we execute the same block of code as above, but with dilation
    {

      dilate(thresh_pupil_image, thresh_pupil_image, Mat(), Point(-1, -1), nr_dilation_applied, 1, 1); //mainly in case of torus (glint exactly in the middle of the pupil)
      dilation_flag = true;
      
      contours_pupil_image = thresh_pupil_image.clone();
      findContours(contours_pupil_image, pupil_contours, pupil_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
      cvtColor(contours_pupil_image, contours_pupil_image_BGR, COLOR_GRAY2BGR);
      drawContours(contours_pupil_image_BGR, pupil_contours, -1, pupil_contours_color, 2);

      pupil_x.resize(pupil_contours.size());
      pupil_y.resize(pupil_contours.size());
      pupil_radius.resize(pupil_contours.size());

      cvtColor(contours_pupil_image, pupil_detection, COLOR_GRAY2BGR);

      pupil_contours_poly.resize( pupil_contours.size() );
      pupil_boundRect.resize( pupil_contours.size() );
      pupil_center.resize( pupil_contours.size() );
      pupil_radius2.resize( pupil_contours.size() );

      for(int idx=0; idx<pupil_contours.size(); idx++)
	{

	  pupil_mu = moments(pupil_contours[idx], false);
	  area = pupil_mu.m00;

	  approxPolyDP( Mat(pupil_contours[idx]), pupil_contours_poly[idx], 3, true );
	  pupil_boundRect[idx] = boundingRect( Mat(pupil_contours_poly[idx]) );
	  minEnclosingCircle( (Mat)pupil_contours_poly[idx], pupil_center[idx], pupil_radius2[idx] );
      
	  if (area>min_pupil_area && area<max_pupil_area && std::abs(1-(area/(3.14*pupil_radius2[idx]*pupil_radius2[idx])))<pupil_area_limit && std::abs(1-1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width)<pupil_rect_limit)
	    {
	      if (dilation_flag == true)
		std::cout << "\nPRZYŁOŻONO DYLACJĘ ("<< nr_dilation_applied <<" razy)"<<std::endl;
	      
	      std::cout << "Źrenica nr " << idx+1 << " | R=" << pupil_radius2[idx] << std::endl << min_pupil_area<<"<P<"<<max_pupil_area<<" = " << area << " | P_k=" << 3.14*pupil_radius2[idx]*pupil_radius2[idx] << " | P/P_k="<< area/(3.14*pupil_radius2[idx]*pupil_radius2[idx]) << std::endl << "Warunek1 (1-P/P_k<"<<pupil_area_limit<<"): " << std::abs(1-(area/(3.14*pupil_radius2[idx]*pupil_radius2[idx]))) << std::endl << "height_rect=" << pupil_boundRect[idx].height << " | width_rect=" << pupil_boundRect[idx].width << " | h/w=" << 1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width << std::endl << "Warunek2 (1-h/w<" << pupil_rect_limit<<"): " << std::abs(1-1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width) << std::endl;
	  
	      drawContours(pupil_detection, pupil_contours, idx, pupil_color, 2);

	      pupil_x[i]=pupil_mu.m10/area;
	      pupil_y[i]=pupil_mu.m01/area;
	      pupil_radius[i]=sqrt(area/M_PI);
	  
	      line(pupil_detection, Point(pupil_x[i]-5, pupil_y[i]-5), Point(pupil_x[i]+5,pupil_y[i]+5), pupil_color, 2, 8);
	      line(pupil_detection, Point(pupil_x[i]-5, pupil_y[i]+5), Point(pupil_x[i]+5,pupil_y[i]-5), pupil_color, 2, 8);
	      i++;
	    }
	}    

    }

  // Glint contours

  int glint_thresh = 100; //empirically chosen (with reserve) 135
  threshold( filtered_image, thresh_glint_image, glint_thresh, 255, THRESH_BINARY);
  
  Scalar glint_color(255, 0, 0);
  Scalar glint_contours_color(0, 255, 255);
  vector<vector<Point> > glint_contours;
  vector<Vec4i> glint_hierarchy;

  contours_glint_image = thresh_glint_image.clone();
  findContours(contours_glint_image, glint_contours, glint_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  cvtColor(contours_glint_image, contours_glint_image_BGR, COLOR_GRAY2BGR);
  drawContours(contours_glint_image_BGR, glint_contours, -1, glint_contours_color, 2);

  int glint_radius;
  int lenght = 3, lenght2 = 4, thickness = 2;
  Moments glint_mu;
  Rect glint_bound;
  int j=0;

  // Glint detection
  
  std::vector<double> glint_x(glint_contours.size());
  std::vector<double> glint_y(glint_contours.size());
  std::vector<double> distance(glint_contours.size());

  vector<vector<Point> > glint_contours_poly( glint_contours.size() );
  vector<Rect> glint_boundRect( glint_contours.size() );
  vector<Point2f> glint_center( glint_contours.size() );
  vector<float> glint_radius2( glint_contours.size() );
  
  double glint_area_limit=0.35, glint_rect_limit=0.35, min_glint_area=100, max_glint_area=1000;
  
  cvtColor(contours_glint_image, glint_detection, COLOR_GRAY2BGR);
  
  for(int idx=0; idx< glint_contours.size(); idx++)
    {

      glint_mu = moments(glint_contours[idx], false);
      area = glint_mu.m00;

      approxPolyDP( Mat(glint_contours[idx]), glint_contours_poly[idx], 3, true );
      glint_boundRect[idx] = boundingRect( Mat(glint_contours_poly[idx]) );
      minEnclosingCircle( (Mat)glint_contours_poly[idx], glint_center[idx], glint_radius2[idx] );

      std::cout << "\nOdbicie nr " << idx+1 << " | R=" << glint_radius2[idx] << std::endl << "P=" << area << " | P_k=" << 3.14*glint_radius2[idx]*glint_radius2[idx] << " | P/P_k="<< area/(3.14*glint_radius2[idx]*glint_radius2[idx]) << std::endl << "Warunek1 (1-P/P_k): " << std::abs(1-(area/(3.14*glint_radius2[idx]*glint_radius2[idx]))) << std::endl << "height_rect=" << glint_boundRect[idx].height << " | width_rect=" << glint_boundRect[idx].width << " | h/w=" << 1.*glint_boundRect[idx].height/glint_boundRect[idx].width << std::endl << "Warunek2 (1-h/w): " << std::abs(1-1.*glint_boundRect[idx].height/glint_boundRect[idx].width) << std::endl;

      if ( area>min_glint_area && area<max_glint_area && std::abs(1-(area/(3.14*glint_radius2[idx]*glint_radius2[idx])))<glint_area_limit && std::abs(1-1.*glint_boundRect[idx].height/glint_boundRect[idx].width)<glint_rect_limit )
        {
	  
	  std::cout << "\nOdbicie nr " << idx+1 << " | R=" << glint_radius2[idx] << std::endl << min_glint_area << "<P<"<< max_glint_area<<" = " << area << " | P_k=" << 3.14*glint_radius2[idx]*glint_radius2[idx] << " | P/P_k="<< area/(3.14*glint_radius2[idx]*glint_radius2[idx]) << std::endl << "Warunek1 (1-P/P_k<"<< glint_area_limit <<"): " << std::abs(1-(area/(3.14*glint_radius2[idx]*glint_radius2[idx]))) << std::endl << "height_rect=" << glint_boundRect[idx].height << " | width_rect=" << glint_boundRect[idx].width << " | h/w=" << 1.*glint_boundRect[idx].height/glint_boundRect[idx].width << std::endl << "Warunek2 (1-h/w<"<<glint_rect_limit<<"): " << std::abs(1-1.*glint_boundRect[idx].height/glint_boundRect[idx].width) << std::endl;

	  drawContours(glint_detection, glint_contours, idx, glint_color, 2);
	  
	  glint_x[j]=glint_mu.m10/glint_mu.m00;
	  glint_y[j]=glint_mu.m01/glint_mu.m00;

	  line(glint_detection, Point(glint_x[j]-lenght, glint_y[j]-lenght), Point(glint_x[j]+lenght, glint_y[j]+lenght), glint_color, thickness, 8);
	  line(glint_detection, Point(glint_x[j]-lenght, glint_y[j]+lenght), Point(glint_x[j]+lenght, glint_y[j]-lenght), glint_color, thickness, 8);
	  j++;
        }
    }
  
  // Finding distances and making decision

  distances_image = img.clone();
  decision_image = img.clone();

  Scalar distance_color(255, 255, 0);
  Scalar decision_color;

  double decision_threshold = 20;
  double analizing_threshold = 50;

  double min_distance = analizing_threshold;
  double min_pupil_x, min_pupil_y, min_glint_x, min_glint_y;

  if ( i!=0 && j!=0 )
    {
      for(int idx3=0; idx3<i; idx3++)
	{
 
	  for(int idx4=0; idx4<j; idx4++)
	    {
	      distance[idx4] = sqrt( (glint_x[idx4]-pupil_x[idx3])*(glint_x[idx4]-pupil_x[idx3]) + (glint_y[idx4]-pupil_y[idx3])*(glint_y[idx4]-pupil_y[idx3]));
	  
	      if ( min_distance >= distance[idx4])
		{
		  min_distance = distance[idx4];
		  min_pupil_x = pupil_x[idx3];
		  min_pupil_y = pupil_y[idx3];
		  min_glint_x = glint_x[idx4];
		  min_glint_y = glint_y[idx4];
		}

	      line( distances_image, Point( glint_x[idx4] - lenght2, glint_y[idx4] - lenght2 ), Point( glint_x[idx4] +lenght2, glint_y[idx4] + lenght2 ), distance_color, 1.5, 8);
	      line( distances_image, Point( glint_x[idx4] - lenght2, glint_y[idx4] + lenght2 ), Point( glint_x[idx4] + lenght2, glint_y[idx4] - lenght2 ), distance_color, 1.5, 8);
	      line( distances_image, Point( glint_x[idx4], glint_y[idx4]), Point(pupil_x[idx3], pupil_y[idx3]), distance_color, 1.5 );

	      std::string text = boost::lexical_cast<std::string>(round(distance[idx4]));

	      if (glint_x[idx4]>pupil_x[idx3] && glint_y[idx4]>pupil_y[idx3])
		{
		  putText( distances_image, text, Point( glint_x[idx4]-std::abs(glint_x[idx4]-pupil_x[idx3])/2., glint_y[idx4]-std::abs(glint_y[idx4]-pupil_y[idx3])/2. ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 1 );
		}

	      else if (glint_x[idx4]>pupil_x[idx3] && glint_y[idx4]<pupil_y[idx3])
		{
		  putText( distances_image, text, Point( glint_x[idx4]-std::abs(glint_x[idx4]-pupil_x[idx3])/2., glint_y[idx4]+std::abs(glint_y[idx4]-pupil_y[idx3])/2. ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 1 );	      
		}

	      else if (glint_x[idx4]<pupil_x[idx3] && glint_y[idx4]<pupil_y[idx3])
		{
		  putText( distances_image, text, Point( glint_x[idx4]+std::abs(glint_x[idx4]-pupil_x[idx3])/2., glint_y[idx4]+std::abs(glint_y[idx4]-pupil_y[idx3])/2. ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 1 );	      
		}

	      else
		{
		  putText( distances_image, text, Point( glint_x[idx4]+std::abs(glint_x[idx4]-pupil_x[idx3])/2., glint_y[idx4]-std::abs(glint_y[idx4]-pupil_y[idx3])/2. ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 1 );
		}
	    }
	}
  
      std::string text = boost::lexical_cast<std::string>(round(min_distance));
      std::string analizing = "Analizing threshold = "+boost::lexical_cast<std::string>(round(analizing_threshold));
      std::string decision = "Decision threshold = "+boost::lexical_cast<std::string>(round(decision_threshold));

      if (min_distance < analizing_threshold)
	{
	  if (min_distance<=decision_threshold)
	    {
	      decision_color = Scalar(0, 255, 0);
	    }
	  
	  else
	    {
	      decision_color = Scalar(0, 0, 255);
	    }
  
	  line( decision_image, Point( min_glint_x - lenght2, min_glint_y - lenght2 ), Point( min_glint_x +lenght2, min_glint_y + lenght2 ), decision_color, 2, 8);
	  line( decision_image, Point( min_glint_x - lenght2, min_glint_y + lenght2 ), Point( min_glint_x + lenght2, min_glint_y - lenght2 ), decision_color, 2, 8);
	  line( decision_image, Point( min_glint_x, min_glint_y), Point(min_pupil_x, min_pupil_y), decision_color, 2 );
  	  
	  if (min_glint_x>min_pupil_x && min_glint_y>min_pupil_y)
	    {
	      putText( decision_image, text, Point( min_glint_x-std::abs(min_glint_x-min_pupil_x)/2., min_glint_y-std::abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1.5, decision_color, 4 );
	    }
	  else if (min_glint_x>min_pupil_x && min_glint_y<min_pupil_y)
	    {
	      putText( decision_image, text, Point( min_glint_x-std::abs(min_glint_x-min_pupil_x)/2., min_glint_y+std::abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1.5, decision_color, 4 );  
	    }

	  else if (min_glint_x<min_pupil_x && min_glint_y<min_pupil_y)
	    {
	      putText( decision_image, text, Point( min_glint_x+std::abs(min_glint_x-min_pupil_x)/2., min_glint_y+std::abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1.5, decision_color, 4 );
	    }

	  else
	    {
	      putText( decision_image, text, Point( min_glint_x+std::abs(min_glint_x-min_pupil_x)/2., min_glint_y-std::abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1.5, decision_color, 4 );
	    }
	  
	}
      
      putText( decision_image, analizing, Point( x_size-60, y_border ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 4 );
      putText( decision_image, decision, Point( x_size-60, 2*y_border ), FONT_HERSHEY_SIMPLEX, 1, distance_color, 4 );

    }

  namedWindow("Original Image", CV_WINDOW_NORMAL);
  cvMoveWindow("Original Image", x_border, y_border+y_size/2.);
  imshow("Original Image", img);
  cvResizeWindow("Original Image", x_size, y_size);
  
  namedWindow("Gray + Gaussian kernel(7,7)", CV_WINDOW_NORMAL);
  cvMoveWindow("Gray + Gaussian kernel(7,7)", x_border+x_size+x_border, y_border+y_size/2.);
  imshow("Gray + Gaussian kernel(7,7)", filtered_image);
  cvResizeWindow("Gray + Gaussian kernel(7,7)", x_size, y_size);
  
  namedWindow("Pupil segmentation", CV_WINDOW_NORMAL);
  cvMoveWindow("Pupil segmentation", x_border+x_size+x_border+x_size+x_border, y_border);
  imshow("Pupil segmentation", thresh_pupil_image);
  cvResizeWindow("Pupil segmentation", x_size, y_size);
  
  namedWindow("Glint segmentation", CV_WINDOW_NORMAL);
  cvMoveWindow("Glint segmentation", x_border+x_size+x_border+x_size+x_border, y_border+y_size+y_border);
  imshow("Glint segmentation", thresh_glint_image);
  cvResizeWindow("Glint segmentation", x_size, y_size);
  
  namedWindow("Pupil - contours", CV_WINDOW_NORMAL);
  cvMoveWindow("Pupil - contours", x_border+x_size+x_border+x_size+x_border+x_size+x_border, y_border);
  imshow("Pupil - contours", contours_pupil_image);
  cvResizeWindow("Pupil - contours", x_size, y_size);
  
  namedWindow("Glint - contours", CV_WINDOW_NORMAL);
  cvMoveWindow("Glint - contours", x_border+x_size+x_border+x_size+x_border+x_size+x_border, y_border+y_size+y_border);
  imshow("Glint - contours", contours_glint_image);
  cvResizeWindow("Glint - contours", x_size, y_size);
  
  pressed_button=waitKey( 0 );

  if (pressed_button == 'q')
    return 0;
  
  cvDestroyWindow("Original Image");
  cvDestroyWindow("Gray + Gaussian kernel(7,7)");
  cvDestroyWindow("Pupil segmentation");
  cvDestroyWindow("Glint segmentation");
  cvDestroyWindow("Pupil - contours");
  cvDestroyWindow("Glint - contours");
  
  namedWindow("Pupil - contours", CV_WINDOW_NORMAL);
  cvMoveWindow("Pupil - contours", x_border, y_border);
  imshow("Pupil - contours", contours_pupil_image_BGR);
  cvResizeWindow("Pupil - contours", x_size, y_size);
  
  namedWindow("Glint - contours", CV_WINDOW_NORMAL);
  cvMoveWindow("Glint - contours", x_border, y_border+y_size+y_border);
  imshow("Glint - contours", contours_glint_image_BGR);
  cvResizeWindow("Glint - contours", x_size, y_size);

  namedWindow("Pupil detection", CV_WINDOW_NORMAL);
  cvMoveWindow("Pupil detection", x_border+x_size+x_border, y_border);
  imshow("Pupil detection", pupil_detection);
  cvResizeWindow("Pupil detection", x_size, y_size);
  
  namedWindow("Glint detection", CV_WINDOW_NORMAL);
  cvMoveWindow("Glint detection", x_border+x_size+x_border, y_border+y_size+y_border);
  imshow("Glint detection", glint_detection);
  cvResizeWindow("Glint detection", x_size, y_size);
  
  namedWindow("All distances", CV_WINDOW_NORMAL);
  cvMoveWindow("All distances", x_border+x_size+x_border+x_size+x_border, y_border+y_size/2.);
  imshow("All distances", distances_image);
  cvResizeWindow("All distances", x_size, y_size);

  namedWindow("Minimal distance", CV_WINDOW_NORMAL);
  cvMoveWindow("Minimal distance", x_border+x_size+x_border+x_size+x_border+x_size+x_border, y_border+y_size/2.);
  imshow("Minimal distance", decision_image);
  cvResizeWindow("Minimal distance", x_size, y_size);
  
  pressed_button=waitKey( 0 );

  if (pressed_button == 'q')
    return 0;

  cvDestroyWindow("Pupil - contours");
  cvDestroyWindow("Glint - contours");
  cvDestroyWindow("Pupil detection");
  cvDestroyWindow("Glint detection");
  cvDestroyWindow("All distances");
  cvDestroyWindow("Minimal distance");

  Scalar circle_color(255, 0, 255);
  Scalar rect_color(255, 255, 0);

  vector<vector<Point> > pupil_contours_poly2( pupil_contours.size() );
  vector<Rect> pupil_boundRect2( pupil_contours.size() );
  vector<Point2f>pupil_center2( pupil_contours.size() );
  vector<float>pupil_radius3( pupil_contours.size() );

  for(int idx=0; idx<pupil_contours.size(); idx++)
    {

      Mat marked_pupil_img=img.clone();

      pupil_mu = moments(pupil_contours[idx], false);
      area = pupil_mu.m00;

      cvtColor(contours_pupil_image, pupil_detection, COLOR_GRAY2BGR);

      approxPolyDP( Mat(pupil_contours[idx]), pupil_contours_poly2[idx], 3, true );
      pupil_boundRect2[idx] = boundingRect( Mat(pupil_contours_poly2[idx]) );
      minEnclosingCircle( (Mat)pupil_contours_poly2[idx], pupil_center2[idx], pupil_radius3[idx] );

      std::cout << std::endl;
      std::cout << "###############################  " << idx+1 << "  #############################"<<std::endl;

      if (area > min_pupil_area && area < max_pupil_area && std::abs(1-(area/(3.14*pupil_radius3[idx]*pupil_radius3[idx]))) < pupil_area_limit &&  std::abs(1-1.*pupil_boundRect2[idx].height/pupil_boundRect2[idx].width)< pupil_rect_limit)
  	std::cout << "OK -- ";
      else
  	std::cout << " False -- ";
      
      if (area > min_pupil_area)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(area < max_pupil_area)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(std::abs(1-(area/(3.14*pupil_radius3[idx]*pupil_radius3[idx]))) < pupil_area_limit)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(std::abs(1-1.*pupil_boundRect2[idx].height/pupil_boundRect2[idx].width)< pupil_rect_limit)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";

      std::cout << std::endl;

      if (dilation_flag == true)
	std::cout << "\nPRZYŁOŻONO DYLACJĘ ("<< nr_dilation_applied <<" razy)"<<std::endl;
      else
	std::cout<<std::endl;
      
      std::cout << "Źrenica nr " << idx+1 << std::endl;
      std::cout << "Warunek 1: P="<< area << " >" << min_pupil_area<< std::endl;
      if (area > min_pupil_area)
  	std::cout << " OK "<< std::endl;
      else
  	std::cout << " False "<< std::endl;

      std::cout << "Warunek 2: P="<< area <<" <"<< max_pupil_area << std::endl;
      if (area < max_pupil_area)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;

      std::cout<<"R=" << pupil_radius3[idx] << " | P_k=" << 3.14*pupil_radius3[idx]*pupil_radius3[idx] << " | P/P_k="<< area/(3.14*pupil_radius3[idx]*pupil_radius3[idx]) << std::endl;

      std::cout << "Warunek3: |1-P/P_k|=" << std::abs(1-(area/(3.14*pupil_radius3[idx]*pupil_radius3[idx]))) << " <" << pupil_area_limit <<std::endl;
      if (std::abs(1-(area/(3.14*pupil_radius3[idx]*pupil_radius3[idx]))) < pupil_area_limit)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;
      
      std::cout<< "height_rect=" << pupil_boundRect2[idx].height << " | width_rect=" << pupil_boundRect2[idx].width << " | h/w=" << 1.*pupil_boundRect2[idx].height/pupil_boundRect2[idx].width << std::endl;
      
      std::cout<<"Warunek4: |1-h/w|=" << std::abs(1-1.*pupil_boundRect2[idx].height/pupil_boundRect2[idx].width) << " <" << pupil_rect_limit << std::endl;
      if ( std::abs(1-1.*pupil_boundRect2[idx].height/pupil_boundRect2[idx].width)< pupil_rect_limit)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;

      rectangle( pupil_detection, pupil_boundRect2[idx].tl(), pupil_boundRect2[idx].br(), rect_color, 2, 8, 0 );
      drawContours(pupil_detection, pupil_contours, idx, pupil_color, 2);
      circle( pupil_detection, pupil_center2[idx], (int)pupil_radius3[idx], circle_color, 2, 8, 0 );

      pupil_x[idx]=pupil_mu.m10/area;
      pupil_y[idx]=pupil_mu.m01/area;
	  
      line(pupil_detection, Point(pupil_x[idx]-5, pupil_y[idx]-5), Point(pupil_x[idx]+5,pupil_y[idx]+5), pupil_color, 2, 8);
      line(pupil_detection, Point(pupil_x[idx]-5, pupil_y[idx]+5), Point(pupil_x[idx]+5,pupil_y[idx]-5), pupil_color, 2, 8);

      // rectangle( marked_pupil_img, pupil_boundRect2[idx].tl(), pupil_boundRect2[idx].br(), rect_color, 2, 8, 0 );
      drawContours(marked_pupil_img, pupil_contours, idx, pupil_color, 3);
      // circle( marked_pupil_img, pupil_center2[idx], (int)pupil_radius3[idx], circle_color, 2, 8, 0 );

      namedWindow("Pupil", CV_WINDOW_NORMAL);
      cvMoveWindow("Pupil", x_border, y_border);
      imshow("Pupil", pupil_detection);
      cvResizeWindow("Pupil", 2*x_size, 2*y_size);

      namedWindow("Origin image", CV_WINDOW_NORMAL);
      cvMoveWindow("Origin image", x_border+x_size+x_border, 2*y_size + 5*y_border);
      imshow("Origin image", img);
      cvResizeWindow("Origin image", x_size, y_size);

      namedWindow("Marked image", CV_WINDOW_NORMAL);
      cvMoveWindow("Marked image", x_border, 2*y_size +5*y_border);
      imshow("Marked image", marked_pupil_img);
      cvResizeWindow("Marked image", x_size, y_size);
	  
      pressed_button=waitKey( 0 );

      if (pressed_button == 'q')
	return 0;

      cvDestroyWindow("Pupil");
      cvDestroyWindow("Origin image");
      cvDestroyWindow("Marked image");
      
      pressed_button=waitKey( 0 );
       
      if (pressed_button == 'q')
	return 0;
	  
      cvDestroyWindow("Pupil");
      cvDestroyWindow("Origin image");
      cvDestroyWindow("Marked image");

    }    

  vector<vector<Point> > glint_contours_poly2( glint_contours.size() );
  vector<Rect> glint_boundRect2( glint_contours.size() );
  vector<Point2f> glint_center2( glint_contours.size() );
  vector<float> glint_radius3( glint_contours.size() );
  
  for(int idx=0; idx< glint_contours.size(); idx++)
    {

      Mat marked_glint_img=img.clone();

      glint_mu = moments(glint_contours[idx], false);
      area = glint_mu.m00;

      cvtColor(contours_glint_image, glint_detection, COLOR_GRAY2BGR);

      approxPolyDP( Mat(glint_contours[idx]), glint_contours_poly2[idx], 3, true );
      glint_boundRect2[idx] = boundingRect( Mat(glint_contours_poly2[idx]) );
      minEnclosingCircle( (Mat)glint_contours_poly2[idx], glint_center2[idx], glint_radius3[idx] );

      std::cout << std::endl;
      std::cout << "###############################  " << idx+1 << "  #############################"<<std::endl;
      
      if (area > min_glint_area && area < max_glint_area && std::abs(1-(area/(3.14*glint_radius3[idx]*glint_radius3[idx]))) < glint_area_limit &&  std::abs(1-1.*glint_boundRect2[idx].height/glint_boundRect2[idx].width)< glint_rect_limit)
  	std::cout << "OK -- ";
      else
  	std::cout << " False -- ";
      
      if (area > min_glint_area)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(area < max_glint_area)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(std::abs(1-(area/(3.14*glint_radius3[idx]*glint_radius3[idx]))) < glint_area_limit)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";
      if(std::abs(1-1.*glint_boundRect2[idx].height/glint_boundRect2[idx].width)< glint_rect_limit)
  	std::cout <<"OK ";
      else
  	std::cout << "False ";

      std::cout << std::endl;

      std::cout << "\nOdbicie nr " << idx+1 << std::endl;
      std::cout << "Warunek 1: P="<< area << " >" << min_glint_area<< std::endl;
      if (area > min_glint_area)
  	std::cout << " OK "<< std::endl;
      else
  	std::cout << " False "<< std::endl;

      std::cout << "Warunek 2: P="<< area <<" <"<< max_glint_area << std::endl;
      if (area < max_glint_area)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;

      std::cout<<"R=" << glint_radius3[idx] << " | P_k=" << 3.14*glint_radius3[idx]*glint_radius3[idx] << " | P/P_k="<< area/(3.14*glint_radius3[idx]*glint_radius3[idx]) << std::endl;

      std::cout << "Warunek3: |1-P/P_k|=" << std::abs(1-(area/(3.14*glint_radius3[idx]*glint_radius3[idx]))) << " <" << glint_area_limit <<std::endl;
      if (std::abs(1-(area/(3.14*glint_radius3[idx]*glint_radius3[idx]))) < glint_area_limit)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;
      
      std::cout<< "height_rect=" << glint_boundRect2[idx].height << " | width_rect=" << glint_boundRect2[idx].width << " | h/w=" << 1.*glint_boundRect2[idx].height/glint_boundRect2[idx].width << std::endl;
      
      std::cout<<"Warunek4: |1-h/w|=" << std::abs(1-1.*glint_boundRect2[idx].height/glint_boundRect2[idx].width) << " <" << glint_rect_limit << std::endl;
      if ( std::abs(1-1.*glint_boundRect2[idx].height/glint_boundRect2[idx].width)< glint_rect_limit)
  	std::cout << " OK " << std::endl;
      else
  	std::cout << " False " << std::endl;

      rectangle( glint_detection, glint_boundRect2[idx].tl(), glint_boundRect2[idx].br(), rect_color, 2, 8, 0 );
      drawContours(glint_detection, glint_contours, idx, glint_color, 2);
      circle( glint_detection, glint_center2[idx], (int)glint_radius3[idx], circle_color, 2, 8, 0 );

      // rectangle( marked_glint_img, glint_boundRect2[idx].tl(), glint_boundRect2[idx].br(), rect_color, 2, 8, 0 );
      drawContours(marked_glint_img, glint_contours, idx, glint_color, 3);
      // circle( marked_glint_img, glint_center2[idx], (int)glint_radius3[idx], circle_color, 2, 8, 0 );

      namedWindow("Glint", CV_WINDOW_NORMAL);
      cvMoveWindow("Glint", x_border, 0);
      imshow("Glint", glint_detection);
      cvResizeWindow("Glint", 2*x_size, 2*y_size);

      namedWindow("Origin image", CV_WINDOW_NORMAL);
      cvMoveWindow("Origin image", x_border+x_size+x_border, 2*y_size + 5*y_border);
      imshow("Origin image", img);
      cvResizeWindow("Origin image", x_size, y_size);

      namedWindow("Marked image", CV_WINDOW_NORMAL);
      cvMoveWindow("Marked image", x_border, 2*y_size +5*y_border);
      imshow("Marked image", marked_glint_img);
      cvResizeWindow("Marked image", x_size, y_size);
	  
      pressed_button=waitKey( 0 );

      if (pressed_button == 'q')
	return 0;

      cvDestroyWindow("Glint");
      cvDestroyWindow("Origin image");
      cvDestroyWindow("Marked image");
      
    }
  
  return 0;
}
