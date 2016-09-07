/** 
    Script visualizes the principle of operation of the EyeSwitch program.
    Colors meaning: red - too far from LED area, orange- buffer fills up, green- buffer full->press, blue- frozen mode after press, 
    cyan- end of frozen mode.
    
    To see dependencies read README file. 
    
    To run type:
    cmake .
    make
    python server.py & ./show (make sure that server.py is not running in the foreground)
    
    To view the steps of image processing run script 'steps' with selected image by typing:
    ./steps <Image_Path>
    
    @author Jerzy Grynczewski
    @version 2.0
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <boost/lexical_cast.hpp>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;

int main( int argc, char** argv )
{
    std::ofstream myfile;
    myfile.open("alerts.txt");

    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    std::cout << "Connecting to etr_switch server..." << std::endl;
    socket.connect ("tcp://localhost:5559");
    
    VideoCapture cap(1);

    if(!cap.isOpened())
    {
        std::cout<<"No camera avaible"<<std::endl;
        return -1;
    }

    Mat img, gray_image, filtered_image, thresh_pupil_image, contours_pupil_image, contours_pupil_image_BGR, thresh_glint_image, contours_glint_image, contours_glint_image_BGR, pupil_detection, glint_detection, distances_image, decision_image;
    char pressed_key = ' ';

    int buffor=0, full_buffor=5, buffor_inertia=4, count=0, count_max = 10, press_count = 0, press_count_max = 20; //press_count tells how many snapshots pass since the last press.
 
    bool press_flag=false;
    int k=0;
    
    std::stringstream ss;
    string name = "cropped_";
    string type = ".jpg";

    for(;;)
    {
        Mat frame;
        cap >> frame;
        flip(frame, img, 1);
        cvtColor(img, gray_image, COLOR_BGR2GRAY);
        GaussianBlur(gray_image, filtered_image, Size(7,7), 0, 0);

        // Pupil contours

        int pupil_thresh = 50;
        threshold(filtered_image, thresh_pupil_image, pupil_thresh, 255, THRESH_BINARY_INV);

        Scalar pupil_color( 255, 0, 0 );
        Scalar pupil_contours_color( 0, 255, 255 );
        vector<vector<Point> > pupil_contours;
        vector<Vec4i> pupil_hierarchy;

        contours_pupil_image = thresh_pupil_image.clone();
        findContours(contours_pupil_image, pupil_contours, pupil_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Pupil detection

        int i = 0;

        Moments pupil_mu;
        std::vector<int> pupil_x(pupil_contours.size());
        std::vector<int> pupil_y(pupil_contours.size());
        std::vector<int> pupil_radius(pupil_contours.size());
        double area, pupil_r;
	Rect pupil_bound;
  
	bool dilation_flag = false;

	vector<vector<Point> > pupil_contours_poly( pupil_contours.size() );
	vector<Rect> pupil_boundRect( pupil_contours.size() );
	vector<Point2f>pupil_center( pupil_contours.size() );
	vector<float>pupil_radius2( pupil_contours.size() );

	int nr_dilation_applied = 5;
	double pupil_area_limit=0.5, pupil_rect_limit=0.45, min_pupil_area=700, max_pupil_area=8000;

        for(int idx=0; idx< pupil_contours.size(); idx++)
	  {
            pupil_mu = moments(pupil_contours[idx], false);
            area = pupil_mu.m00;

	    approxPolyDP( Mat(pupil_contours[idx]), pupil_contours_poly[idx], 3, true );
	    pupil_boundRect[idx] = boundingRect( Mat(pupil_contours_poly[idx]) );
	    minEnclosingCircle( (Mat)pupil_contours_poly[idx], pupil_center[idx], pupil_radius2[idx] );
      
	    if (area>min_pupil_area && area<max_pupil_area && std::abs(1-(area/(3.14*pupil_radius2[idx]*pupil_radius2[idx])))<pupil_area_limit && std::abs(1-1.*pupil_boundRect[idx].height/pupil_boundRect[idx].width)<pupil_rect_limit)
	      {
                pupil_x[i]=pupil_mu.m10/area;
                pupil_y[i]=pupil_mu.m01/area;
                pupil_radius[i]=sqrt(area/M_PI);
		
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

		    pupil_x[i]=pupil_mu.m10/area;
		    pupil_y[i]=pupil_mu.m01/area;
		    pupil_radius[i]=sqrt(area/M_PI);
	  
		    i++;
		  }
	      }    
	    
	  }

	int glint_thresh = 100; //empirically chosen (with reserve)
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

	    if ( area>min_glint_area && area<max_glint_area && std::abs(1-(area/(3.14*glint_radius2[idx]*glint_radius2[idx])))<glint_area_limit && std::abs(1-1.*glint_boundRect[idx].height/glint_boundRect[idx].width)<glint_rect_limit )
	      {
	  
		glint_x[j]=glint_mu.m10/glint_mu.m00;
		glint_y[j]=glint_mu.m01/glint_mu.m00;

		j++;
	      }
	  }

        decision_image = img.clone();

        Scalar decision_color, press_color = Scalar(0, 255, 0);
	
        double analizing_threshold = 50;
        double decision_threshold = 20;

        double min_distance = analizing_threshold;
        double min_pupil_x, min_pupil_y, min_glint_x, min_glint_y;

        //Finding minimum distance and making decision
        double press_min_distance, press_min_pupil_x, press_min_pupil_y, press_min_glint_x, press_min_glint_y;

        if ( i!=0 && j!=0 )//at least one candidate for pupil and for glints
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
                }
            }

            if (min_distance<analizing_threshold) //if user looks at analizing area
	      {

		if (press_flag == false) //if pass enough time after previous switch or hasn't been any switch so far
		  {
		    
		    if (min_distance<=decision_threshold && buffor==full_buffor) //if user looks at LED area and buffer is full
		      {

			decision_color = Scalar(0, 255, 0); //green
		    
			zmq::message_t request (6);
			memcpy ((void *) request.data (), "press", 5);
			socket.send (request);
			
			//  Get the reply.
			zmq::message_t reply;
			socket.recv (&reply);
			std::cout << "Received ok" << std::endl;
			buffor = 0;

			press_min_distance = min_distance;
			press_min_pupil_x = min_pupil_x;
			press_min_pupil_y = min_pupil_y;
			press_min_glint_x = min_glint_x;
			press_min_glint_y = min_glint_y;
			
			press_flag = true;
			
		      }

		    else if (min_distance<=decision_threshold) // if user looks on LED and buffer isn't full.
		      {
			++buffor;
			decision_color = Scalar(0, 125, 255); //orange
		      }
		    
		    else // if user doesn't look on the LED area
		      {
			decision_color = Scalar(0, 0, 255); //red
			++count; // this variable is bound with buffor inertia. Buffor inertia tells when program should stop count and zero itself - after how many snapshots.
		      }
		  }

		else if (press_flag == true) //if just has switched
		  {
		    if (press_count < press_count_max) // and pass not enough time (frozen state)
		      {

			decision_color = Scalar(255, 0, 0);

			line( decision_image, Point( press_min_glint_x - lenght2, press_min_glint_y - lenght2 ), Point( press_min_glint_x +lenght2, press_min_glint_y + lenght2 ), press_color, 1.5, 8);
			line( decision_image, Point( press_min_glint_x - lenght2, press_min_glint_y + lenght2 ), Point( press_min_glint_x + lenght2, press_min_glint_y - lenght2 ), press_color, 1.5, 8);
			line( decision_image, Point( press_min_glint_x, press_min_glint_y), Point(press_min_pupil_x, press_min_pupil_y), press_color, 1.5 );
			
			std::string text = boost::lexical_cast<std::string>(round(press_min_distance));
			
			if (press_min_glint_x>press_min_pupil_x && press_min_glint_y>press_min_pupil_y)
			  {
			    putText( decision_image, text, Point( press_min_glint_x-abs(press_min_glint_x-press_min_pupil_x)/2., press_min_glint_y-abs(press_min_glint_y-press_min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, press_color, 4 );
			  }

			else if (press_min_glint_x>press_min_pupil_x && press_min_glint_y<press_min_pupil_y)
			  {
			    putText( decision_image, text, Point( press_min_glint_x-abs(press_min_glint_x-press_min_pupil_x)/2., press_min_glint_y+abs(press_min_glint_y-press_min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, press_color, 4 );
			  }

			else if (press_min_glint_x<press_min_pupil_x && press_min_glint_y<press_min_pupil_y)
			  {
			    putText( decision_image, text, Point( press_min_glint_x+abs(press_min_glint_x-press_min_pupil_x)/2., press_min_glint_y+abs(press_min_glint_y-press_min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, press_color, 4 );
			  }
			
			else
			  {
			    putText( decision_image, text, Point( press_min_glint_x+abs(press_min_glint_x-press_min_pupil_x)/2., press_min_glint_y-abs(press_min_glint_y-press_min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, press_color, 4 );
			  }

			++press_count;
		      }

		    else //(pass enough time)
		      {
			decision_color = Scalar(255, 255, 0);
			press_count = 0;
			press_flag = false;
		      }
		  }
		
		if (count > buffor_inertia)
		  {
                    buffor=0;
                    count=0;
		  }
		
                line( decision_image, Point( min_glint_x - lenght2, min_glint_y - lenght2 ), Point( min_glint_x +lenght2, min_glint_y + lenght2 ), decision_color, 1.5, 8);
                line( decision_image, Point( min_glint_x - lenght2, min_glint_y + lenght2 ), Point( min_glint_x + lenght2, min_glint_y - lenght2 ), decision_color, 1.5, 8);
                line( decision_image, Point( min_glint_x, min_glint_y), Point(min_pupil_x, min_pupil_y), decision_color, 1.5 );
		
                std::string text = boost::lexical_cast<std::string>(round(min_distance));
		
                if (min_glint_x>min_pupil_x && min_glint_y>min_pupil_y)
		  {
		    putText( decision_image, text, Point( min_glint_x-abs(min_glint_x-min_pupil_x)/2., min_glint_y-abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, decision_color, 4 );
		  }

                else if (min_glint_x>min_pupil_x && min_glint_y<min_pupil_y)
		  {
                    putText( decision_image, text, Point( min_glint_x-abs(min_glint_x-min_pupil_x)/2., min_glint_y+abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, decision_color, 4 );
		  }
		
                else if (min_glint_x<min_pupil_x && min_glint_y<min_pupil_y)
		  {
                    putText( decision_image, text, Point( min_glint_x+abs(min_glint_x-min_pupil_x)/2., min_glint_y+abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, decision_color, 4 );
		  }
		
                else
		  {
                    putText( decision_image, text, Point( min_glint_x+abs(min_glint_x-min_pupil_x)/2., min_glint_y-abs(min_glint_y-min_pupil_y)/2. ), FONT_HERSHEY_SIMPLEX, 1, decision_color, 4 );
		  }
	      }

	    else if (count > buffor_inertia) 
	      {
		count = 0;
		buffor = 0;
		press_flag = false;
		press_count = 0;
	      }

	    else ++count; //looking outside the analizing area shouldn't frozen the time, otherwise user would back with the same state as he was left and yet by this time user was looking on sth else.
        }

	else if (count > buffor_inertia)
	  {

	    count = 0;
	    buffor = 0;
	    press_flag = false;
	    press_count = 0;
	  }

	else ++count; // even if nothing found time shouldn't be frozen.

        int x_border = 50, y_border = 2*x_border, x_size = 2*300, y_size = 2*225;	
	
        namedWindow("Original Image", CV_WINDOW_NORMAL);
        cvMoveWindow("Original Image", x_border, y_border+y_size/10.);
        imshow("Original Image", img);
        cvResizeWindow("Original Image", x_size, y_size);

        namedWindow("Decision", CV_WINDOW_NORMAL);
        cvMoveWindow("Decision", x_border+x_size+x_border, y_border+y_size/10.);
        imshow("Decision", decision_image);
        cvResizeWindow("Decision", x_size, y_size);
	
	ss << name << k << type;
	string filename = ss.str();
	ss.str("");
	
	imwrite(filename, decision_image);
	k++;
	pressed_key = waitKey(30);
	if(pressed_key == 'z') myfile << k << std::endl;    

        if(pressed_key == 'q'){
	  myfile.close();
	  break; 
	}

    }
    
    return 0;
}
