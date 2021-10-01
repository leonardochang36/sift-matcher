//
//  test_sift_matcher.cpp
//  xcode_proj
//
//  Created by LChang on 5/13/18.
//  Copyright © 2018 AAA. All rights reserved.
//

#include "test_sift_matcher.h"

using namespace std;
using namespace cv;


static void help_sift_matcher()
{
    cout << "\nTest app to show the SIFT extraction and matching.\n";
    
    cout << "Hot keys: \n"
    "\tnone" << endl;
}

const char* keys_sift_matcher =
{
    "{input |/Users/AAA/Desktop/CursoRPyMD/tec_notebook.mov| Video capture url}"
    "{template |/Users/AAA/Desktop/CursoRPyMD/tec_notebook_.jpg| Template image url}"
};

int test_sift_matcher(int argc, const char * argv[])
{
    help_sift_matcher();
    
    ////////////////// Reading arguments ///////////////////
    CommandLineParser parser(argc, argv, keys_sift_matcher);
    std::string inputf = parser.get<std::string>("input");
    std::string templatef = parser.get<std::string>("template");
    
    //-- Load template and video input
    Mat tmpl = imread(templatef);
    VideoCapture vin(inputf);
    
    //-- Create SIFT extractor and descriptor
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    
    //-- Step 1: Detect the keypoints in template:
    std::vector<KeyPoint> keypoints_1;
    f2d->detect( tmpl, keypoints_1 );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1;
    f2d->compute( tmpl, keypoints_1, descriptors_1 );
    
    //-- Process video feed
    Mat frame;
    if (vin.isOpened()) {
        for (;;) {
            
            //-- Read current frame
            vin >> frame;
            
            if (frame.empty())
                break;
            
            resize(frame, frame, Size(), 0.5f, 0.5f);
            
            //-- Mark starting time
            int64 e1 = getTickCount();
            
            //-- Step 1: Detect the keypoints
            std::vector<KeyPoint> keypoints_2;
            f2d->detect( frame, keypoints_2 );
            
            //-- Step 2: Calculate descriptors (feature vectors)
            Mat descriptors_2;
            f2d->compute( frame, keypoints_2, descriptors_2 );
            
            //-- Step 3: Matching descriptor vectors using BFMatcher :
            BFMatcher matcher;
            std::vector< DMatch > matches;
            matcher.match( descriptors_1, descriptors_2, matches );
            
            //-- Step 4: Prune matches, retain "good" matches (i.e. whose distance is
            //-- less than 2*min_dist, or a small arbitrary value ( 0.02 ) in the
            //-- event that min_dist is very small)
            
            // Quick calculation of max and min distances between keypoints
            double max_dist = 0; double min_dist = 100;
            for( int i = 0; i < descriptors_1.rows; i++ )
            {
                double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }
            
            // Select only "good" matches
            std::vector< DMatch > good_matches;
            for( int i = 0; i < descriptors_1.rows; i++ )
            {
                if( matches[i].distance <= max(2*min_dist, 0.02) )
                {
                    good_matches.push_back( matches[i]);
                }
            }
            
            //-- Draw visual feedback
            Mat img_matches;
            drawMatches( tmpl, keypoints_1, frame, keypoints_2,
                        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            
            double e2 =  (double)getTickCount();
            double time = (e2 - e1)/ (double)getTickFrequency();
            
            cv::putText(img_matches, "FPS: " + std::to_string(1/time), Point(10, img_matches.rows - 20), FONT_HERSHEY_TRIPLEX, 1, CV_RGB(255, 255, 0), 1, 8);
            
            imshow( "Good Matches", img_matches );
//            cout << "\rFPS: " << 1/time;
            
            int key = waitKey(5);
            if (key == 27)
                break;
        }
    }
    

    return 0;
}
