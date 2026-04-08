#include "kf_tracker/CKalmanFilter.h"
#include "kf_tracker/featureDetection.h"
#include "opencv2/video/tracking.hpp"
#include "pcl_ros/point_cloud.h"
#include <algorithm>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cstdio>
#include <boost/make_shared.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <string.h>

#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>

#include <limits>
#include <utility>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace cv;

// Maximum number of tracked obstacles
const int NUM_FILTERS = 10;

ros::Publisher objID_pub;

// KF init
int stateDim = 4; // [x,y,v_x,v_y]
int measDim = 2;  // [z_x,z_y]
int ctrlDim = 0;

cv::KalmanFilter KF[NUM_FILTERS];
ros::Publisher pub_cluster[NUM_FILTERS];
ros::Publisher tracked_objects_pub;
ros::Publisher markerPub;

std::vector<geometry_msgs::Point> prevClusterCenters;

cv::Mat state(stateDim, 1, CV_32F);
cv::Mat_<float> measurement(2, 1);

std::vector<int> objID;

bool firstFrame = true;

// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point &p1, geometry_msgs::Point &p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
}

// Forward declarations
void publish_tracked_objects();
void KFT(const std_msgs::Float32MultiArray ccs);

std::pair<int, int> findIndexOfMin(std::vector<std::vector<float>> distMat) {
  // cout << "findIndexOfMin cALLED\n";
  std::pair<int, int> minIndex;
  float minEl = std::numeric_limits<float>::max();
  // cout << "minEl=" << minEl << "\n";
  for (int i = 0; i < distMat.size(); i++)
    for (int j = 0; j < distMat.at(0).size(); j++) {
      if (distMat[i][j] < minEl) {
        minEl = distMat[i][j];
        minIndex = std::make_pair(i, j);
      }
    }
  // cout << "minIndex=" << minIndex.first << "," << minIndex.second << "\n";
  return minIndex;
}



void KFT(const std_msgs::Float32MultiArray ccs) {

  // First predict, to update the internal statePre variable
  std::vector<cv::Mat> pred;
  for (int i = 0; i < NUM_FILTERS; i++) {
    pred.push_back(KF[i].predict());
  }

  // Get measurements
  std::vector<geometry_msgs::Point> clusterCenters;

  int i = 0;
  for (std::vector<float>::const_iterator it = ccs.data.begin();
       it != ccs.data.end(); it += 3) {
    geometry_msgs::Point pt;
    pt.x = *it;
    pt.y = *(it + 1);
    pt.z = *(it + 2);

    clusterCenters.push_back(pt);
  }

  std::vector<geometry_msgs::Point> KFpredictions;
  i = 0;
  for (auto it = pred.begin(); it != pred.end(); it++) {
    geometry_msgs::Point pt;
    pt.x = (*it).at<float>(0);
    pt.y = (*it).at<float>(1);
    pt.z = (*it).at<float>(2);

    KFpredictions.push_back(pt);
  }

  // Find the cluster that is more probable to be belonging to a given KF.
  objID.clear();
  objID.resize(NUM_FILTERS);

  std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
  std::vector<std::vector<float>> distMat;

  for (int filterN = 0; filterN < NUM_FILTERS; filterN++) {
    std::vector<float> distVec;
    for (int n = 0; n < NUM_FILTERS; n++) {
      distVec.push_back(
          euclidean_distance(KFpredictions[filterN], copyOfClusterCenters[n]));
    }

    distMat.push_back(distVec);
    // cout << "filterN=" << filterN << "\n";
  }

  // cout << "distMat.size()" << distMat.size() << "\n";
  // cout << "distMat[0].size()" << distMat.at(0).size() << "\n";
  // DEBUG: print the distMat
  // for (const auto &row : distMat) {
  //   for (const auto &s : row)
  //     std::cout << s << ' ';
  //   std::cout << std::endl;
  // }

  for (int clusterCount = 0; clusterCount < NUM_FILTERS; clusterCount++) {
    // 1. Find min(distMax)==> (i,j);
    std::pair<int, int> minIndex(findIndexOfMin(distMat));
    // cout << "Received minIndex=" << minIndex.first << "," << minIndex.second
    //      << "\n";
    // 2. objID[i]=clusterCenters[j]; counter++
    objID[minIndex.first] = minIndex.second;

    // 3. distMat[i,:]=10000; distMat[:,j]=10000
    distMat[minIndex.first] =
        std::vector<float>(NUM_FILTERS, 10000.0);
    for (int row = 0; row < distMat.size(); row++) {
      distMat[row][minIndex.second] = 10000.0;
    }
    // cout << "clusterCount=" << clusterCount << "\n";
  }

  visualization_msgs::MarkerArray clusterMarkers;

  for (int i = 0; i < NUM_FILTERS; i++) {
    visualization_msgs::Marker m;
    m.id = i;
    m.header.frame_id = "front_laser";

    // Skip objects with zero velocity
    float vx = KF[i].statePre.at<float>(2);
    float vy = KF[i].statePre.at<float>(3);
    if (vx == 0.0f && vy == 0.0f) {
      m.action = visualization_msgs::Marker::DELETE;
      clusterMarkers.markers.push_back(m);
      continue;
    }

    m.type = visualization_msgs::Marker::CUBE;
    m.scale.x = 0.3;
    m.scale.y = 0.3;
    m.scale.z = 0.3;
    m.action = visualization_msgs::Marker::ADD;
    m.color.a = 1.0;
    m.color.r = i % 2 ? 1 : 0;
    m.color.g = i % 3 ? 1 : 0;
    m.color.b = i % 4 ? 1 : 0;

    geometry_msgs::Point clusterC(KFpredictions[i]);
    m.pose.position.x = clusterC.x;
    m.pose.position.y = clusterC.y;
    m.pose.position.z = clusterC.z;

    clusterMarkers.markers.push_back(m);
  }

  prevClusterCenters = clusterCenters;

  markerPub.publish(clusterMarkers);

  std_msgs::Int32MultiArray obj_id;
  for (auto it = objID.begin(); it != objID.end(); it++)
    obj_id.data.push_back(*it);
  // Publish the object IDs
  objID_pub.publish(obj_id);
  // convert clusterCenters from geometry_msgs::Point to floats
  std::vector<std::vector<float>> cc;
  for (int i = 0; i < NUM_FILTERS; i++) {
    vector<float> pt;
    pt.push_back(clusterCenters[objID[i]].x);
    pt.push_back(clusterCenters[objID[i]].y);
    pt.push_back(clusterCenters[objID[i]].z);

    cc.push_back(pt);
  }

  // The update phase
  for (int i = 0; i < NUM_FILTERS; i++) {
    float meas[2] = {cc[i].at(0), cc[i].at(1)};
    cv::Mat measMat = cv::Mat(2, 1, CV_32F, meas);
    if (!(meas[0] == 0.0f || meas[1] == 0.0f))
      Mat estimated = KF[i].correct(measMat);
  }

  // Publish tracked objects state (coordinates and velocity)
  publish_tracked_objects();
}

void publish_cloud(ros::Publisher &pub,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
  sensor_msgs::PointCloud2::Ptr clustermsg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*cluster, *clustermsg);
  clustermsg->header.frame_id = "front_laser";
  clustermsg->header.stamp = ros::Time::now();
  pub.publish(*clustermsg);
}

// Publish tracked objects' state (x, y, vx, vy for each object)
void publish_tracked_objects() {
  std_msgs::Float32MultiArray state_msg;
  
  visualization_msgs::MarkerArray tracked_markers;
  
  // cout << "\n=== TRACKED OBJECTS ===" << endl;
  
  for (int i = 0; i < NUM_FILTERS; i++) {
    float x  = roundf(KF[i].statePre.at<float>(0) * 100) / 100;
    float y  = roundf(KF[i].statePre.at<float>(1) * 100) / 100;
    float vx = roundf(KF[i].statePre.at<float>(2) * 100) / 100;
    float vy = roundf(KF[i].statePre.at<float>(3) * 100) / 100;

    // Skip objects with zero velocity (stationary/empty filters)
    if (vx == 0.0f && vy == 0.0f) {
      // Delete stale markers in rviz
      visualization_msgs::Marker del_marker;
      del_marker.header.frame_id = "front_laser";
      del_marker.header.stamp = ros::Time::now();
      del_marker.id = i;
      del_marker.action = visualization_msgs::Marker::DELETE;
      tracked_markers.markers.push_back(del_marker);
      visualization_msgs::Marker del_text;
      del_text.header.frame_id = "base_link";
      del_text.header.stamp = ros::Time::now();
      del_text.id = i + 100;
      del_text.action = visualization_msgs::Marker::DELETE;
      tracked_markers.markers.push_back(del_text);
      continue;
    }

    state_msg.data.push_back((float)i);
    state_msg.data.push_back(x);
    state_msg.data.push_back(y);
    state_msg.data.push_back(vx);
    state_msg.data.push_back(vy);
    
    // Print readable output
    // cout << "Obj " << i << ": pos=(" << fixed << setprecision(2) 
    //      << x << ", " << y << ") vel=(" 
    //      << scientific << setprecision(2) << vx << ", " << vy << ")" << endl;
    
    // Create marker for this object
    visualization_msgs::Marker marker;
    marker.header.frame_id = "front_laser";
    marker.header.stamp = ros::Time::now();
    marker.id = i;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.0;
    
    marker.scale.x = 0.4;
    marker.scale.y = 0.4;
    marker.scale.z = 0.4;
    
    // Color based on object ID
    marker.color.a = 0.8;
    marker.color.r = (i % 2 == 0) ? 1.0 : 0.0;
    marker.color.g = (i % 3 == 0) ? 1.0 : 0.0;
    marker.color.b = (i % 4 == 0) ? 1.0 : 0.0;
    
    tracked_markers.markers.push_back(marker);
    
    // Create text marker with object info
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "base_link";
    text_marker.header.stamp = ros::Time::now();
    text_marker.id = i + 100;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    
    text_marker.pose.position.x = x;
    text_marker.pose.position.y = y;
    text_marker.pose.position.z = 0.5;
    
    text_marker.scale.z = 0.3;
    text_marker.color.a = 1.0;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    
    char text_str[50];
    snprintf(text_str, sizeof(text_str), "ID:%d\n(%.2f,%.2f)", (int)i, x, y);
    text_marker.text = text_str;
    
    tracked_markers.markers.push_back(text_marker);
  }
  
  // cout << "=====================\n" << endl;
  
  tracked_objects_pub.publish(state_msg);
  markerPub.publish(tracked_markers);
}

laser_geometry::LaserProjection projector_;

// Forward declaration
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input);

void scan_cb(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
    sensor_msgs::PointCloud2 cloud;
    projector_.projectLaser(*scan_msg, cloud);
    cloud_cb(boost::make_shared<sensor_msgs::PointCloud2>(cloud));
  }

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)

{
  if (firstFrame) {
    // Initialize Kalman Filters
    float dvx = 0.01f;
    float dvy = 0.01f;
    float dx = 1.0f;
    float dy = 1.0f;
    float sigmaP = 0.01;
    float sigmaQ = 0.1;

    for (int i = 0; i < NUM_FILTERS; i++) {
      KF[i] = cv::KalmanFilter(stateDim, measDim, ctrlDim, CV_32F);
      KF[i].transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                              dvx, 0, 0, 0, 0, dvy);
      cv::setIdentity(KF[i].measurementMatrix);
      setIdentity(KF[i].processNoiseCov, Scalar::all(sigmaP));
      cv::setIdentity(KF[i].measurementNoiseCov, cv::Scalar(sigmaQ));
    }

    // Process the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::fromROSMsg(*input, *input_cloud);

    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.08);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(600);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    std::vector<pcl::PointIndices>::const_iterator it;
    std::vector<int>::const_iterator pit;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
    std::vector<pcl::PointXYZ> clusterCentroids;

    for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);
      float x = 0.0;
      float y = 0.0;
      int numPts = 0;
      for (pit = it->indices.begin(); pit != it->indices.end(); pit++) {

        cloud_cluster->points.push_back(input_cloud->points[*pit]);
        x += input_cloud->points[*pit].x;
        y += input_cloud->points[*pit].y;
        numPts++;
      }

      pcl::PointXYZ centroid;
      centroid.x = x / numPts;
      centroid.y = y / numPts;
      centroid.z = 0.0;

      cluster_vec.push_back(cloud_cluster);
      clusterCentroids.push_back(centroid);
    }

    // Ensure at least NUM_FILTERS clusters exist
    while (cluster_vec.size() < NUM_FILTERS) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);
      empty_cluster->points.push_back(pcl::PointXYZ(0, 0, 0));
      cluster_vec.push_back(empty_cluster);
    }

    while (clusterCentroids.size() < NUM_FILTERS) {
      pcl::PointXYZ centroid;
      centroid.x = 0.0;
      centroid.y = 0.0;
      centroid.z = 0.0;
      clusterCentroids.push_back(centroid);
    }

    // Set initial state for all filters
    for (int i = 0; i < NUM_FILTERS; i++) {
      KF[i].statePre.at<float>(0) = clusterCentroids.at(i).x;
      KF[i].statePre.at<float>(1) = clusterCentroids.at(i).y;
      KF[i].statePre.at<float>(2) = 0; // initial v_x
      KF[i].statePre.at<float>(3) = 0; // initial v_y
    }

    firstFrame = false;

    for (int i = 0; i < NUM_FILTERS; i++) {
      geometry_msgs::Point pt;
      pt.x = clusterCentroids.at(i).x;
      pt.y = clusterCentroids.at(i).y;
      prevClusterCenters.push_back(pt);
    }
  }

  else {
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::fromROSMsg(*input, *input_cloud);

    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(600);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    std::vector<pcl::PointIndices>::const_iterator it;
    std::vector<int>::const_iterator pit;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
    std::vector<pcl::PointXYZ> clusterCentroids;

    for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
      float x = 0.0;
      float y = 0.0;
      int numPts = 0;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);
      for (pit = it->indices.begin(); pit != it->indices.end(); pit++) {

        cloud_cluster->points.push_back(input_cloud->points[*pit]);

        x += input_cloud->points[*pit].x;
        y += input_cloud->points[*pit].y;
        numPts++;
      }

      pcl::PointXYZ centroid;
      centroid.x = x / numPts;
      centroid.y = y / numPts;
      centroid.z = 0.0;

      cluster_vec.push_back(cloud_cluster);
      clusterCentroids.push_back(centroid);
    }

    // Ensure at least NUM_FILTERS clusters exist
    while (cluster_vec.size() < NUM_FILTERS) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);
      empty_cluster->points.push_back(pcl::PointXYZ(0, 0, 0));
      cluster_vec.push_back(empty_cluster);
    }

    while (clusterCentroids.size() < NUM_FILTERS) {
      pcl::PointXYZ centroid;
      centroid.x = 0.0;
      centroid.y = 0.0;
      centroid.z = 0.0;
      clusterCentroids.push_back(centroid);
    }

    std_msgs::Float32MultiArray cc;
    for (int i = 0; i < NUM_FILTERS; i++) {
      cc.data.push_back(clusterCentroids.at(i).x);
      cc.data.push_back(clusterCentroids.at(i).y);
      cc.data.push_back(clusterCentroids.at(i).z);
    }

    KFT(cc);
    int i = 0;
    bool publishedCluster[NUM_FILTERS];
    for (auto it = objID.begin(); it != objID.end(); it++) {
      if (i < NUM_FILTERS) {
        // Skip publishing clusters with zero velocity
        float vx = KF[i].statePre.at<float>(2);
        float vy = KF[i].statePre.at<float>(3);
        if (vx == 0.0f && vy == 0.0f) {
          i++;
          continue;
        }
        publish_cloud(pub_cluster[i], cluster_vec[*it]);
        publishedCluster[i] = true;
        i++;
      }
    }
  }
}

int main(int argc, char **argv) {
  // ROS init
  ros::init(argc, argv, "kf_tracker");
  ros::NodeHandle nh;

  // cout << "About to setup callback\n";

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("front/scan", 1, scan_cb);

  // Create publishers for each cluster
  for (int i = 0; i < NUM_FILTERS; i++) {
    std::string topic_name = "cluster_" + std::to_string(i);
    pub_cluster[i] = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
  }

  objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);
  tracked_objects_pub = nh.advertise<std_msgs::Float32MultiArray>("tracked_objects", 1);
  markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);

  ros::spin();
}