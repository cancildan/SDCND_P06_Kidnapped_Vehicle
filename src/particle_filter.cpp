/*
 * particle_filter.cpp
 *
 */
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

//Engine used for generating particles
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (is_initialized) {
    return;
    }
    
    // Initializing
    num_particles = 100;
  
    // Extracting standard deviations
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // Creating normal distributions
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    // Generate particles with normal distribution
    for (int i = 0; i < num_particles; ++i){
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;
        particles.push_back(particle);
        weights.push_back(1.0);
        }
  
        //Set as initialized
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // Extracting standard deviations
    double std_pos_x = std_pos[0];
    double std_pos_y = std_pos[1];
    double std_pos_theta = std_pos[2];
  
    // Creating normal distributions
    normal_distribution<double> dist_x(0, std_pos_x);
    normal_distribution<double> dist_y(0, std_pos_y);
    normal_distribution<double> dist_theta(0, std_pos_theta);

    for (int i = 0; i < num_particles; i++)
    {
        double theta = particles[i].theta;
        // yaw_rate is quite small
        if (fabs(yaw_rate) < EPS)
        {
        	particles[i].x += velocity * delta_t * cos(theta);
        	particles[i].y += velocity * delta_t * sin(theta);
        }
        // motion model while yaw_rate is not small
        else
        {
        	particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
        	particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
        	particles[i].theta += yaw_rate * delta_t;
        }
        // Adding noise.
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    unsigned int nObservations = observations.size();
    unsigned int nPredictions = predicted.size();
  
    for (unsigned int i = 0; i < nObservations; i++)
    {
        // Initialize min distance as a really big number.
        double closestDistance = numeric_limits<double>::max();

        // initialize the id counter variable.
        int index = 0;

        // scan landmarks to find the closest predictions under assigned observations
        for (unsigned int j = 0; j < nPredictions; j++)
        {
            // calculate the distance between j-th prediction and i-th observation
            double currentDistance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

            // criteria to update the index in the loop
            if (currentDistance < closestDistance)
            {
            	// if criteria fulfilled, assign id label to the i-th observation
            	observations[i].id = index;
            // update the closestDistance variable.
            closestDistance = currentDistance;
            }
            index++;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for(int i=0; i<num_particles; i++)
    {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;
            
        vector<LandmarkObs> mappedObservations;
        vector<LandmarkObs> inRangeLandmarks;

        mappedObservations.clear();

        // Transform particle observations from vehicle coordinates to map coordinates
        for (int j=0; j<observations.size(); j++)
        {
            //transform observations to world map coordinates
            double observation_x = x + observations[j].x * cos(theta) - observations[j].y * sin(theta);
            double observation_y = y + observations[j].x * sin(theta) + observations[j].y * cos(theta);

            LandmarkObs transformedObs = {observations[j].id, observation_x, observation_y};
            mappedObservations.push_back(transformedObs);
        }

        // Get landmarks in particle's range
        for (unsigned j=0; j<map_landmarks.landmark_list.size(); j++)
        {
            int landmark_id = map_landmarks.landmark_list[j].id_i;
            double landmark_x  = map_landmarks.landmark_list[j].x_f;
            double landmark_y  = map_landmarks.landmark_list[j].y_f;

            if (dist(particles[i].x, particles[i].y, landmark_x, landmark_y) <= sensor_range)
            {
                LandmarkObs landmarkObservation = {landmark_id, landmark_x, landmark_y};
                //cout << "landmarkObservation.id" << landmarkObservation.id << endl;
                inRangeLandmarks.push_back(landmarkObservation);
            }

        }

        // in case of any missing landmarks
        if (inRangeLandmarks.size() == 0) {
            //cout << "No landmarks specified" << endl;
            continue;
        }

        // Data association
        dataAssociation(inRangeLandmarks, mappedObservations);
        
        // Setup weights based on particle observation and actual observation.
        particles[i].weight = 1.0;
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];
        double nx = 2.0 * std_x * std_x;
        double ny = 2.0 * std_y * std_y;
        double gaussian_norm = 2.0 * M_PI * std_x * std_y;

        for (unsigned int j=0; j<mappedObservations.size(); j++)
        {
            double x_sq = pow((mappedObservations[j].x - inRangeLandmarks[mappedObservations[j].id].x),2);
            double y_sq = pow((mappedObservations[j].y - inRangeLandmarks[mappedObservations[j].id].y),2);

            // product of the obersvation weight
            particles[i].weight *=  exp(-(x_sq/nx + y_sq/ny)) / gaussian_norm;
            //cout << "weight: " << particles[i].weight << endl;
        }
        // calculated weights for each particle
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    double maxWeight = *std::max_element(weights.cbegin(), weights.cend());  
    // Creating distributions.
    uniform_real_distribution<double> distDouble(0.0, maxWeight);
    uniform_int_distribution<int> distInt(0, num_particles - 1);
  
    // Generating index.
    int index = std::rand() % num_particles;

    double beta = 0.0;
    
    // the wheel
    vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++) {
      beta += distDouble(gen) * 2.0;
      while( beta > weights[index]) {
        beta -= weights[index];
        index = (index + 1) % num_particles;
      }
      resampledParticles.push_back(particles[index]);
    }

    particles = resampledParticles;
  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1); 
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}