#include <cstdlib>
#include <random>
#include <iostream>
#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <opencv2/imgcodecs.hpp>
#include "localisation_pkg/srv/localisation.hpp"

#define PI 3.141592f

struct individual{
    float x;
    float y;
    float theta;
};

const cv::String mapFilename = "/home/orin/vision_github/VISION/src/vision_pkg/vision_pkg/maps/distance_test_field.png";

cv::Mat mapImage;

int MAP_WIDTH;
int MAP_HEIGHT;

float CROSSOVER_RATE = 0.7f;
float MUTATION_RATE = 0.2f;

float LINEAR_MUTATION_STD_DEV = 30.0f;
float POLAR_MUTATION_STD_DEV = 0.30f;

const int MAX_GENERATIONS = 100;
const int POPULATION_SIZE = 200;
float EARLY_BREAK_THRESHOLD = 0.95;
int PLATEAU_THRESHOLD = 10;

const bool LOGGING = true;

std::default_random_engine generator;

std::uniform_real_distribution<float> normalized(0.0f, 1.0f);

std::uniform_int_distribution<int> populationSize(0, POPULATION_SIZE-1);

std::normal_distribution<float> linearMutation(0.0f, LINEAR_MUTATION_STD_DEV);
std::normal_distribution<float> polarMutation(0.0f, POLAR_MUTATION_STD_DEV);

void initPopulation(int length, individual population[], std::vector<float32_t> const &boundsData){
    for(int i = 0; i < length; i++){
        population[i].x = boundsData[0] + normalized(generator) * (boundsData[1] - boundsData[0]);
        population[i].y = boundsData[2] + normalized(generator) * (boundsData[3] - boundsData[2]);
        population[i].theta = boundsData[4] + normalized(generator) * (boundsData[5] - boundsData[4]);
    }
}

void selectPopulation(int length, float fitnessScores[],individual currGen[] ,individual nextGen[]){
    for(int i = 0; i < length; i++){
        int id1 = populationSize(generator);
        int id2 = populationSize(generator);
        nextGen[i] = (fitnessScores[id1] > fitnessScores[id2]) ? currGen[id1] : currGen[id2];
    }
}

void crossover(individual& parent1, individual& parent2, individual& child1, individual& child2){
    if(normalized(generator) > CROSSOVER_RATE){
        child1 = parent1;
        child2 = parent2;
        return;
    }

    child1.x = (normalized(generator) > 0.5f)?parent1.x:parent2.x;
    child1.y = (normalized(generator) > 0.5f)?parent1.y:parent2.y;
    child1.theta = (normalized(generator) > 0.5f)?parent1.theta:parent2.theta;

    child2.x = (normalized(generator) > 0.5f)?parent1.x:parent2.x;
    child2.y = (normalized(generator) > 0.5f)?parent1.y:parent2.y;
    child2.theta = (normalized(generator) > 0.5f)?parent1.theta:parent2.theta;
    return; 
}

void mutate(individual &indi, std::vector<float32_t> const &boundsData){
    if(normalized(generator) > MUTATION_RATE) return;
    indi.x += linearMutation(generator);
    indi.x = (indi.x>boundsData[1])? boundsData[1]: indi.x;
    indi.x = (indi.x<boundsData[0])? boundsData[0]: indi.x;

    indi.y += linearMutation(generator);
    indi.y = (indi.y>boundsData[3])? boundsData[3]: indi.y;
    indi.y = (indi.y<boundsData[2])? boundsData[2]: indi.y;

    indi.theta += polarMutation(generator);
    indi.theta = (indi.theta>boundsData[5])? (float)boundsData[5]: indi.theta;
    indi.theta = (indi.theta<boundsData[4])? (float)boundsData[4]: indi.theta;

    return;
}

int getFitness(individual &indi, int numPoints, std::vector<int16_t> const &inputData, const int FIXED_POINT_ONE, const int FIXED_SHIFT){
    int newX, newY, sum = 0;
    int cachedSin = (int)(std::sin(indi.theta)*FIXED_POINT_ONE);
    int cachedCos = (int)(std::cos(indi.theta)*FIXED_POINT_ONE);
    int indiXFixedPoint = indi.x*FIXED_POINT_ONE;
    int indiYFixedPoint = indi.y*FIXED_POINT_ONE;
    for(int i = 0; i < 2*numPoints; i+=2){
        newX = (((inputData[i])*cachedCos) - ((inputData[i+1])*cachedSin) + indiXFixedPoint) >> FIXED_SHIFT;
        newY = (((inputData[i])*cachedSin) + ((inputData[i+1])*cachedCos) + indiYFixedPoint) >> FIXED_SHIFT;
        if(newX < 0 || newX >= MAP_WIDTH || newY < 0 || newY >= MAP_HEIGHT) continue;
        sum += mapImage.at<uchar>(newY, newX);
    }
    
    return sum;
}

void runGeneticAlgorithm(int numPoints, std::vector<int16_t> const &inputData, std::vector<float32_t> const &boundsData, float &xOut, float &yOut, float &thetaOut){

    auto start = std::chrono::high_resolution_clock::now();

    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    float fitnessScores[POPULATION_SIZE];
    individual population[POPULATION_SIZE];
    initPopulation(POPULATION_SIZE, population, boundsData);

    individual bestIndividual = population[0];
    float bestFitness = -INFINITY;

    float currentBestFitness = -INFINITY;
    int currentBestIndividualId = 0;

    float lastBestFitness = -INFINITY;
    int plateauCount = 0;

    float fitnessNormalizationFactor = 1.0f/(255.0f * (float)numPoints);
    const int FIXED_SHIFT = 16;
    const int FIXED_POINT_ONE = 1 << FIXED_SHIFT;

    for(int genNum = 0; genNum < MAX_GENERATIONS; genNum++){
        currentBestFitness = -INFINITY;
        for(int i = 0; i < POPULATION_SIZE; i++){
            fitnessScores[i] = fitnessNormalizationFactor * (float)getFitness(population[i], numPoints, inputData, FIXED_POINT_ONE, FIXED_SHIFT);
            if(fitnessScores[i] > currentBestFitness){
                currentBestFitness = fitnessScores[i];
                currentBestIndividualId = i;
            } 
        }
        if(currentBestFitness > bestFitness){
            bestFitness = currentBestFitness;
            bestIndividual = population[currentBestIndividualId];
        }

        if(bestFitness > EARLY_BREAK_THRESHOLD){
            if(LOGGING) std::cout << "Early break after " << genNum << " generations" << std::endl;
            break;
        }

        if(bestFitness == lastBestFitness){
            plateauCount++;
            if(plateauCount > PLATEAU_THRESHOLD){
                if(LOGGING) std::cout << "Plateau reached after " << genNum << " generations" << std::endl;
                break;
            }
        }else{
            plateauCount = 0;
            lastBestFitness = bestFitness;
        }

        individual selected[POPULATION_SIZE];
        selectPopulation(POPULATION_SIZE, fitnessScores, population, selected);

        for(int i = 0; i+1 < POPULATION_SIZE; i += 2){
            crossover(selected[i], selected[i+1], population[i], population[i+1]);
            mutate(population[i], boundsData);
            mutate(population[i+1], boundsData);
        }

        population[0] = bestIndividual;
        
    }

    auto stop = std::chrono::high_resolution_clock::now();

    xOut = bestIndividual.x;
    yOut = bestIndividual.y;
    thetaOut = bestIndividual.theta;

    if(LOGGING){
        std::cout << "X = " << bestIndividual.x << " Y = " << bestIndividual.y << " Theta = " << bestIndividual.theta << " Fitness = " << bestFitness << std::endl;
        int timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << 1000000/timeTaken << " fps " << timeTaken << " μs" <<std::endl;
    }
}


void localise(const std::shared_ptr<localisation_pkg::srv::Localisation::Request> request, std::shared_ptr<localisation_pkg::srv::Localisation::Response> response)
{   
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Incoming request with bounds %f<x<%f %f<y<%f %f<theta<%f", request->bounds.data[0], request->bounds.data[1], request->bounds.data[2], request->bounds.data[3], request->bounds.data[4], request->bounds.data[5]);

    float x, y, theta;
    
    runGeneticAlgorithm(request->points.data.size()/2, request->points.data, request->bounds.data, x, y, theta);

    auto spatialMessage = std_msgs::msg::Float32MultiArray();
    spatialMessage.data.resize(3);
    spatialMessage.data[0] = x;
    spatialMessage.data[1] = y;
    spatialMessage.data[2] = theta;

    response->transform = spatialMessage;

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sending back response: X = [%f] Y = [%f] Theta = [%f]", (float)(response->transform.data[0]), (float)(response->transform.data[1]), (float)(response->transform.data[2]));
}

int main(int argc, char *argv[]){
    rclcpp::init(argc, argv);

    mapImage = cv::imread(mapFilename, cv::IMREAD_GRAYSCALE);
    MAP_HEIGHT = mapImage.rows;
    MAP_WIDTH = mapImage.cols;
    std::cout<<MAP_HEIGHT<<" "<<MAP_WIDTH<<std::endl;

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("localisation_server");

    rclcpp::Service<localisation_pkg::srv::Localisation>::SharedPtr service =
    node->create_service<localisation_pkg::srv::Localisation>("localise", &localise);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ready to localise");

    rclcpp::spin(node);
    rclcpp::shutdown();
}