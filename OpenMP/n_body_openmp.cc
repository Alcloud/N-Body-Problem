#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h> /* for OpenMP */
#include <iostream>

#include <vector>
#include <fstream>
#include <sstream> 
#include <iterator>
#include <algorithm>

#define NUM_THS 1 /* the number of threads in OpenMP */

#define ITERATION 3650 /* run in ITERATION times */
#define DELTA_T 0.00027397260273972603
#define EPS 0.000001

/* Gravitational constant */
// const double G_CONS=0.0000000000667259;
const double G_CONS = 0.03765;

/* body structure */
struct Body
{
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double w;          /* weight */
};

template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

int main(int argc, char *argv[])
{
    int const max_bodies = 22;
    /* start to record time */
    // double Elapsed_time = omp_get_wtime();
    /* open file to record time */
    // FILE *pFile;
    // pFile = fopen("omp_htw.csv", "w");

    /* N-Body variables */

    std::ifstream dataFile("htw.csv");
	std::string line;

	// new lines will be skipped unless we stop it from happening:    
	dataFile.unsetf(std::ios_base::skipws);

	// count the newlines with an algorithm specialized for counting:
	unsigned NUM_BODY = std::count(
		std::istream_iterator<char>(dataFile),
		std::istream_iterator<char>(),
		'\n');

	dataFile.clear();
	dataFile.seekg(0, std::ios::beg);

    struct Body Nbody[NUM_BODY]; /* record the properties of all bodies */
    
    int i, j, k, n;
    
    std::cout << "loading data " << NUM_BODY << " bodies... ";
    i = 0;
	while (std::getline(dataFile, line)) {
		std::vector<std::string> pos = split(line, ',');
        Nbody[i].x = atof(pos[0].c_str());
        Nbody[i].y = atof(pos[1].c_str());
        Nbody[i].z = atof(pos[2].c_str());
        Nbody[i].w = atof(pos[3].c_str());

        Nbody[i].vx = 0;
        Nbody[i].vy = 0;
        Nbody[i].vz = 0;
        i++;
		// std::cout << pos_host->x << ", " << pos_host->y << "\n";
	}
    int max_treads = omp_get_max_threads();
    std::cout << "done\n" << "max trheads: " << max_treads << "\n" ;

    int bodyR[NUM_BODY];         /* to record the radius of bodies */
    double newVx[NUM_BODY];      /* for calculate each body's velocity */
    double newVy[NUM_BODY];
    double newVz[NUM_BODY];

    /* start to simulate N-body */
    for (k = 0; k < ITERATION; k++)
    {

        /* set the number of threads */
        omp_set_num_threads(NUM_THS);
#pragma omp parallel private(j)
        {
/* Calculate the position of bodies by point-to-point in each iteration */
#pragma omp for schedule(static)
            for (i = 0; i < NUM_BODY; i++)
            {
                for (j = 0; j < NUM_BODY; j++)
                {
                    if (j == i)
                    { /* there is no need to calculate the effect from itself */
                        continue;
                    }
                    double delta_x = Nbody[j].x - Nbody[i].x;
                    double delta_y = Nbody[j].y - Nbody[i].y;
                    double delta_z = Nbody[j].z - Nbody[i].z;
                    double distance = 1 / sqrt(pow(delta_x, 2) + pow(delta_y, 2) + pow(delta_z, 2) + EPS);

                    double force = G_CONS * Nbody[j].w * distance * distance * distance;
                    newVx[i] = newVx[i] + force * delta_x;
                    newVy[i] = newVy[i] + force * delta_y;
                    newVz[i] = newVz[i] + force * delta_z;
                }
            }
/* update the new data */
#pragma omp for schedule(static)
            for (i = 0; i < NUM_BODY; i++)
            {
                Nbody[i].x = Nbody[i].x + Nbody[i].vx * DELTA_T + 0.5 * DELTA_T * DELTA_T * newVx[i];
                Nbody[i].y = Nbody[i].y + Nbody[i].vy * DELTA_T + 0.5 * DELTA_T * DELTA_T * newVy[i];
                Nbody[i].z = Nbody[i].z + Nbody[i].vz * DELTA_T + 0.5 * DELTA_T * DELTA_T * newVz[i];
                Nbody[i].vx = Nbody[i].vx + DELTA_T * newVx[i];
                Nbody[i].vy = Nbody[i].vy + DELTA_T * newVy[i];
                Nbody[i].vz = Nbody[i].vz + DELTA_T * newVz[i];
                newVx[i] = 0;
                newVy[i] = 0;
                newVz[i] = 0;
                // fprintf(pFile,"%lf,%lf,%lf", Nbody[i].x, Nbody[i].y, Nbody[i].z);
                // if(i<NUM_BODY-1){
                //     fprintf(pFile,",");
                // }
            }
        }
        // fprintf(pFile,"\n");
    }

    /* stop to record time */
    // Elapsed_time = omp_get_wtime() - Elapsed_time;
    // fclose(pFile);
    // printf("Total run time=%f\n",Elapsed_time);
    /* Program Finished */
    return 0;
}