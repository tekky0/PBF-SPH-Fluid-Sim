#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <GLFW\glfw3.h>

#define gridX 50
#define gridY 50

#define particleNUM 250
#define p_radii 1.0f
#define sqrp_radii (p_radii*p_radii)
#define pstride 8
#define max_neighbors ((int)(particleNUM))
#define PI 3.14
#define rho0 75.0f
#define dt 0.03f
#define tinystep 0.0001f
#define SpikyPow2ScalingFactor 1
#define SpikyPow3DerivativeScalingFactor 3
#define SpikyPow3ScalingFactor 1
#define k 500.0f
#define nk 300.0f
#define SpikyPow2DerivativeScalingFactor 3
#define gravityY -9.81f
#define gravityX 0.0f

float* particles = NULL;
//x,y,u,v,density, near density, px,py

//hash
#define hk1 15823
#define hk2 9737333
#define spacing 3
#define buckets (int)((gridX / p_radii) * (gridY / p_radii))
int* hashtable = NULL;
int* p_list = NULL;
int* prefix_sum = NULL;
int* nlist = NULL;

int start_hash() {
	hashtable = calloc(buckets, sizeof(int));
	prefix_sum = calloc(buckets + 1, sizeof(int));
	p_list = calloc(particleNUM, sizeof(int));

	for (int i = 0; i < buckets; i++) {
		hashtable[i] = 0;
	}
}

void home_particles() {
    // --- STEP 1: Count how many particles per bucket ---
    for (int i = 0; i < buckets; i++) {
        hashtable[i] = 0;
    }

    for (int i = 0; i < particleNUM; i++) {
        int x = (int)(particles[i * pstride] / p_radii) * hk1;
        int y = (int)(particles[i * pstride + 1] / p_radii) * hk2;
        int hash = (x ^ y) % buckets;
        if (hash < 0) hash += buckets; // ensure positive hash
        hashtable[hash]++;
    }

    // --- STEP 2: Build prefix sum into prefix_sum[] ---
    prefix_sum[0] = 0;
    for (int i = 1; i <= buckets; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + hashtable[i - 1];
    }

    // --- STEP 3: Scatter particle IDs into p_list based on hash ---
    // Reset counts to reuse as offset tracker
    for (int i = 0; i < buckets; i++) {
        hashtable[i] = 0;
    }

    for (int i = 0; i < particleNUM; i++) {
        int x = (int)(particles[i * pstride] / p_radii) * hk1;
        int y = (int)(particles[i * pstride + 1] / p_radii) * hk2;
        int hash = (x ^ y) % buckets;
        if (hash < 0) hash += buckets;

        int index = prefix_sum[hash] + hashtable[hash];
        p_list[index] = i;  // store particle ID
        hashtable[hash]++;
    }
}

void spawn_particles() {
    // Allocate memory for particles: x, y, cell, density
    particles = calloc(particleNUM * pstride, sizeof(float));

    for (int i = 0; i < particleNUM; i++) {
        float randX = ((float)rand() / RAND_MAX) * gridX;
        float randY = ((float)rand() / RAND_MAX) * gridY;

        particles[i * pstride + 0] = randX; // x
        particles[i * pstride + 1] = randY; // y
        particles[i * pstride + 2] = 0.0f;  // cell or unused
        particles[i * pstride + 3] = 0.0f;  // density or any other value
        particles[i * pstride + 4] = 0.0f;  // density or any other value

    }
}

int get_neighbors_all_ids(int pid, int* neighbor_ids) {
    int count = 0;

    float px = particles[pid * pstride + 0];
    float py = particles[pid * pstride + 1];

    int cx = (int)(px / p_radii);
    int cy = (int)(py / p_radii);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;

            int hx = nx * hk1;
            int hy = ny * hk2;
            int h = (hx ^ hy) % buckets;
            if (h < 0) h += buckets;

            int start = prefix_sum[h];
            int end = prefix_sum[h + 1];

            for (int i = start; i < end && count < max_neighbors; i++) {
                int nid = p_list[i];
                neighbor_ids[count++] = nid;
            }
        }
    }

    return count;
}

void magnitude(float* dst) {
    if (*dst < 0.0f)
        *dst = -*dst;
}

float power(float base, int exponent) {
    float result = 1.0f;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

float SpikyKernelPow2(float dst, float radius)
{
    if (dst < radius)
    {
        float v = radius - dst;
        return v * v * SpikyPow2ScalingFactor;
    }
    return 0;
}

float DerivativeSpikyPow3(float dst, float radius)
{
    if (dst <= radius)
    {
        float v = radius - dst;
        return -v * v * SpikyPow3DerivativeScalingFactor;
    }
    return 0;
}

float DerivativeSpikyPow2(float dst, float radius)
{
    if (dst <= radius)
    {
        float v = radius - dst;
        return -v * SpikyPow2DerivativeScalingFactor;
    }
    return 0;
}

float SpikyKernelPow3(float dst, float radius)
{
	if (dst < radius)
	{
		float v = radius - dst;
		return v * v * v * SpikyPow3ScalingFactor;
	}
	return 0;
}

void calcDensity(int i, int count) {
    float density = 0.0f;
    float neardensity = 0.0f;

    for (int j = 0; j < count; j++) {
        int neighbor = nlist[j];
        if (neighbor == i) continue;
        float dx = particles[neighbor * pstride+6] - particles[i * pstride + 6];
        float dy = particles[neighbor * pstride+7] - particles[i * pstride + 7];
        float dst = sqrt(dx * dx + dy * dy);
        density += SpikyKernelPow2(dst, p_radii);
        neardensity += SpikyKernelPow3(dst, p_radii);
    }
    particles[i * pstride + 4] = density;
    particles[i * pstride + 5] = neardensity;
}

float calcSharedPressure(float densi, float densj) {
    float shared_pressure = (densj + densi)*0.5;
    return shared_pressure;
}

float pressure_fromDensity(float rhoi) {
    return (rhoi - rho0) * k;
}
float pressure_fromNearDensity(float nrhoi) {
    return nrhoi * nk;
}

float dot(float a, float b) {
    return (a * a + b * b);
}

void calcGrad(int i, int count) {
    float density = particles[i * pstride + 4];
    if (density < 1e-6f || isnan(density)) return;
    float pressureForcex = 0.0f;
    float pressureForcey = 0.0f;

    float near_density = particles[i * pstride + 5];
    float px = particles[i * pstride + 6];
    float py = particles[i * pstride + 7];
    float nearPressure = pressure_fromNearDensity(near_density);
    float pressure = pressure_fromDensity(density);
    

    for (int j = 0; j < count; j++) {
        int n = nlist[j];
        if (n == i) continue;

        
        float dx = particles[n * pstride + 6] - px;
        float dy = particles[n * pstride + 7] - py;
        if (isnan(dx) || isnan(dy)) continue;

        float sqrDistToNeighbor = dx * dx+ dy * dy;
        if (sqrDistToNeighbor > sqrp_radii) continue;
       
        float dst = sqrtf(sqrDistToNeighbor);
        float dirToNeighbor_x = dst > 0 ? dx : 1e-9;
        float dirToNeighbor_y = dst > 0 ? dy : 1e-9;
        
        float neighborDensity = particles[n * pstride + 4];
        float neighborNearDensity = particles[n * pstride + 5];
        float neighborPressure = pressure_fromDensity(neighborDensity);
        float neighborNearPressure = pressure_fromNearDensity(neighborNearDensity);

        float sharedPressure = calcSharedPressure(neighborPressure, pressure);
        float sharedNearPressure = calcSharedPressure(neighborNearPressure, nearPressure);

        float pressureScaler = DerivativeSpikyPow2(dst, p_radii) * sharedPressure;
        float nearPressureScaler = DerivativeSpikyPow3(dst, p_radii) * sharedNearPressure;
        pressureForcex += dirToNeighbor_x * pressureScaler / neighborDensity;
        pressureForcey += dirToNeighbor_y * pressureScaler / neighborDensity;
        pressureForcex += dirToNeighbor_x * nearPressureScaler / neighborNearDensity;
        pressureForcey += dirToNeighbor_y * nearPressureScaler / neighborNearDensity;    
    }

    if (!isnan(pressureForcex) && !isnan(pressureForcey)) {
        particles[i * pstride + 2] = pressureForcex * dt;
        particles[i * pstride + 3] = pressureForcey * dt;
    }
}

void gravity(int i) {
    particles[i * pstride + 2] += gravityX;
    particles[i * pstride + 3] += gravityY;
    particles[i * pstride + 6] += particles[i*pstride]*tinystep;
    particles[i * pstride + 7] += particles[i*pstride+1]*tinystep;

}
void timestep(int i) {
    particles[i * pstride] += particles[i * pstride + 2] * dt;
    particles[i * pstride + 1] += particles[i * pstride + 3] * dt;
}

void boundary_check(int i) {
    const float bounce_damping = .4f;  // adjust to control how strong the bounce is
        int pi = i * pstride;
        float x = particles[pi];
        float y = particles[pi + 1];

        // Check X boundaries
        if (x < 0.0f) {
            particles[pi] = 0.1f;
            particles[pi + 2] *= -bounce_damping;  // reverse and dampen X velocity
        }
        else if (x > gridX) {
            particles[pi] = gridX - 0.1f;
            particles[pi + 2] *= -bounce_damping;
        }

        // Check Y boundaries
        if (y < 0.0f) {
            particles[pi + 1] = 0.1f;
            particles[pi + 3] *= -bounce_damping;  // reverse and dampen Y velocity
        }
        else if (y > gridY) {
            particles[pi + 1] = gridY - 0.1f;
            particles[pi + 3] *= -bounce_damping;
        }
}

void sim() {
        home_particles();
    for (int i = 0; i < particleNUM; i++) {
        int count = get_neighbors_all_ids(i,nlist);
        printf("0 %.2f\n", particles[2]);

        gravity(i);

        calcDensity(i, count);
        printf("1 %.2f\n", particles[2]);

        calcGrad(i, count);
        printf("2 %.2f\n", particles[2]);

        timestep(i);
        printf("3 %.2f\n", particles[2]);

        boundary_check(i);
        printf("4 %.2f\n", particles[2]);

    }
}


int main() {
    start_hash();
    nlist = calloc(max_neighbors, sizeof(int));
    spawn_particles();
    GLFWwindow* window;
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }

    window = glfwCreateWindow(800, 800, "Fluid Sim", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
    glfwSwapInterval(1); // Optional: vsync

    int count = 0;
    while (!glfwWindowShouldClose(window)) {

        // --- Run simulation every loop ---
        sim();

        // --- Check if it's time to render (every 4 ms = 250 Hz) ---
        if (1) {
            count++;
            glClear(GL_COLOR_BUFFER_BIT);
            glLoadIdentity();
            glColor3f(1.0f, 1.0f, 1.0f); // White particles
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (int i = 0; i < particleNUM; i++) {
                float x = particles[i * pstride];
                float y = particles[i * pstride + 1];
                float nx = (x / 100.0f) * 2.0f - 1.0f;
                float ny = (y / 100.0f) * 2.0f - 1.0f;
                glVertex2f(nx, ny);
            }
            glEnd();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    return 0;

}