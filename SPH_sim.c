#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <GLFW\glfw3.h>

#define grid_X 16
#define grid_Y 16

//particle class
#define MAX_NUM_PARTICLES 200
#define stride 5
//x,y,u,v,l
//note that particles ids are in their respective index

//world definitions
#define spawn_spacing 1.0f
#define dt 0.0167f
#define grav_X 0.0f
#define grav_Y -9.81f
#define range 2.0f
#define kappa 0.1f
#define pie 3.14f
//pressure correction takes range*-0.2
#define radius 0.2f
#define e 0.0001f
#define rho0 12.0f
//bear in mind 'range' is the radius of the kernel functions detection range

float* particles = NULL;
float* neighbor_p = NULL;
int* neighbor_prefix = NULL;
int* neighbor_names = NULL;
//individual kernel gradient, dot_product, x,y
#define n_stride 4

float power(float target, int value) {
    float ref = target;
    for (int i = value; i < value; i++) {
        ref *= target;
    }
    return ref;
}

void integrate_particles(int p_num) {
    particles = calloc(p_num * stride, sizeof(float));
    neighbor_p = calloc(p_num * stride * p_num, sizeof(float));
    neighbor_prefix = calloc(p_num * p_num + 1, sizeof(int));
    neighbor_names = calloc(p_num * p_num, sizeof(int));
    int count = 0;
    float y = 0.0f;

    for (int i = 0; i < grid_Y && count < p_num; i++) {
        float x = 0.0f;
        for (int j = 0; j < grid_X && count < p_num; j++) {
            particles[count * stride] = x; // x position
            particles[count * stride + 1] = y; // y position
            // u, v, c, p, f, s left at 0
            count++;
            x += spawn_spacing;
        }
        y += spawn_spacing;
    }
}

void apply_force() {
    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        particles[i * stride + 2] += grav_X;
        particles[i * stride + 3] += grav_Y;
    }
}
void timestep() {
    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        int pc = i * stride;
        particles[pc] += particles[pc + 2] * dt;
        particles[i * stride + 1] += particles[pc + 3] * dt;
    }
}

float dist_p2p(int pi, int pj) {
    float xi = particles[pi * stride];
    float yi = particles[pi * stride + 1];
    float xj = particles[pj * stride];
    float yj = particles[pj * stride + 1];

    float dx = xi - xj;
    float dy = yi - yj;
    float absolute_dist = sqrt(dx * dx + dy * dy);
    if (absolute_dist > range) return 0.0f;
    return absolute_dist;
}

void set_p_neighbor() {
    int index = 0;  // flat index in neighbor_p
    neighbor_prefix[0] = 0;
    int name_index = 0;
    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        int count = 0;

        for (int j = 0; j < MAX_NUM_PARTICLES; j++) {
            if (i == j) continue;

            float xi = particles[i * stride];
            float yi = particles[i * stride + 1];
            float xj = particles[j * stride];
            float yj = particles[j * stride + 1];

            float dx = xi - xj;
            float dy = yi - yj;
            float dot = dx * dx + dy * dy;
            float abs = sqrt(dot);
            if (abs > range) {
                continue;
            }
            else {

                if (abs > 0.0f && abs < range) {
                    neighbor_names[name_index++] = j;
                    float weight = 315.0f * power(abs,3) / (64.0f* pie * power(range, 9));
                    neighbor_p[index++] = weight;        // kernel weight
                    neighbor_p[index++] = dot;
                    neighbor_p[index++] = dx;
                    neighbor_p[index++] = dy;
                    count++;
                }
            }
        }

        neighbor_prefix[i + 1] = neighbor_prefix[i] + count;
    }
}

void test_neighbors() {
    for (int frame = 0; frame < 3; frame++) {
        for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
            int diff = neighbor_prefix[i + 1] - neighbor_prefix[i];
            printf("target particle %d has %d neighboring particles\n", i, diff);
            for (int j = 0; j < diff; j++) {
                printf("%d .) %.2f \n", j + 1, neighbor_p[j]);
            }

        }

    }
}

void mov_correct() {
    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        int first = neighbor_prefix[i];
        int last = neighbor_prefix[i + 1];
        float gradientsum = 0.0f;
        float weightsum = 0.0f;
        for (int j = first; j < last; j++) {
            float weight = neighbor_p[j * n_stride];
            float dot = neighbor_p[j * n_stride + 1];
            weightsum += weight;
            gradientsum += dot;
        }
        particles[i * stride + 4] = -((weightsum / rho0 - 1.0f) / (gradientsum + e));
    }
}

void position_change() {
    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        float lambda_i = particles[i * stride + 4];
        int start = neighbor_prefix[i];
        int end = neighbor_prefix[i + 1];
        float sum = 0.0f;
        float lx = 0.0f;
        float ly = 0.0f;
        for (; start < end; start++) {
            int k = neighbor_names[start];
            if (k == i) continue;
            float p_corr = -kappa*power(neighbor_p[start * n_stride] / .1f*range, 7);
            float lambda_j = particles[k * stride + 4];
            float scale_coeff = lambda_i + lambda_j + p_corr;
            lx += neighbor_p[start * n_stride + 2] * scale_coeff;
            ly += neighbor_p[start * n_stride + 3] * scale_coeff;

        }

        particles[i * stride] += lx / rho0;
        particles[i * stride + 1] += ly / rho0;
    }
}

void boundary_check() {
    const float bounce_damping = .2f;  // adjust to control how strong the bounce is

    for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
        int pi = i * stride;
        float x = particles[pi];
        float y = particles[pi + 1];

        // Check X boundaries
        if (x < 0.0f) {
            particles[pi] = 0.0f;
            particles[pi + 2] *= -bounce_damping;  // reverse and dampen X velocity
        }
        else if (x > grid_X) {
            particles[pi] = grid_X;
            particles[pi + 2] *= -bounce_damping;
        }

        // Check Y boundaries
        if (y < 0.0f) {
            particles[pi + 1] = 0.0f;
            particles[pi + 3] *= -bounce_damping;  // reverse and dampen Y velocity
        }
        else if (y > grid_Y) {
            particles[pi + 1] = grid_Y;
            particles[pi + 3] *= -bounce_damping;
        }
    }
}


void main() {
    //printf("integrating...\n");
    //printf("done!\n");
    //printf("detecting neighbors...\n");
    //printf("done!\n");
    //test_neighbors();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 800, "FLIP Sim", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    integrate_particles(MAX_NUM_PARTICLES);
    int framef = 0;
        clock_t start = clock();
    while (!glfwWindowShouldClose(window)) {
        printf("frames: %d\n", framef++);
        glClear(GL_COLOR_BUFFER_BIT);
        // Simulate particles
        apply_force();
        set_p_neighbor();

        mov_correct();
        position_change();
        timestep();
        boundary_check();
        (particles[1] > 50.0f) ? printf("Particle 0: x=%.2f y=%.2f\n", particles[0], particles[1]) : printf("");
        
        
        glClear(GL_COLOR_BUFFER_BIT);
        glLoadIdentity();
        glColor3f(100.0f, 100.0f, 100.0f); // White particles
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < MAX_NUM_PARTICLES; i++) {
            float x = particles[i * stride];
            float y = particles[i * stride + 1];
            float nx = (x / 100.0f) * 2.0f - 1.0f;
            float ny = (y / 100.0f) * 2.0f - 1.0f;
            glVertex2f(nx, ny);
        }
        glEnd();


        glfwSwapBuffers(window);
        glfwPollEvents();
    }

        clock_t end = clock();
        double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Simulation took %f seconds\n", elapsed_time);
        printf("calculated sim fps: %f!\n", framef / elapsed_time);
    glfwDestroyWindow(window);
    glfwTerminate();

    //int frames = 5000;
        //integrate_particles(MAX_NUM_PARTICLES);

        // your simulation loop
        //for (int i = 0; i < frames; i++) {
        //    apply_force();
        //    set_p_neighbor();
        //    mov_correct();
        //    position_change();
        //    timestep();
        //}

}