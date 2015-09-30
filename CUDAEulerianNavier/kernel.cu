
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "GLFW\glfw3.h"
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "GLFW/glfw3dll.lib")

#include <glm\glm.hpp>

#include <cusp\array1d.h>
#include <cusp\coo_matrix.h>
#include <cusp\krylov\bicgstab.h>
#include <cusp\krylov\cg.h>
#include <cusp\monitor.h>
#include <cusp\precond\diagonal.h>

#include <Windows.h>

#include "fluidQ.h"

using namespace std;
using namespace glm;

const float imageWidth = 600,
imageHeight = 600;

float PI = 3.14159;

const int mapW = 128,
mapH = 128;

#define dt (1.f / 30.f)
const float rho = 0.1;

float p[mapW][mapH];

GLFWwindow* window;

enum cellType
{
	WATER, AIR, SOLID
};
cellType type[mapW * mapH];

vector<vec2> parts;

fluidQ* u;
fluidQ* v;

fluidQ u_device, v_device;
vec2* parts_device;
cellType* type_device;

//-----------------------------------------------------------------------------
// device functions
//-----------------------------------------------------------------------------
__device__ int getIdx()
{
	return threadIdx.x + blockDim.x*blockIdx.x;
}

//-----------------------------------------------------------------------------
// global functions
//-----------------------------------------------------------------------------
__global__ void clearCellType(cellType* type)
{
	int idx = getIdx();

	if (idx >= mapW * mapH)
		return;

	if (type[idx] != SOLID)
		type[idx] = AIR;
}
__global__ void updateParticles(fluidQ u, fluidQ v, vec2* parts, int numParts, cellType* type)
{
	int idx = getIdx();

	if (idx >= numParts)
		return;

	float uVel = u.lerp(parts[idx].x, parts[idx].y);
	float vVel = v.lerp(parts[idx].x, parts[idx].y);

	parts[idx].x += uVel * dt;
	parts[idx].y += vVel * dt;

	if (parts[idx].x < 0)
		parts[idx].x = 0;
	if (parts[idx].y < 0)
		parts[idx].y = 0;
	if (parts[idx].x > mapW - 0.01)
		parts[idx].x = mapW - 0.01;
	if (parts[idx].y > mapH - 0.01)
		parts[idx].y = mapH - 0.01;
	
	if (type[(int)parts[idx].y * mapW + (int)parts[idx].x] != SOLID)
		type[(int)parts[idx].y * mapW + (int)parts[idx].x] = WATER;
}

//-----------------------------------------------------------------------------
// host methods
//-----------------------------------------------------------------------------
float nrand()
{
	return (float)rand() / RAND_MAX;
}

void spawnUniformParticles(int x, int y, float n)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			parts.push_back(vec2(x + nrand() / n + i / n, y + nrand() / n + j / n));
		}
	}
}

void setupParticles()
{
	for (int x = 0; x < mapW + 1; ++x)
	{
		for (int y = 0; y < mapH + 1; ++y)
		{
			u->set(x, y, 0);
			v->set(x, y, 0);
		}
	}

	for (int x = 0; x < mapW / 4; ++x)
	{
		for (int y = 0; y < mapH; ++y)
		{
			spawnUniformParticles(x, y, 2);
		}
	}
}
void createWalls()
{
	for (int y = 0; y < mapH; ++y)
	{
		type[y * mapW + 25] = SOLID;
		type[y * mapW + mapW - 10] = SOLID;
	}
	for (int i = 0; i <= 64; ++i)
	{
		type[(64 - i) * mapW + i] = SOLID;
		if (i < 64) 
			type[(64 - i - 1) * mapW + i] = SOLID;
	}
}

void applyExternal()
{
	for (int y = 0; y < mapH + 1; ++y)
		for (int x = 0; x < mapW; ++x)
			v->at(x, y) -= 9 * dt;
}
void clearCellType()
{
	int blocksize = 1024;
	int numBlocks = mapW * mapH / blocksize + 1;

	cudaMemcpy(type_device, &type, mapW * mapH * sizeof(cellType), cudaMemcpyHostToDevice);

	clearCellType <<< numBlocks, blocksize >>> (type_device);
	cudaDeviceSynchronize();
}
void updateParticles()
{
	int blocksize = 1024;
	int numBlocks = parts.size() / blocksize + 1;

	cudaMemcpy(u_device.cur, u->cur, u->w * u->h * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v_device.cur, v->cur, v->w * v->h * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(parts_device, &parts[0], parts.size() * sizeof(vec2), cudaMemcpyHostToDevice);

	updateParticles <<< numBlocks, blocksize >>> (u_device, v_device, parts_device, parts.size(), type_device);
	cudaDeviceSynchronize();

	cudaMemcpy(&parts[0], parts_device, parts.size() * sizeof(vec2), cudaMemcpyDeviceToHost);
	cudaMemcpy(&type, type_device, mapW * mapH * sizeof(cellType), cudaMemcpyDeviceToHost);
}
int layerField[mapW][mapH];
void extrapolate()
{
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			if (type[y * mapW + x] == WATER)
				layerField[x][y] = 0;
			else
				layerField[x][y] = -1;
		}
	}

	for (int i = 1; i < 6; ++i)
	{
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				if (layerField[x][y] != -1)
					continue;

				bool l, t, r, b;
				l = t = r = b = false;
				float uAvg = 0; int uN = 0;
				float vAvg = 0; int vN = 0;
				if (x > 0)
				{
					if (layerField[x - 1][y] == i - 1)
					{
						l = true;
						uAvg += u->at(x, y);
						++uN;
					}
				}
				if (y > 0)
				{
					if (layerField[x][y - 1] == i - 1)
					{
						t = true;
						vAvg += v->at(x, y);
						++vN;
					}
				}
				if (x < mapW - 1)
				{
					if (layerField[x + 1][y] == i - 1)
					{
						r = true;
						uAvg += u->at(x + 1, y);
						++uN;
					}
				}
				if (y < mapH - 1)
				{
					if (layerField[x][y + 1] == i - 1)
					{
						b = true;
						vAvg += v->at(x, y + 1);
						++vN;
					}
				}

				if (!(l || t || r || b))
					continue;

				uAvg = uAvg / max(1, uN);
				vAvg = vAvg / max(1, vN);

				if (x > 0)
				{
					if (type[y * mapW + x - 1] != WATER)
					{
						u->at(x, y) = uAvg;
					}
				}
				if (y > 0)
				{
					if (type[(y - 1) * mapW + x] != WATER)
					{
						v->at(x, y) = vAvg;
					}
				}
				if (x < mapW - 1)
				{
					if (type[y * mapW + x + 1] != WATER)
					{
						u->at(x + 1, y) = uAvg;
					}
				}
				if (y < mapH - 1)
				{
					if (type[(y + 1) * mapW + x] != WATER)
					{
						v->at(x, y + 1) = vAvg;
					}
				}

				layerField[x][y] = i;
			}
		}
	}
}

vector<int> reposistion;
int numParts[mapW][mapH];
void computeReposition()
{
	reposistion.clear();
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			numParts[x][y] = 0;
		}
	}

	for (int i = 0; i < parts.size(); ++i)
	{
		vec2 p = parts[i];
		int x = p.x;
		int y = p.y;

		if (type[y * mapW + x] != SOLID)
			numParts[x][y] += 1;

		if (type[y * mapW + x] == SOLID)
			reposistion.push_back(i);
		if (numParts[x][y] > 8)
			reposistion.push_back(i);
	}
}
void repositionParticles()
{
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			if (type[y * mapW + x] == SOLID)
				continue;

			boolean isLonely = false;
			isLonely &= (type[y * mapW + x] == AIR);
			if (x > 0)
				isLonely &= (type[y * mapW + x - 1] == WATER);
			if (y > 0)
				isLonely &= (type[(y - 1) * mapW + x] == WATER);
			if (x < mapW - 1)
				isLonely &= (type[y * mapW + x + 1] == WATER);
			if (y < mapH - 1)
				isLonely &= (type[(y + 1) * mapW + x] == WATER);

			if (numParts[x][y] == 0 && !isLonely)
				continue;

			while (reposistion.size() != 0 && numParts[x][y] < 4)
			{
				int i = reposistion.back();
				reposistion.pop_back();

				parts[i].x = x + nrand();
				parts[i].y = y + nrand();

				numParts[x][y] += 1;

				type[y * mapW + x] = WATER;
			}
		}
	}
}

struct cuspTriple
{
	int row, col;
	float amount;
};
//int countBuffer[mapW][mapH];
cusp::array1d<float, cusp::host_memory> pressure(mapW * mapH);
void project()
{
	/*{
		int counter = 0;
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				if (type[y * mapW + x] == WATER)
				{
					countBuffer[x][y] = counter;
					++counter;
				}
				else
					countBuffer[x][y] = -1;
			}
		}
	}*/
	
	cusp::array1d<float, cusp::host_memory> b(mapW * mapH);
	{
		float scale = rho / dt;
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				/*int index = countBuffer[x][y];
				if (index == -1)
					continue;*/
				int index = y * mapW + x;
				
				/*if (type[y * mapW + x] != WATER)
				{
					b[index] = 0;
					continue;
				}*/

				b[index] = scale * (u->at(x + 1, y) - u->at(x, y) +
					v->at(x, y + 1) - v->at(x, y));
			}
		}
	}

	vector<cuspTriple> data;
	{
		for (int y = 0; y < mapH; ++y)
		{
			for (int x = 0; x < mapW; ++x)
			{
				float scale = 1;
				int n = 0;

				/*if (type[y * mapW + x] != WATER)
				{
					continue;
				}*/

				if (x > 0) 
				{
					if (type[y * mapW + x - 1] != SOLID)
					{
						if (type[y * mapW + x - 1] == WATER)
						{
							cuspTriple t;
							t.row = y * mapW + x;
							t.col = y * mapW + x - 1;
							t.amount = 1;
							data.push_back(t);
						}
						++n;
					}
				}
				if (y > 0) {
					if (type[(y - 1) * mapW + x] != SOLID)
					{
						if (type[(y - 1) * mapW + x] == WATER)
						{
							cuspTriple t;
							t.row = y * mapW + x;
							t.col = (y - 1) * mapW + x;
							t.amount = 1;
							data.push_back(t);
						}
						++n;
					}
				}
				if (x < mapW - 1) {
					if (type[y * mapW + x + 1] != SOLID)
					{
						if (type[y * mapW + x + 1] == WATER)
						{
							cuspTriple t;
							t.row = y * mapW + x;
							t.col = y * mapW + x + 1;
							t.amount = 1;
							data.push_back(t);
						}
						++n;
					}
				}
				if (y < mapH - 1) {
					if (type[(y + 1) * mapW + x] != SOLID)
					{
						if (type[(y + 1) * mapW + x] == WATER)
						{
							cuspTriple t;
							t.row = y * mapW + x;
							t.col = (y + 1) * mapW + x;
							t.amount = 1;
							data.push_back(t);
						}
						++n;
					}
				}

				cuspTriple t;
				t.row = y * mapW + x;
				t.col = y * mapW + x;
				t.amount = -n;
				data.push_back(t);
			}
		}

	}
	cusp::coo_matrix<int, float, cusp::host_memory> A(mapW * mapH, mapW * mapH, data.size());
	{
		for (int i = 0; i < data.size(); ++i)
		{
			A.row_indices[i] = data[i].row;
			A.column_indices[i] = data[i].col;
			A.values[i] = data[i].amount;
		}
	}

	cusp::default_monitor<float> monitor(b, 600, 0.01, 0);
	cusp::precond::diagonal<float, cusp::host_memory> M(A);

	cusp::krylov::cg(A, pressure, b, monitor, M);
}
void applyPressure()
{
	float scale = dt / (rho);

	for (int y = 0; y < mapH; y++)
	{
		for (int x = 0; x < mapW; x++)
		{
			if (type[y * mapW + x] != WATER)
				continue;

			float p = pressure[y * mapW + x];

			u->at(x, y) -= scale * p;
			u->at(x + 1, y) += scale * p;
			v->at(x, y) -= scale * p;
			v->at(x, y + 1) += scale * p;
		}
	}
}

void enforceBoundary()
{
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			if (type[y * mapW + x] != SOLID)
				continue;

			u->at(x, y) = 0;
			u->at(x + 1, y) = 0;

			v->at(x, y) = 0;
			v->at(x, y + 1) = 0;
		}
	}

	for (int x = 0; x < mapW + 1; ++x)
	{
		v->set(x, 0, 0);
		v->set(x, v->h - 1, 0);
	}
	for (int y = 0; y < mapH + 1; ++y)
	{
		u->set(0, y, 0);
		u->set(u->w - 1, y, 0);
	}
}

void update()
{
	/*static int iter = 0;
	if (iter > 800)
	{
		return;
	}
	++iter;
	printf("%d\n", iter);*/
	
	applyExternal();

	clearCellType();
	updateParticles();
	extrapolate();

	computeReposition();
	repositionParticles();

	project();
	applyPressure();

	enforceBoundary();

	u->advect(dt, u, v);
	v->advect(dt, u, v);

	u->flip();
	v->flip();

	//printf("%d\n", (int)parts.size());

	// diagnostics
	{
		double rx, ry;
		glfwGetCursorPos(window, &rx, &ry);
		rx /= imageWidth / mapW; ry /= imageHeight / mapH;
		ry = mapH - ry;

		//printf("%i, %i \n", (int)rx, (int)ry);

		if (0 <= rx && rx < mapW &&
			0 <= ry && ry < mapH)
		{
			printf("-----------------------------\n");
			printf("V: %f, %f\n", u->lerp(rx, ry), v->lerp(rx, ry));
			printf("P: %f\n", p[(int)rx][(int)ry]);
			printf("-----------------------------\n\n");
		}

		if (0 <= rx && rx < mapW &&
			0 <= ry && ry < mapH)
		{
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT))
			{
				//ink->set(rx, ry, ink->at(rx, ry) + 1);
				//vel[(int)rx][(int)ry] += p;
				//printf("clicalsdhfalskdfuahsdkxlfajh\n");

				for (int y = 0; y < mapH; ++y)
				{
					for (int x = 0; x < mapW; ++x)
					{
						float ix = (int)rx + 0.5f;
						float iy = (int)ry + 0.5f;

						vec2 r = vec2(ix, iy) - vec2(x, y);
						float t = 500 / dot(r, r);
						r = normalize(r);

						u->at(x, y) += r.x * t;
						v->at(x, y) += r.y * t;
					}
				}

			}
		}
	}
}

void draw()
{
	glViewport(0, 0, imageWidth, imageHeight);
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glTranslatef(-1, -1, 0);
	glScalef(2, 2, 1);
	glScalef(1.f / mapW, 1.f / mapH, 1);

	/*glBegin(GL_QUADS);
	{
		for (int x = 0; x < mapW; ++x)
		{
			for (int y = 0; y < mapH; ++y)
			{
				float f = numParts[x][y] / 4.f;
				glColor3f(0, 0, f);
				//glColor3f(0, 0, 0);
				if (type[y * mapW + x] == SOLID)
					glColor3f(0, 1, 0);

				glVertex2f(x, y);
				glVertex2f(x + 1, y);
				glVertex2f(x + 1, y + 1);
				glVertex2f(x, y + 1);
			}
		}
	}
	glEnd();*/

	glColor3f(1, 0, 0);
	glBegin(GL_POINTS);
	{
		for (vec2 p : parts)
		{
			glVertex2f(p.x, p.y); 
		}
	}
	glEnd();
}

int main()
{
	srand(time(0));

	if (!glfwInit())
	{
		printf("couldn't initialize GLFW");
		return 0;
	}

	// no window hints. don't really care

	window = glfwCreateWindow(imageWidth, imageHeight, "lulz", NULL, NULL);
	if (!window)
	{
		printf("failed to open glfw window");
		return 0;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	u = new fluidQ();
	u->create(mapW + 1, mapH + 1, 0.0, 0.5, 1);
	v = new fluidQ();
	v->create(mapW + 1, mapH + 1, 0.5, 0.0, 1);
	setupParticles();

	memset(p, 0, mapW * mapH * sizeof(float));
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			type[y * mapW + x] = AIR;
		}
	}
	//createWalls();

	int maxParts = 70000;
	u_device.createCUDA(u->w, u->h, u->ox, u->oy, u->delta_x);
	v_device.createCUDA(v->w, v->h, v->ox, v->oy, v->delta_x);
	cudaMalloc((void**)&parts_device, maxParts * sizeof(vec2));
	cudaMalloc((void**)&type_device, mapW * mapH * sizeof(cellType));

	// main loop
	auto currentTime = chrono::high_resolution_clock::now();
	float accumulator = 0;
	int iter = 0;
	while (!glfwWindowShouldClose(window))
	{
		auto newTime = chrono::high_resolution_clock::now();
		float frameTime = chrono::duration_cast<chrono::milliseconds>(newTime - currentTime).count();

		if (frameTime >= dt)
		{
			update();

			++iter;
			//printf("%d\n", iter);

			currentTime = newTime;
		}

		draw();

		//Sleep(500);

		//printf("%d\n", window);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	printf("%d", window);
	glfwDestroyWindow(window);
	glfwTerminate();

	delete u;
	delete v;

	return 0;
}