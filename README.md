# MatrixMath
http://playground.arduino.cc/Code/MatrixMath

A minimal linear algebra library for Arduino. This gives you all the basics in a lean package, up to in-place matrix inversion.
Matrices are represented as simple 2D arrays, so you need to check dimension agreement manually.

A far more capable, testable, and friendly linear algebra library for Arduino is https://github.com/tomstewart89/BasicLinearAlgebra

And a related library for vector geometry manipulation is https://github.com/tomstewart89/Geometry

# MatrixMath Library - Comprehensive Arduino Integration Guide

## Overview
MatrixMath is a minimal linear algebra library for Arduino, providing essential matrix operations for embedded systems. This library handles coordinate transformations, rotation matrices, and mathematical calculations required for robotics, navigation, and sensor fusion applications.

## Applications
MatrixMath is commonly used for:
- **Coordinate Transformations**: Converting between different coordinate systems
- **Rotation Matrices**: Calculating orientation and rotation operations
- **Sensor Fusion**: Combining data from multiple sensors (GPS, compass, accelerometer)
- **Robotics**: Mathematical operations for robotic control systems
- **Navigation**: Position and orientation calculations
- **Computer Graphics**: 3D transformations and projections

## Installation

### Arduino IDE
```cpp
// Include in your sketch
#include <MatrixMath.h>
```

### PlatformIO
```ini
lib_deps = 
    MatrixMath
```

## Core Functions

### Matrix Operations

#### Matrix Multiplication
```cpp
// Multiply two matrices: C = A * B
void Matrix.Multiply(float* A, float* B, int m, int p, int n, float* C);

// Example: 3x3 rotation matrix multiplication
float rotationX[3][3] = {{1,0,0}, {0,cos(angle),-sin(angle)}, {0,sin(angle),cos(angle)}};
float rotationY[3][3] = {{cos(angle),0,sin(angle)}, {0,1,0}, {-sin(angle),0,cos(angle)}};
float result[3][3];

Matrix.Multiply((float*)rotationX, (float*)rotationY, 3, 3, 3, (float*)result);
```

#### Matrix Addition
```cpp
// Add two matrices: C = A + B
void Matrix.Add(float* A, float* B, int m, int n, float* C);

// Example: Adding offset matrices
float baseMatrix[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
float offsetMatrix[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
float resultMatrix[2][2];

Matrix.Add((float*)baseMatrix, (float*)offsetMatrix, 2, 2, (float*)resultMatrix);
```

#### Matrix Subtraction
```cpp
// Subtract matrices: C = A - B
void Matrix.Subtract(float* A, float* B, int m, int n, float* C);

// Example: Error calculation between measured and target values
float measured[3][1] = {{sensor_x}, {sensor_y}, {sensor_z}};
float target[3][1] = {{target_x}, {target_y}, {target_z}};
float error[3][1];

Matrix.Subtract((float*)measured, (float*)target, 3, 1, (float*)error);
```

#### Matrix Transpose
```cpp
// Transpose matrix: B = A^T
void Matrix.Transpose(float* A, int m, int n, float* B);

// Example: Converting row vector to column vector
float rowVector[1][3] = {{x, y, z}};
float columnVector[3][1];

Matrix.Transpose((float*)rowVector, 1, 3, (float*)columnVector);
```

#### Matrix Scaling
```cpp
// Scale matrix by scalar: B = k * A
void Matrix.Scale(float* A, int m, int n, float k);

// Example: Converting units (degrees to radians)
float angles[3][1] = {{roll_deg}, {pitch_deg}, {yaw_deg}};
Matrix.Scale((float*)angles, 3, 1, PI/180.0); // Convert to radians
```

#### Matrix Inversion
```cpp
// Invert matrix in-place: A = A^(-1)
int Matrix.Invert(float* A, int n);

// Example: Solving linear system for calibration
float calibrationMatrix[3][3] = {
    {sensor1_x_coeff, sensor1_y_coeff, sensor1_z_coeff},
    {sensor2_x_coeff, sensor2_y_coeff, sensor2_z_coeff},
    {sensor3_x_coeff, sensor3_y_coeff, sensor3_z_coeff}
};

if (Matrix.Invert((float*)calibrationMatrix, 3) == 0) {
    // Inversion successful, use inverted matrix
} else {
    // Matrix is singular, handle error
}
```

## Common Use Cases

### Coordinate System Transformation
```cpp
// Transform from one coordinate system to another
void transformCoordinates(float input[3], float transformMatrix[3][3], float output[3]) {
    Matrix.Multiply((float*)transformMatrix, input, 3, 3, 1, output);
}

// Example: ECEF to ENU coordinate transformation
void ecefToEnu(float ecef[3], float refLat, float refLon, float enu[3]) {
    float sinLat = sin(refLat * PI / 180.0);
    float cosLat = cos(refLat * PI / 180.0);
    float sinLon = sin(refLon * PI / 180.0);
    float cosLon = cos(refLon * PI / 180.0);
    
    // ENU transformation matrix
    float transformMatrix[3][3] = {
        {-sinLon, cosLon, 0},
        {-sinLat*cosLon, -sinLat*sinLon, cosLat},
        {cosLat*cosLon, cosLat*sinLon, sinLat}
    };
    
    transformCoordinates(ecef, transformMatrix, enu);
}
```

### Rotation Matrix Operations
```cpp
// Create rotation matrices for 3D rotations
void createRotationMatrixX(float angle, float rotMatrix[3][3]) {
    float c = cos(angle);
    float s = sin(angle);
    
    float rotX[3][3] = {
        {1, 0, 0},
        {0, c, -s},
        {0, s, c}
    };
    
    memcpy(rotMatrix, rotX, sizeof(rotX));
}

void createRotationMatrixY(float angle, float rotMatrix[3][3]) {
    float c = cos(angle);
    float s = sin(angle);
    
    float rotY[3][3] = {
        {c, 0, s},
        {0, 1, 0},
        {-s, 0, c}
    };
    
    memcpy(rotMatrix, rotY, sizeof(rotY));
}

void createRotationMatrixZ(float angle, float rotMatrix[3][3]) {
    float c = cos(angle);
    float s = sin(angle);
    
    float rotZ[3][3] = {
        {c, -s, 0},
        {s, c, 0},
        {0, 0, 1}
    };
    
    memcpy(rotMatrix, rotZ, sizeof(rotZ));
}

// Combine multiple rotations
void combineRotations(float roll, float pitch, float yaw, float finalRotation[3][3]) {
    float rotX[3][3], rotY[3][3], rotZ[3][3];
    float temp[3][3];
    
    createRotationMatrixX(roll, rotX);
    createRotationMatrixY(pitch, rotY);
    createRotationMatrixZ(yaw, rotZ);
    
    // Combine: R = Rz * Ry * Rx
    Matrix.Multiply((float*)rotZ, (float*)rotY, 3, 3, 3, (float*)temp);
    Matrix.Multiply((float*)temp, (float*)rotX, 3, 3, 3, (float*)finalRotation);
}
```

### Sensor Calibration
```cpp
// Apply calibration matrix to sensor readings
void applySensorCalibration(float rawReadings[3], float calibMatrix[3][3], float calibratedReadings[3]) {
    Matrix.Multiply((float*)calibMatrix, rawReadings, 3, 3, 1, calibratedReadings);
}

// Example: Magnetometer calibration
void calibrateMagnetometer(float magRaw[3], float magCalibrated[3]) {
    // Pre-computed calibration matrix from calibration procedure
    float magCalibMatrix[3][3] = {
        {1.02, 0.03, -0.01},
        {0.03, 0.98, 0.02},
        {-0.01, 0.02, 1.01}
    };
    
    applySensorCalibration(magRaw, magCalibMatrix, magCalibrated);
}

// Example: Accelerometer calibration
void calibrateAccelerometer(float accelRaw[3], float accelCalibrated[3]) {
    float accelCalibMatrix[3][3] = {
        {1.001, 0.002, 0.001},
        {0.002, 0.999, 0.003},
        {0.001, 0.003, 1.002}
    };
    
    applySensorCalibration(accelRaw, accelCalibMatrix, accelCalibrated);
}
```

### Kalman Filter Implementation
```cpp
// Basic Kalman filter matrices
struct KalmanFilter {
    float state[4];           // State vector [x, y, vx, vy]
    float covariance[4][4];   // Covariance matrix
    float processNoise[4][4]; // Process noise matrix
    float measurementNoise[2][2]; // Measurement noise matrix
};

void kalmanPredict(KalmanFilter* kf, float dt) {
    // State transition matrix
    float F[4][4] = {
        {1, 0, dt, 0},
        {0, 1, 0, dt},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    
    // Predict state: x = F * x
    float newState[4];
    Matrix.Multiply((float*)F, kf->state, 4, 4, 1, newState);
    memcpy(kf->state, newState, sizeof(newState));
    
    // Predict covariance: P = F * P * F^T + Q
    float temp[4][4], FT[4][4];
    Matrix.Transpose((float*)F, 4, 4, (float*)FT);
    Matrix.Multiply((float*)F, (float*)kf->covariance, 4, 4, 4, (float*)temp);
    Matrix.Multiply((float*)temp, (float*)FT, 4, 4, 4, (float*)kf->covariance);
    Matrix.Add((float*)kf->covariance, (float*)kf->processNoise, 4, 4, (float*)kf->covariance);
}

void kalmanUpdate(KalmanFilter* kf, float measurement[2]) {
    // Measurement matrix H = [1 0 0 0; 0 1 0 0]
    float H[2][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}};
    float HT[4][2];
    Matrix.Transpose((float*)H, 2, 4, (float*)HT);
    
    // Innovation: y = z - H * x
    float predicted[2];
    Matrix.Multiply((float*)H, kf->state, 2, 4, 1, predicted);
    float innovation[2];
    Matrix.Subtract(measurement, predicted, 2, 1, innovation);
    
    // Innovation covariance: S = H * P * H^T + R
    float temp[2][4], S[2][2];
    Matrix.Multiply((float*)H, (float*)kf->covariance, 2, 4, 4, (float*)temp);
    Matrix.Multiply((float*)temp, (float*)HT, 2, 4, 2, (float*)S);
    Matrix.Add((float*)S, (float*)kf->measurementNoise, 2, 2, (float*)S);
    
    // Kalman gain: K = P * H^T * S^(-1)
    if (Matrix.Invert((float*)S, 2) == 0) {
        float K[4][2];
        Matrix.Multiply((float*)kf->covariance, (float*)HT, 4, 4, 2, (float*)temp);
        Matrix.Multiply((float*)temp, (float*)S, 4, 2, 2, (float*)K);
        
        // Update state: x = x + K * y
        float correction[4];
        Matrix.Multiply((float*)K, innovation, 4, 2, 1, correction);
        Matrix.Add(kf->state, correction, 4, 1, kf->state);
        
        // Update covariance: P = (I - K * H) * P
        float I[4][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
        float KH[4][4];
        Matrix.Multiply((float*)K, (float*)H, 4, 2, 4, (float*)KH);
        Matrix.Subtract((float*)I, (float*)KH, 4, 4, (float*)KH);
        Matrix.Multiply((float*)KH, (float*)kf->covariance, 4, 4, 4, (float*)kf->covariance);
    }
}
```



### History

2016 Vasilis Georgitzikis / Package code into easy-install Arduino library. 

2013 Charlie Matlack / Add in-place matrix inverse function, very helpful in limited memory environment of Arduino, and general clean-up.

Unknown original author, Arduino form user RobH45345, posted code to Arduino playground.

### License

GPL2. 
