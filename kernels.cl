#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t interpolationSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t hpSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define LPOS(pos) pos.x+pos.y*get_global_size(0)+pos.z*get_global_size(0)*get_global_size(1)


// Intialize 3D image to 0
__kernel void init3DImage(
    __write_only image3d_t image
    ) {
    write_imagei(image, (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0), 0);
}

// Intialize 2D image to 0
__kernel void init2DImage(
    __write_only image2d_t image
    ) {
    write_imagei(image, (int2)(get_global_id(0), get_global_id(1)), 0);
}

// Intialize int buffer to 0
__kernel void initIntBuffer(
    __global int * buffer
    ) {
    buffer[get_global_id(0)] = 0;
}

// Intialize char buffer to 0
__kernel void initCharBuffer(
    __global char * buffer
    ) {
    buffer[get_global_id(0)] = 0;
}

// Intialize int buffer to its ID
__kernel void initIntBufferID(
    __global int * buffer,
    __private int sum
    ) {
    int id = get_global_id(0); 
    if(id >= sum)
        id = 0;
    buffer[id] = id;
}



__constant int4 cubeOffsets2D[4] = {
    {0, 0, 0, 0},
    {0, 1, 0, 0},
    {1, 0, 0, 0},
    {1, 1, 0, 0},
};

__constant int4 cubeOffsets[8] = {
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 0, 1, 0},
    {1, 0, 1, 0},
    {0, 1, 0, 0},
    {1, 1, 0, 0},
    {0, 1, 1, 0},
    {1, 1, 1, 0},
};

__kernel void constructHPLevel3D(
    __read_only image3d_t readHistoPyramid,
    __write_only image3d_t writeHistoPyramid
    ) { 

    int4 writePos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int4 readPos = writePos*2;
    int writeValue = read_imagei(readHistoPyramid, hpSampler, readPos).x + // 0
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[1]).x + // 1
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[2]).x + // 2
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[3]).x + // 3
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[4]).x + // 4
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[5]).x + // 5
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[6]).x + // 6
    read_imagei(readHistoPyramid, hpSampler, readPos+cubeOffsets[7]).x; // 7

    write_imagei(writeHistoPyramid, writePos, writeValue);
}

__kernel void constructHPLevel2D(
    __read_only image2d_t readHistoPyramid,
    __write_only image2d_t writeHistoPyramid
    ) { 

    int2 writePos = {get_global_id(0), get_global_id(1)};
    int2 readPos = writePos*2;
    int writeValue = 
        read_imagei(readHistoPyramid, hpSampler, readPos).x + 
        read_imagei(readHistoPyramid, hpSampler, readPos+(int2)(1,0)).x + 
        read_imagei(readHistoPyramid, hpSampler, readPos+(int2)(0,1)).x + 
        read_imagei(readHistoPyramid, hpSampler, readPos+(int2)(1,1)).x;

    write_imagei(writeHistoPyramid, writePos, writeValue);
}

int3 scanHPLevel2D(int target, __read_only image2d_t hp, int3 current) {

    int4 neighbors = {
        read_imagei(hp, hpSampler, current.xy).x,
        read_imagei(hp, hpSampler, current.xy + (int2)(0,1)).x,
        read_imagei(hp, hpSampler, current.xy + (int2)(1,0)).x,
        0
    };

    int acc = current.z + neighbors.s0;
    int4 cmp;
    cmp.s0 = acc <= target;
    acc += neighbors.s1;
    cmp.s1 = acc <= target;
    acc += neighbors.s2;
    cmp.s2 = acc <= target;

    current += cubeOffsets2D[(cmp.s0+cmp.s1+cmp.s2)].xyz;
    current.x = current.x*2;
    current.y = current.y*2;
    current.z = current.z +
    cmp.s0*neighbors.s0 +
    cmp.s1*neighbors.s1 +
    cmp.s2*neighbors.s2; 
    return current;

}


int4 scanHPLevel3D(int target, __read_only image3d_t hp, int4 current) {

    int8 neighbors = {
        read_imagei(hp, hpSampler, current).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[1]).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[2]).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[3]).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[4]).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[5]).x,
        read_imagei(hp, hpSampler, current + cubeOffsets[6]).x,
        0
    };

    int acc = current.s3 + neighbors.s0;
    int8 cmp;
    cmp.s0 = acc <= target;
    acc += neighbors.s1;
    cmp.s1 = acc <= target;
    acc += neighbors.s2;
    cmp.s2 = acc <= target;
    acc += neighbors.s3;
    cmp.s3 = acc <= target;
    acc += neighbors.s4;
    cmp.s4 = acc <= target;
    acc += neighbors.s5;
    cmp.s5 = acc <= target;
    acc += neighbors.s6;
    cmp.s6 = acc <= target;


    current += cubeOffsets[(cmp.s0+cmp.s1+cmp.s2+cmp.s3+cmp.s4+cmp.s5+cmp.s6)];
    current.s0 = current.s0*2;
    current.s1 = current.s1*2;
    current.s2 = current.s2*2;
    current.s3 = current.s3 +
    cmp.s0*neighbors.s0 +
    cmp.s1*neighbors.s1 +
    cmp.s2*neighbors.s2 +
    cmp.s3*neighbors.s3 +
    cmp.s4*neighbors.s4 +
    cmp.s5*neighbors.s5 +
    cmp.s6*neighbors.s6; 
    return current;

}

int4 traverseHP3D(
    int target,
    int HP_SIZE,
    image3d_t hp0,
    image3d_t hp1,
    image3d_t hp2,
    image3d_t hp3,
    image3d_t hp4,
    image3d_t hp5,
    image3d_t hp6,
    image3d_t hp7,
    image3d_t hp8,
    image3d_t hp9
    ) {
    int4 position = {0,0,0,0}; // x,y,z,sum
    if(HP_SIZE > 512)
    position = scanHPLevel3D(target, hp9, position);
    if(HP_SIZE > 256)
    position = scanHPLevel3D(target, hp8, position);
    if(HP_SIZE > 128)
    position = scanHPLevel3D(target, hp7, position);
    if(HP_SIZE > 64)
    position = scanHPLevel3D(target, hp6, position);
    if(HP_SIZE > 32)
    position = scanHPLevel3D(target, hp5, position);
    if(HP_SIZE > 16)
    position = scanHPLevel3D(target, hp4, position);
    if(HP_SIZE > 8)
    position = scanHPLevel3D(target, hp3, position);
    position = scanHPLevel3D(target, hp2, position);
    position = scanHPLevel3D(target, hp1, position);
    position = scanHPLevel3D(target, hp0, position);
    position.x = position.x / 2;
    position.y = position.y / 2;
    position.z = position.z / 2;
    return position;
}

int2 traverseHP2D(
    int target,
    int HP_SIZE,
    image2d_t hp0,
    image2d_t hp1,
    image2d_t hp2,
    image2d_t hp3,
    image2d_t hp4,
    image2d_t hp5,
    image2d_t hp6,
    image2d_t hp7,
    image2d_t hp8,
    image2d_t hp9,
    image2d_t hp10,
    image2d_t hp11,
    image2d_t hp12
    ) {
    int3 position = {0,0,0};
    if(HP_SIZE > 4096)
    position = scanHPLevel2D(target, hp12, position);
    if(HP_SIZE > 2048)
    position = scanHPLevel2D(target, hp11, position);
    if(HP_SIZE > 1024)
    position = scanHPLevel2D(target, hp10, position);
    if(HP_SIZE > 512)
    position = scanHPLevel2D(target, hp9, position);
    if(HP_SIZE > 256)
    position = scanHPLevel2D(target, hp8, position);
    if(HP_SIZE > 128)
    position = scanHPLevel2D(target, hp7, position);
    if(HP_SIZE > 64)
    position = scanHPLevel2D(target, hp6, position);
    if(HP_SIZE > 32)
    position = scanHPLevel2D(target, hp5, position);
    if(HP_SIZE > 16)
    position = scanHPLevel2D(target, hp4, position);
    if(HP_SIZE > 8)
    position = scanHPLevel2D(target, hp3, position);
    position = scanHPLevel2D(target, hp2, position);
    position = scanHPLevel2D(target, hp1, position);
    position = scanHPLevel2D(target, hp0, position);
    position.x = position.x / 2;
    position.y = position.y / 2;
    return position.xy;
}


__kernel void createPositions3D(
        __global int * positions,
        __private int HP_SIZE,
        __private int sum,
        __read_only image3d_t hp0, // Largest HP
        __read_only image3d_t hp1,
        __read_only image3d_t hp2,
        __read_only image3d_t hp3,
        __read_only image3d_t hp4,
        __read_only image3d_t hp5
        ,__read_only image3d_t hp6
        ,__read_only image3d_t hp7
        ,__read_only image3d_t hp8
        ,__read_only image3d_t hp9
    ) {
    int target = get_global_id(0);
    if(target >= sum)
        target = 0;
    int4 pos = traverseHP3D(target,HP_SIZE,hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9);
    vstore3(pos.xyz, target, positions);
}

__kernel void createPositions2D(
        __global int * positions,
        __private int HP_SIZE,
        __private int sum,
        __read_only image2d_t hp0, // Largest HP
        __read_only image2d_t hp1,
        __read_only image2d_t hp2,
        __read_only image2d_t hp3,
        __read_only image2d_t hp4,
        __read_only image2d_t hp5
        ,__read_only image2d_t hp6
        ,__read_only image2d_t hp7
        ,__read_only image2d_t hp8
        ,__read_only image2d_t hp9
        ,__read_only image2d_t hp10
        ,__read_only image2d_t hp11
        ,__read_only image2d_t hp12
    ) {
    int target = get_global_id(0);
    if(target >= sum)
        target = 0;
    int2 pos = traverseHP2D(target,HP_SIZE,hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9,hp10,hp11,hp12);
    vstore2(pos, target, positions);
}

__kernel void linkLengths(
        __global int const * restrict positions,
        __write_only image2d_t lengths
        ) {
    const float3 xa = convert_float3(vload3(get_global_id(0), positions));
    const float3 xb = convert_float3(vload3(get_global_id(1), positions));

    write_imagef(lengths, (int2)(get_global_id(0), get_global_id(1)), distance(xa,xb));
}

__kernel void compact(
        __read_only image2d_t lengths,
        volatile __global int * incs,
        __write_only image2d_t compacted_lengths,
        __private float maxDistance
        ) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float length = read_imagef(lengths, sampler, (int2)(i,j)).x;
    if(length < maxDistance && length > 0.0f) {
        volatile int nr = atomic_inc(&(incs[i]));
        write_imagef(compacted_lengths, (int2)(i,nr), (float4)(length, j, 0, 0));
    }
}

__kernel void linkCenterpoints(
        __read_only image3d_t TDF,
        __read_only image3d_t radius,
        __global int const * restrict positions,
        __write_only image2d_t edges,
        __read_only image3d_t intensity,
        __read_only image2d_t compacted_lengths,
        __private int sum,
        __private float minAvgTDF,
        __private float maxDistance
    ) {
    int id = get_global_id(0);
    if(id >= sum)
        id = 0;
    float3 xa = convert_float3(vload3(id, positions));

    int2 bestPair;
    float shortestDistance = maxDistance*2;
    bool validPairFound = false;
    for(int i = 0; i < sum; i++) {
        float2 cl = read_imagef(compacted_lengths, sampler, (int2)(id,i)).xy;

        // reached the end?
        if(cl.x == 0.0f)
            break;

    float3 xb = convert_float3(vload3(cl.y, positions));
    int db = round(cl.x);
    if(db >= shortestDistance)
        continue;
    for(int j = 0; j < i; j++) {
        float2 cl2  = read_imagef(compacted_lengths, sampler, (int2)(id,j)).xy;
        if(cl2.y == cl.y) 
            continue;

        // reached the end?
        if(cl2.x == 0.0f)
            break;
    float3 xc = convert_float3(vload3(cl2.y, positions));

    // Check distance between xa and xb
    int dc = round(cl2.x);

    float minTDF = 0.0f;
    float maxVarTDF = 1.005f;
    float maxIntensity = 1.3f;
    float maxAvgIntensity = 1.2f;
    float maxVarIntensity = 1.005f;

    if(db+dc < shortestDistance) {
        // Check angle
        float3 ab = (xb-xa);
        float3 ac = (xc-xa);
        float angle = acos(dot(normalize(ab), normalize(ac)));
        if(angle < 2.0f) // 120 degrees
            continue;
        // Check TDF
        float avgTDF = 0.0f;
        float avgIntensity = 0.0f;
        bool invalid = false;
        //printf("%d - %d \n", db, dc);
        for(int k = 0; k <= db; k++) {
            float alpha = (float)k/db;
            float3 p = xa+ab*alpha;
            float t = read_imagef(TDF, interpolationSampler, p.xyzz).x; 
            float i = read_imagef(intensity, interpolationSampler, p.xyzz).x; 
            avgIntensity += i;
            avgTDF += t;
            if(i > maxIntensity || t < minTDF) {
                invalid = true;
                break;
            }
        }
        if(invalid)
            continue;
        avgTDF /= db+1;
        avgIntensity /= db+1;
        if(avgTDF < minAvgTDF)
            continue;
        if(avgIntensity > maxAvgIntensity)
            continue;

        float varTDF = 0.0f;
        float varIntensity = 0.0f;
        for(int k = 0; k <= db; k++) {
            float alpha = (float)k/db;
            float3 p = xa+ab*alpha;
            float t = read_imagef(TDF, interpolationSampler, p.xyzz).x; 
            float i = read_imagef(intensity, interpolationSampler, p.xyzz).x; 
            varIntensity += (i-avgIntensity)*(i-avgIntensity);
            varTDF += (t-avgTDF)*(t-avgTDF);
            if(i > maxIntensity || t < minTDF) {
                invalid = true;
                break;
            }
        }
        if(invalid)
            continue;

        if(db > 4 && varIntensity / (db+1) > maxVarIntensity)
            continue;
        if(db > 4 && varTDF / (db+1) > maxVarTDF)
            continue;

        avgTDF = 0.0f;
        avgIntensity = 0.0f;
        varTDF = 0.0f;
        varIntensity = 0.0f;
        for(int k = 0; k <= dc; k++) {
            float alpha = (float)k/dc;
            float3 p = xa+ac*alpha;
            float t = read_imagef(TDF, interpolationSampler, p.xyzz).x; 
            float i = read_imagef(intensity, interpolationSampler, p.xyzz).x; 
            avgTDF += t;
            avgIntensity += i;
        }
        avgTDF /= dc+1;
        avgIntensity /= dc+1;

        if(avgTDF < minAvgTDF)
            continue;

        if(avgIntensity > maxAvgIntensity)
            continue;

        for(int k = 0; k <= db; k++) {
            float alpha = (float)k/db;
            float3 p = xa+ab*alpha;
            float t = read_imagef(TDF, interpolationSampler, p.xyzz).x; 
            float i = read_imagef(intensity, interpolationSampler, p.xyzz).x; 
            varIntensity += (i-avgIntensity)*(i-avgIntensity);
            varTDF += (t-avgTDF)*(t-avgTDF);
        }

        if(dc > 4 && varIntensity / (dc+1) > maxVarIntensity)
            continue;
        if(dc > 4 && varTDF / (dc+1) > maxVarTDF)
            continue;
        //printf("avg i: %f\n", avgIntensity );
        //printf("avg tdf: %f\n", avgTDF );
        //printf("var i: %f\n", varIntensity / (dc+1));
        //printf("var tdf: %f\n", varTDF / (dc+1));

        validPairFound = true;
        bestPair.x = cl.y;
        bestPair.y = cl2.y;
        shortestDistance = db+dc;
    }
    }}

    if(validPairFound) {
        // Store edges
        int2 edge = {id, bestPair.x};
        int2 edge2 = {id, bestPair.y};
        write_imagei(edges, edge, 1);
        write_imagei(edges, edge2, 1);
    }
}

__kernel void graphComponentLabeling(
        __global int const * restrict edges,
        volatile __global int * C,
        __global int * m,
        __private int sum
        ) {
    int id = get_global_id(0);
    if(id >= sum)
        id = 0;
    int2 edge = vload2(id, edges);
    const int ca = C[edge.x];
    const int cb = C[edge.y];

    // Find the smallest C value and store in C in the others
    if(ca == cb) {
        return;
    } else {
        if(ca < cb) {
            // ca is smallest
            volatile int i = atomic_min(&C[edge.y], ca);
        } else {
            // cb is smallest
            volatile int i = atomic_min(&C[edge.x], cb);
        }
        m[0] = 1; // register change
    }
}

__kernel void calculateTreeLength(
        __global int const * restrict C,
        volatile __global int * S
    ) {
    const int id = get_global_id(0);
    atomic_inc(&S[C[id]]);
}

__kernel void removeSmallTrees(
        __global int const * restrict edges,
        __global int const * restrict vertices,
        __global int const * restrict C,
        __global int const * restrict S,
        __private int minTreeLength,
        __write_only image3d_t centerlines
    ) {
   // Find the edges that are part of the large trees 
    const int id = get_global_id(0);
    int2 edge = vload2(id, edges);
    const int ca = C[edge.x];
    if(S[ca] >= minTreeLength) {
        const float3 xa = convert_float3(vload3(edge.x, vertices));
        const float3 xb = convert_float3(vload3(edge.y, vertices));
        int l = round(length(xb-xa));
        for(int i = 0; i < l; i++) {
            const float alpha = (float)i/l;
            write_imagei(centerlines, convert_int3(round(xa+(xb-xa)*alpha)).xyzz, 1);
        }
    }
}

__kernel void combine(
    __global float * TDFsmall,
    __global float * radiusSmall,
    __global float * TDFlarge,
    __global float * radiusLarge
    ) {
    uint i = get_global_id(0);
    if(TDFlarge[i] < TDFsmall[i]) {
        TDFlarge[i] = TDFsmall[i];
        radiusLarge[i] = radiusSmall[i];
    }
}

__kernel void initGrowing(
	__read_only image3d_t centerline,
	__write_only image3d_t initSegmentation
	) {
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    if(read_imagei(centerline, sampler, pos).x == 1) {
	
        for(int a = -1; a < 2; a++) {
        for(int b = -1; b < 2; b++) {
        for(int c = -1; c < 2; c++) {
            int4 n;
            n.x = pos.x + a;
            n.y = pos.y + b;
            n.z = pos.z + c;
	    if(read_imagei(centerline, sampler, n).x == 0)
	    write_imagei(initSegmentation, n, 2);
        }}}
	}

}



__kernel void grow(
	__read_only image3d_t currentSegmentation,
	__read_only image3d_t gvf,
	__write_only image3d_t nextSegmentation,
	__global int * stop
	) {

    int4 X = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    char value = read_imagei(currentSegmentation, sampler, X).x;
    // value of 2, means to check it, 1 means it is already accepted
    if(value == 1) {
	    write_imagei(nextSegmentation, X, 1);
    }else if(value == 2){
	float FNXw = read_imagef(gvf, sampler, X).w;

	bool continueGrowing = false;
	for(int a = -1; a < 2; a++) {
	for(int b = -1; b < 2; b++) {
	for(int c = -1; c < 2; c++) {
	    if(a == 0 && b == 0 && c == 0)
		continue;

	    int4 Y;
	    Y.x = X.x + a;
	    Y.y = X.y + b;
	    Y.z = X.z + c;
	    
	    char valueY = read_imagei(currentSegmentation, sampler, Y).x;
	    if(valueY != 1) {
		float4 FNY = read_imagef(gvf, sampler, Y);
		FNY.x /= FNY.w;
		FNY.y /= FNY.w;
		FNY.z /= FNY.w;
	    if(FNY.w > FNXw || FNXw < 0.1f) {

		int4 Z;
		float maxDotProduct = -2.0f;
		for(int a2 = -1; a2 < 2; a2++) {
		for(int b2 = -1; b2 < 2; b2++) {
		for(int c2 = -1; c2 < 2; c2++) {
		    if(a2 == 0 && b2 == 0 && c2 == 0)
			continue;
		    int4 Zc;
		    Zc.x = Y.x+a2;
		    Zc.y = Y.y+b2;
		    Zc.z = Y.z+c2;
		    float3 YZ;
		    YZ.x = Zc.x-Y.x;
		    YZ.y = Zc.y-Y.y;
		    YZ.z = Zc.z-Y.z;
		    YZ = normalize(YZ);
		    if(dot(FNY.xyz, YZ) > maxDotProduct) {
			maxDotProduct = dot(FNY.xyz, YZ);
			Z = Zc;
		    }
		}}}

		if(Z.x == X.x && Z.y == X.y && Z.z == X.z) {
			write_imagei(nextSegmentation, X, 1);
			write_imagei(nextSegmentation, Y, 2);
			continueGrowing = true;
		}
	    }}
	}}}

	if(continueGrowing) {
		// Added new items to list (values of 2)
		stop[0] = 0;
	} else {
		// X was not accepted
	write_imagei(nextSegmentation, X, 0);
	}
}
}

float3 gradientNormalized(
        __read_only image3d_t volume,   // Volume to perform gradient on
        int4 pos,                       // Position to perform gradient on
        int volumeComponent,            // The volume component to perform gradient on: 0, 1 or 2
        int dimensions                  // The number of dimensions to perform gradient in: 1, 2 or 3
    ) {
    float f100, f_100, f010, f0_10, f001, f00_1;
    switch(volumeComponent) {
        case 0:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).x; 
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).x;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).x; 
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).x;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).x;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).x;
    }
    break;
        case 1:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).y;
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).y;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).y;
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).y;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).y;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).y;
    }
    break;
        case 2:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).z;
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).z;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).z;
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).z;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).z;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).z;
    }
    break;
    }

    float3 grad = {
        0.5f*(f100/read_imagef(volume, sampler, pos+(int4)(1,0,0,0)).w-f_100/read_imagef(volume, sampler, pos-(int4)(1,0,0,0)).w), 
        0.5f*(f010/read_imagef(volume, sampler, pos+(int4)(0,1,0,0)).w-f0_10/read_imagef(volume, sampler, pos-(int4)(0,1,0,0)).w),
        0.5f*(f001/read_imagef(volume, sampler, pos+(int4)(0,0,1,0)).w-f00_1/read_imagef(volume, sampler, pos-(int4)(0,0,1,0)).w)
    };


    return grad;
}

float3 gradient(
        __read_only image3d_t volume,   // Volume to perform gradient on
        int4 pos,                       // Position to perform gradient on
        int volumeComponent,            // The volume component to perform gradient on: 0, 1 or 2
        int dimensions                  // The number of dimensions to perform gradient in: 1, 2 or 3
    ) {
    float f100, f_100, f010, f0_10, f001, f00_1;
    switch(volumeComponent) {
        case 0:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).x; 
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).x;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).x; 
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).x;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).x;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).x;
    }
    break;
        case 1:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).y;
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).y;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).y;
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).y;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).y;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).y;
    }
    break;
        case 2:
    f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).z;
    f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).z;
    if(dimensions > 1) {
    f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).z;
    f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).z;
    }
    if(dimensions > 2) {
    f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).z;
    f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).z;
    }
    break;
    }

    float3 grad = {
        0.5f*(f100-f_100), 
        0.5f*(f010-f0_10),
        0.5f*(f001-f00_1)
    };


    return grad;
}



__kernel void cropDataset(
        __read_only image3d_t volume,
        __global short * scanLinesInside,
        __private int sliceDirection
    ) {
    short HUlimit = -150;
    int Wlimit = 30;
    int Blimit = 30;
    int sliceNr = get_global_id(0);
    short scanLines = 0;   
    int scanLineSize, scanLineElementSize;

    if(sliceDirection == 0) {
        scanLineSize = get_image_height(volume);
        scanLineElementSize = get_image_depth(volume);
    } else if(sliceDirection == 1) {
        scanLineSize = get_image_width(volume);
        scanLineElementSize = get_image_depth(volume);
    } else {
        scanLineSize = get_image_height(volume);
        scanLineElementSize = get_image_width(volume);
    }

    for(int scanLine = 0; scanLine < scanLineSize; scanLine++) {
        int currentWcount = 0,
            currentBcount = 0,
            detectedBlackAreas = 0,
            detectedWhiteAreas = 0;
          
        for(int scanLineElement = 0; scanLineElement < scanLineElementSize; scanLineElement ++) {
            int4 pos;
            if(sliceDirection == 0) {
                pos.x = sliceNr;
                pos.y = scanLine;
                pos.z = scanLineElement;
            } else if(sliceDirection == 1) {
                pos.x = scanLine;
                pos.y = sliceNr;
                pos.z = scanLineElement;
            } else {
                pos.x = scanLineElement;
                pos.y = scanLine;
                pos.z = sliceNr;
            }

            if(read_imagei(volume, sampler, pos).x > HUlimit) {
                if(currentWcount == Wlimit) {
                    detectedWhiteAreas++;
                    currentBcount = 0;
                }
                currentWcount++;
            } else {
                if(currentBcount == Blimit) {
                    detectedBlackAreas++;
                    currentWcount = 0;
                }
                currentBcount++;
            }
        }
        if((detectedWhiteAreas == 2 && detectedBlackAreas == 1) || 
            (detectedBlackAreas > 1 && detectedWhiteAreas > 1)) {
            scanLines++;
        } // End scan line
    }
    scanLinesInside[sliceNr] = scanLines;
} 

__kernel void dilate(
        __read_only image3d_t volume, 
        __write_only image3d_t result
        ) {
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    if(read_imagei(volume, sampler, pos).x == 1) {
    for(int a = -1; a < 2 ; a++) { 
        for(int b = -1; b < 2 ; b++) { 
            for(int c = -1; c < 2 ; c++) { 
                int4 nPos = pos + (int4)(a,b,c,0);
		write_imagei(result, nPos, 1);
            }
        }
    }
    }
}

__kernel void erode(
        __read_only image3d_t volume, 
        __write_only image3d_t result
        ) {
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    int value = read_imagei(volume, sampler, pos).x;
    if(value == 1) {

    bool keep = true;
    for(int a = -1; a < 2 ; a++) { 
        for(int b = -1; b < 2 ; b++) { 
            for(int c = -1; c < 2 ; c++) { 
                keep = (read_imagei(volume, sampler, pos + (int4)(a,b,c,0)).x == 1 && keep);
            }
        }
    }
    	write_imagei(result, pos, keep ? 1 : 0);
    } else {
    	write_imagei(result, pos, 0);
    }
}

__kernel void toFloat(
        __read_only image3d_t volume,
        __write_only image3d_t processedVolume,
        __private float minimum,
        __private float maximum,
        __private int type
        ) {
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    float v;
    if(type == 1) {
        v = read_imagei(volume, sampler, pos).x;
    } else if(type == 2) {
        v = read_imageui(volume, sampler, pos).x;
    } else {
        v = read_imagef(volume, sampler, pos).x;
    }

    v = v > maximum ? maximum : v;
    v = v < minimum ? minimum : v;

    // Convert to floating point representation 0 to 1
    float value = (float)(v - minimum) / (float)(maximum - minimum);

    // Store value
    write_imagef(processedVolume, pos, value);
}

__kernel void blurVolumeWithGaussian(
        __read_only image3d_t volume,
        __write_only image3d_t blurredVolume,
        __private int maskSize,
        __constant float * mask
    ) {

    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    int size = maskSize*2+1;
    
    // Collect neighbor values and multiply with gaussian
    float sum = 0.0f;
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    for(int c = -maskSize; c < maskSize+1; c++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            for(int a = -maskSize; a < maskSize+1; a++) {
                sum += mask[a+maskSize+(b+maskSize)*size+(c+maskSize)*size*size]*
                    read_imagef(volume, sampler, pos + (int4)(a,b,c,0)).x; 
            }
        }
    }

    write_imagef(blurredVolume, pos, sum);
}

__kernel void createVectorField(
        __read_only image3d_t volume, 
        __write_only image3d_t vectorField, 
        __private float Fmax,
        __private int vsign
        ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    // Gradient of volume
    float4 F; 
    F.xyz = vsign*gradient(volume, pos, 0, 3); // The sign here is important
    F.w = 0.0f;

    // Fmax normalization
    const float l = length(F);
    F = l < Fmax ? F/(Fmax) : F / (l);
    F.w = 1.0f;

    // Store vector field
    //vstore4(F, LPOS(pos), vectorField);
    write_imagef(vectorField, pos, F);
}

// Forward declaration of eigen_decomp function
void eigen_decomposition(float M[3][3], float V[3][3], float e[3]);

__constant float cosValues[32] = {1.0f, 0.540302f, -0.416147f, -0.989992f, -0.653644f, 0.283662f, 0.96017f, 0.753902f, -0.1455f, -0.91113f, -0.839072f, 0.0044257f, 0.843854f, 0.907447f, 0.136737f, -0.759688f, -0.957659f, -0.275163f, 0.660317f, 0.988705f, 0.408082f, -0.547729f, -0.999961f, -0.532833f, 0.424179f, 0.991203f, 0.646919f, -0.292139f, -0.962606f, -0.748058f, 0.154251f, 0.914742f};
__constant float sinValues[32] = {0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f, 0.989358f, 0.412118f, -0.544021f, -0.99999f, -0.536573f, 0.420167f, 0.990607f, 0.650288f, -0.287903f, -0.961397f, -0.750987f, 0.149877f, 0.912945f, 0.836656f, -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f, -0.663634f, -0.988032f, -0.404038f};

__kernel void circleFittingTDF(
        __read_only image3d_t vectorField,
        __global float * T,
        __global float * Radius,
        __private float rMin,
        __private float rMax,
        __private float rStep
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    // Find Hessian Matrix
    float3 Fx, Fy, Fz;
    if(rMax < 4) {
	    Fx = gradient(vectorField, pos, 0, 1);
	    Fy = gradient(vectorField, pos, 1, 2);
	    Fz = gradient(vectorField, pos, 2, 3);
    } else {
	    Fx = gradientNormalized(vectorField, pos, 0, 1);
	    Fy = gradientNormalized(vectorField, pos, 1, 2);
	    Fz = gradientNormalized(vectorField, pos, 2, 3);
    }


    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    
    // Eigen decomposition
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    //const float3 lambda = {eigenValues[0], eigenValues[1], eigenValues[2]};
    //const float3 e1 = {eigenVectors[0][0], eigenVectors[1][0], eigenVectors[2][0]};
    const float3 e2 = {eigenVectors[0][1], eigenVectors[1][1], eigenVectors[2][1]};
    const float3 e3 = {eigenVectors[0][2], eigenVectors[1][2], eigenVectors[2][2]};

    /*
    if(lambda.y > 0 && lambda.z > 0) {
        T[LPOS(pos)] = 0;
        return;
    }
    */

    // Circle Fitting
    float maxSum = 0.0f;
    float maxRadius = 0.0f;
    const float4 floatPos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    for(float radius = rMin; radius <= rMax; radius += rStep) {
        float radiusSum = 0.0f;
        int samples = 32;
        int stride = 1;
        /*
        if(radius < 3) {
            samples = 8;
            stride = 4;
        } else if(radius < 6) {
            samples = 16;
            stride = 2;
        }
        */

        for(int j = 0; j < samples; j++) {
            float3 V_alpha = cosValues[j*stride]*e3 + sinValues[j*stride]*e2;
            float4 position = floatPos + radius*V_alpha.xyzz;
            float3 V = -read_imagef(vectorField, interpolationSampler, position).xyz;
            radiusSum += dot(V, V_alpha);
        }
        radiusSum /= samples;
        if(radiusSum > maxSum) {
            maxSum = radiusSum;
            maxRadius = radius;
        } else {
            break;
        }
    }

    // Store result
    T[LPOS(pos)] = maxSum;
    Radius[LPOS(pos)] = maxRadius;
}

#define SQR_MAG(pos) read_imagef(vectorField, sampler, pos).w


__kernel void dd(
    __read_only image3d_t vectorField,
    __read_only image3d_t TDF,
    __read_only image3d_t centerpointCandidates,
    __write_only image3d_t centerpoints,
    __private int cubeSize
    ) {

    int4 bestPos;
    float bestGVF = 0.0f;
    int4 readPos = {
        get_global_id(0)*cubeSize,
        get_global_id(1)*cubeSize,
        get_global_id(2)*cubeSize,
        0
    };
    bool found = false;
    for(int a = 0; a < cubeSize; a++) {
    for(int b = 0; b < cubeSize; b++) {
    for(int c = 0; c < cubeSize; c++) {
        int4 pos = readPos + (int4)(a,b,c,0);
        if(read_imagei(centerpointCandidates, sampler, pos).x == 1) {
            float GVF = read_imagef(TDF, sampler, pos).x;
            if(GVF > bestGVF) {
                found = true;
                bestGVF = GVF;
                bestPos = pos;
            }
        }
    }}}
    if(found) {
        write_imagei(centerpoints, bestPos, 1);
    }
}



__kernel void findCandidateCenterpoints(
    __read_only image3d_t TDF,
    __write_only image3d_t centerpoints,
    __private float TDFlimit
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    if(read_imagef(TDF, sampler, pos).x < TDFlimit) {
        write_imagei(centerpoints, pos, 0);
    } else {
        write_imagei(centerpoints, pos, 1);
    }
}

__kernel void findCandidateCenterpoints2(
    __read_only image3d_t TDF,
    __read_only image3d_t radius,
    __read_only image3d_t vectorField,
    __write_only image3d_t centerpoints,
    __private int HP_SIZE,
    __private int sum,
        __read_only image3d_t hp0, // Largest HP
        __read_only image3d_t hp1,
        __read_only image3d_t hp2,
        __read_only image3d_t hp3,
        __read_only image3d_t hp4,
        __read_only image3d_t hp5
        ,__read_only image3d_t hp6
        ,__read_only image3d_t hp7
        ,__read_only image3d_t hp8
        ,__read_only image3d_t hp9
    ) {
    int target = get_global_id(0);
    if(target >= sum)
        target = 0;
    int4 pos = traverseHP3D(target,HP_SIZE,hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9);

    const float thetaLimit = 0.5f;
    const float radii = read_imagef(radius, sampler, pos).x;
    const int maxD = max(min(round(radii), 5.0f), 1.0f);
    bool invalid = false;

    // Find Hessian Matrix
    float3 Fx, Fy, Fz;
    Fx = gradientNormalized(vectorField, pos, 0, 1);
    Fy = gradientNormalized(vectorField, pos, 1, 2);
    Fz = gradientNormalized(vectorField, pos, 2, 3);

    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    
    // Eigen decomposition
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    const float3 e1 = {eigenVectors[0][0], eigenVectors[1][0], eigenVectors[2][0]};

    for(int a = -maxD; a <= maxD; a++) {
    for(int b = -maxD; b <= maxD; b++) {
    for(int c = -maxD; c <= maxD; c++) {
        if(a == 0 && b == 0 && c == 0)
            continue;
        const int4 n = pos + (int4)(a,b,c,0);
        const float3 r = {a,b,c};
        const float dp = dot(e1,r);
        const float3 r_projected = r-e1*dp;
        const float theta = acos(dot(normalize(r), normalize(r_projected)));
        if(theta < thetaLimit && length(r) < maxD) {
            if(SQR_MAG(n) < SQR_MAG(pos)) {
                invalid = true;
                break;
            }    
        }
    }}}

    if(invalid) {
        write_imagei(centerpoints, pos, 0);
    } else {
        write_imagei(centerpoints, pos, 1);
    }
}

__kernel void GVF3DIteration(__read_only image3d_t init_vector_field, __read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    // Load data from shared memory and do calculations
    float2 init_vector = read_imagef(init_vector_field, sampler, pos).xy;
    float4 v = read_imagef(read_vector_field, sampler, pos);
    float3 fx1 = read_imagef(read_vector_field, sampler, pos + (int4)(1,0,0,0)).xyz;
    float3 fx_1 = read_imagef(read_vector_field, sampler, pos - (int4)(1,0,0,0)).xyz;
    float3 fy1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,1,0,0)).xyz;
    float3 fy_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,1,0,0)).xyz;
    float3 fz1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,0,1,0)).xyz;
    float3 fz_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,0,1,0)).xyz; 
    
    // Update the vector field: Calculate Laplacian using a 3D central difference scheme
    float3 laplacian = -6*v.xyz + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

    v.xyz += mu * laplacian - (v.xyz - (float3)(init_vector.x, init_vector.y, v.w))*(init_vector.x*init_vector.x+init_vector.y*init_vector.y+v.w*v.w);

    write_imagef(write_vector_field, writePos, v);
}

__kernel void GVF3DInit(__read_only image3d_t initVectorField, __write_only image3d_t vectorField, __write_only image3d_t newInitVectorField) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 value = read_imagef(initVectorField, sampler, pos);
    value.w = value.z;
    float4 initValue;
   initValue.xy = value.xy; 
    write_imagef(vectorField, pos, value);
    write_imagef(newInitVectorField, pos, initValue);
}

//__kernel void GVF3DFinish(__global float * vectorField, __global float * vectorField2, __global float * sqrMag) {
__kernel void GVF3DFinish(__read_only image3d_t vectorField, __write_only image3d_t vectorField2) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 v = read_imagef(vectorField, sampler, pos); 
    v.w = 0;
    v.w = length(v) > 0.0f ? length(v) : 1.0f;
    //printf("%f %f %f\n", v.x,v.y,v.z);
    //vstore3(v.xyz, LPOS(pos), vectorField2);
    write_imagef(vectorField2, pos, v);
    //sqrMag[LPOS(pos)] = v.w;
}



#define MAX(a, b) ((a)>(b)?(a):(b))

#define SIZE 3

float hypot2(float x, float y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

void tred2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
  }

  // Householder reduction to tridiagonal form.

  for (int i = SIZE-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    float scale = 0.0f;
    float h = 0.0f;
    for (int k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0f) {
      e[i] = d[i-1];
      for (int j = 0; j < i; j++) {
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
        V[j][i] = 0.0f;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      float f = d[i-1];
      float g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e[j] = 0.0f;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0f;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      float hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i-1; k++) {
          V[k][j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.

  for (volatile int i = 0; i < SIZE-1; i++) {
    V[SIZE-1][i] = V[i][i];
    V[i][i] = 1.0f;
    float h = d[i+1];
    if (h != 0.0f) {
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i+1] / h;
      }
      for (int j = 0; j <= i; j++) {
        float g = 0.0f;
        for (int k = 0; k <= i; k++) {
          g += V[k][i+1] * V[k][j];
        }
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V[k][i+1] = 0.0f;
    }
  }
  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
    V[SIZE-1][j] = 0.0f;
  }
  V[SIZE-1][SIZE-1] = 1.0f;
  e[0] = 0.0f;
} 

// Symmetric tridiagonal QL algorithm.

void tql2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int i = 1; i < SIZE; i++) {
    e[i-1] = e[i];
  }
  e[SIZE-1] = 0.0f;

  float f = 0.0f;
  float tst1 = 0.0f;
  float eps = pow(2.0f,-52.0f);
  for (int l = 0; l < SIZE; l++) {

    // Find small subdiagonal element

    tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < SIZE) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        float g = d[l];
        float p = (d[l+1] - g) / (2.0f * e[l]);
        float r = hypot2(p,1.0f);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        float dl1 = d[l+1];
        float h = g - d[l];
        for (int i = l+2; i < SIZE; i++) {
          d[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d[m];
        float c = 1.0f;
        float c2 = c;
        float c3 = c;
        float el1 = e[l+1];
        float s = 0.0f;
        float s2 = 0.0f;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);

          // Accumulate transformation.

          for (int k = 0; k < SIZE; k++) {
            h = V[k][i+1];
            V[k][i+1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;

        // Check for convergence.

      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0f;
  }
  
  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < SIZE-1; i++) {
    int k = i;
    float p = d[i];
    for (int j = i+1; j < SIZE; j++) {
      if (fabs(d[j]) < fabs(p)) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < SIZE; j++) {
        p = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = p;
      }
    }
  }
}

void eigen_decomposition(float A[SIZE][SIZE], float V[SIZE][SIZE], float d[SIZE]) {
  float e[SIZE];
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      V[i][j] = A[i][j];
    }
  }
  tred2(V, d, e);
  tql2(V, d, e);
}


