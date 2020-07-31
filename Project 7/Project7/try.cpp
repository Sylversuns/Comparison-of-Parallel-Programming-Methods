#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <omp.h>
#include "simd.h"
#include "cl.h"
#include "cl_platform.h"

#ifndef LOCAL_SIZE
#define LOCAL_SIZE              256
#endif

const char *                    CL_FILE_NAME = { "first.cl" };
const float                     TOL = 0.0001f;

void                            Wait( cl_command_queue );
int                             LookAtTheBits( float );

int main()
{
	FILE *fp = fopen( "signal.txt", "r" );
	if( fp == NULL )
	{
		fprintf( stderr, "Cannot open file 'signal.txt'\n" );
		exit( 1 );
	}
	int Size;
	fscanf( fp, "%d", &Size );
	float *Array = new float[ 2*Size ];
	float *Sums  = new float[ 1*Size ];
	for( int i = 0; i < Size; i++ )
	{
		fscanf( fp, "%f", &Array[i] );
		Array[i+Size] = Array[i];		// duplicate the array
	}
	fclose( fp );

	// OPENMP SECTION BELOW

        #ifndef _OPENMP
                fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
                return 1;
        #endif	
	
	int NUMT = 16;

	omp_set_num_threads( NUMT );
        fprintf( stderr, "Using %d thread(s).\n", NUMT );

        double time0 = omp_get_wtime();

	#pragma omp parallel for default(none) shared (Array, Size, Sums)
	for( int shift = 0; shift < Size; shift++ )
	{
		float sum = 0.;
		for (int i = 0; i < Size; i++)
		{
			sum += Array[i] * Array[i + shift];
		}
		Sums[shift] = sum;
	}

        double time1 = omp_get_wtime();	

        double MCPS = (double) Size*Size/(time1-time0)/1000000.;


        FILE *fp2 = fopen( "autocorrelation.txt", "w" );
        if( fp2 == NULL )
        {
                fprintf( stderr, "Cannot open file 'autocorrelation.txt'\n" );
                exit( 1 );
        }

	fprintf(fp2, "%d\n", Size);
        
	for( int i = 0; i < Size; i++ )
        {
        	fprintf( fp2, "%f\n", Sums[i] );
        }
        fclose( fp2 );	


        printf("The MegaCalculations Per Second (MCPS) with %d OpenMP thread(s) is %f.\n", NUMT, MCPS);

	// END OF OPENMP SECTION

	

	// SIMD SECTION

	time0 = omp_get_wtime();

	for (int shift = 0; shift < Size; shift++)
	{
		Sums[shift] = SimdMulSum(&Array[0], &Array[0+shift], Size);
	}

	time1 = omp_get_wtime();

        MCPS = (double) Size*Size/(time1-time0)/1000000.;
        printf("The MegaCalculations Per Second (MCPS) with SIMD is %f.\n", MCPS);

	// END OF SIMD SECTION



	// OPENCL SECTION

        // see if we can even open the opencl kernel program
        // (no point going on if we can't):

        FILE *fp3;
#ifdef WIN32
        errno_t err = fopen_s( &fp3, CL_FILE_NAME, "r" );
        if( err != 0 )
#else
        fp3 = fopen( CL_FILE_NAME, "r" );
        if( fp3 == NULL )
#endif
        {       
                fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
                return 1;
        }

        cl_int status;          // returned status from opencl calls
                                // test against CL_SUCCESS

	// get the platform id:

        cl_platform_id platform;
        status = clGetPlatformIDs( 1, &platform, NULL );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clGetPlatformIDs failed (2)\n" );

        // get the device id:

        cl_device_id device;
        status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clGetDeviceIDs failed (2)\n" );
	
	// allocate the host memory buffers:

	float *hArray = new float[ 2*Size ];
	float *hSums  = new float[ 1*Size ];

        // 3. create an opencl context:

        cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clCreateContext failed\n" );

        // 4. create an opencl command queue:

        cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clCreateCommandQueue failed\n" );

	// 5. allocate the device memory buffers:

	cl_mem dArray = clCreateBuffer( context, CL_MEM_READ_ONLY,  2*Size*sizeof(cl_float), NULL, &status );
	cl_mem dSums  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, 1*Size*sizeof(cl_float), NULL, &status );

	// 6. enqueue the commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer( cmdQueue, dArray, CL_FALSE, 0, 2*Size, hArray, 0, NULL, NULL );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

        status = clEnqueueWriteBuffer( cmdQueue, dSums, CL_FALSE, 0, 1*Size, hSums, 0, NULL, NULL );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clEnqueueWriteBuffer failed (2)\n" );

	// 7. read the kernel code from a file:

	fseek( fp3, 0, SEEK_END );
        size_t fileSize = ftell( fp3 );
        fseek( fp3, 0, SEEK_SET );
        char *clProgramText = new char[ fileSize+1 ];           // leave room for '\0'
        size_t n = fread( clProgramText, 1, fileSize, fp3 );
        clProgramText[fileSize] = '\0';
        fclose( fp3 );
        if( n != fileSize )
                fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

	// create the text for the kernel program:

	char *strings[1];
        strings[0] = clProgramText;
        cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clCreateProgramWithSource failed\n" );
        delete [ ] clProgramText;

	// 8. compile and link the kernel code:

	char *options = { "" };
        status = clBuildProgram( program, 1, &device, options, NULL, NULL );
        if( status != CL_SUCCESS )
        {
                size_t size;
                clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
                cl_char *log = new cl_char[ size ];
                clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
                fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
                delete [ ] log;
        }

	// 9. create the kernel object:

	cl_kernel kernel = clCreateKernel( program, "AutoCorrelate", &status );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clCreateKernel failed\n" );

	// 10. setup the arguments to the kernel objects:

	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dArray );
	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dSums  );

	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { Size,         1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

        Wait( cmdQueue );
        time0 = omp_get_wtime( );
	
        status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
        if( status != CL_SUCCESS )
                fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

        Wait( cmdQueue );
        time1 = omp_get_wtime( );

	// 12. read the results buffer back from the device to the host:

        status = clEnqueueReadBuffer( cmdQueue, dSums, CL_TRUE, 0, Size, hSums, 0, NULL, NULL );
        if( status != CL_SUCCESS )
        	fprintf( stderr, "clEnqueueReadBuffer failed\n" );

        MCPS = (double) Size*Size/(time1-time0)/1000000.;
        printf("The MegaCalculations Per Second (MCPS) with OpenCL is %f.\n", MCPS);

	// END OF OPENCL SECTION

	return 0;
}



float
SimdMulSum( float *a, float *b, int len )
{
        float sum[4] = { 0., 0., 0., 0. };
        int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;

        __asm
        (
                ".att_syntax\n\t"
                "movq    -40(%rbp), %r8\n\t"            // a
                "movq    -48(%rbp), %rcx\n\t"           // b
                "leaq    -32(%rbp), %rdx\n\t"           // &sum[0]
                "movups  (%rdx), %xmm2\n\t"             // 4 copies of 0. in xmm2
        );

        for( int i = 0; i < limit; i += SSE_WIDTH )
        {
                __asm
                (
                        ".att_syntax\n\t"
                        "movups (%r8), %xmm0\n\t"       // load the first sse register
                        "movups (%rcx), %xmm1\n\t"      // load the second sse register
                        "mulps  %xmm1, %xmm0\n\t"       // do the multiply
                        "addps  %xmm0, %xmm2\n\t"       // do the add
                        "addq $16, %r8\n\t"
                        "addq $16, %rcx\n\t"
                );
        }

        __asm
        (
                ".att_syntax\n\t"
                "movups  %xmm2, (%rdx)\n\t"     // copy the sums back to sum[ ]
        );

        for( int i = limit; i < len; i++ )
        {
                sum[0] += a[i] * b[i];
        }

        return sum[0] + sum[1] + sum[2] + sum[3];
}

int
LookAtTheBits( float fp )
{
        int *ip = (int *)&fp;
        return *ip;
}


// wait until all queued tasks have taken place:

void
Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
              fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
              fprintf( stderr, "Wait: clWaitForEvents failed\n" );
}
