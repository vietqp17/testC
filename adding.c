#include <stdio.h>
#include <stdlib.h>

// adding two matrix
int func_matadd(int input_1[2][2], int input_2[2][2], int result[2][2]){
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++) {
            result[i][j] = input_1[i][j] + input_2[i][j];
        }
    }
}

int Cfib(int n)
{
    if (n < 2)
        return n;
    else
        return Cfib(n-1)+Cfib(n-2);

    printf("Hello World\n");
}

int main() {    
	    
    int r=2, c=2;
    int i,j,A[2][2],B[2][2],sum_matrix[2][2];

    printf("Matrix A: \n");
    for(i=0; i<r; i++) {
    	for(j=0; j<c; j++) {
        	A[i][j]	= rand()%100;
            printf("%d ", A[i][j]);
    	}
    	printf("\n");
    }   

    printf("\n");

    printf("Matrix B: \n");
    for(i = 0; i<r; i++) {
    	for(j=0; j<c;j++) {
        	B[i][j]	= rand()%100;
            printf("%d ", B[i][j]);
    	}
    	printf("\n");
    }

    // adding two matrices
    func_matadd(A, B, sum_matrix);

    // printing the result
    printf("\nSum of two matrices: \n");
    for (i = 0; i < r; ++i){
        for (j = 0; j < c; ++j) {
            printf("%d   ", sum_matrix[i][j]);
            if (j == c - 1) {
                printf("\n\n");
            }
        }
    }

	return 0;
}