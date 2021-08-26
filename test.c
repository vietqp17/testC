/* A Python extension module has four parts:
        + The header file Python.h.
        + The C function we want to expose as the interface from our module.
        + A table mapping the names of our functions as Python developers see 
        them to C functions inside the extension module.
        + An initialization function.
*/ 

/* ------------------ 1st part: The header fule ------------------ */
#include <Python.h>
#include <arrayobject.h>

/* ------------------ 2nd part: The C functions ------------------ */
int func_matadd(int num_row, int num_col, double **input_1, double **input_2, double **output){

    for (int i = 0; i < num_row; i++){
        for (int j = 0; j < num_col; j++) {
            output[i][j] = input_1[i][j] + input_2[i][j];            
        }
    }
    
    return 0;
}

// Our Python binding to our C function
static PyObject* ext_matadd(PyObject* self, PyObject* args){
    
    int num_row;
    int num_col;        
    PyObject *array1_obj;
    PyObject *array2_obj;
    PyObject *array3_obj;    

    if(!PyArg_ParseTuple(args, "iiOOO", &num_row, &num_col, &array1_obj, &array2_obj, &array3_obj))
        return NULL;

    double **array1;
    double **array2;
    double **array3;

    // create C arrays from numpy objects
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;    
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];

    // Check if whether or not converting to c arrays
    if (PyArray_AsCArray(&array1_obj, (void **)&array1, dims, 2, descr) < 0 || \
        PyArray_AsCArray(&array2_obj, (void **)&array2, dims, 2, descr) < 0 || \
        PyArray_AsCArray(&array3_obj, (void **)&array3, dims, 2, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }   

    func_matadd(num_row, num_col, array1, array2, array3);
    
    return Py_BuildValue("d", 0);    
}

/* ------------------ 3rd part: The method mapping table ------------------ */
static PyMethodDef myMethods[] = {
    { "ext_matadd", ext_matadd, METH_VARARGS, "Sum of matrices"},
    { NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "fib",
    "Test Module",
    -1,
    myMethods
};

/* ------------------ 4nd part: The method mapping table ------------------ */
PyMODINIT_FUNC  PyInit_myModule(void)
{
    import_array();
    return PyModule_Create(&myModule);
}