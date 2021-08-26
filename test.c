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
/* Function 2: A C fibonacci implementation this is nothing special and looks 
   exactly like a normal C version of fibonacci would look.*/
// int Cfib(double *n){

//     printf("hello Cfib _ \n");
    
//     return 0;

// }

// Our Python binding to our C function
// This will take one and only one non-keyword argument
static PyObject* fib(PyObject* self, PyObject* args){
    
    PyObject *array1_obj;
    PyObject *array2_obj;

    if(!PyArg_ParseTuple(args, "OO", &array1_obj, &array2_obj))
        return NULL;

    double **array1;
    double **array2;

    // create C arrays from numpy objects
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];    

    if (PyArray_AsCArray(&array1_obj, (void **)&array1, dims, 2, descr) < 0 || PyArray_AsCArray(&array2_obj, (void ***)&array2, dims, 2, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }

    printf("2D: %f, 2D: %f.\n", array1[1][1], array2[1][2]);

    // return our computed fib number    
    // return Py_BuildValue("d", Cfib(array));
    // return Py_BuildValue("f", Cfib(array1));
}

/* ------------------ 3rd part: The method mapping table ------------------ */
/* Our Module's Function Definition struct. We require this `NULL` to signal
 the end of our method definition */
static PyMethodDef myMethods[] = {
    { "fib", fib, METH_VARARGS, "Fibonancy"},
    { NULL, NULL, 0, NULL}
};

// Our Module Definition struct
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