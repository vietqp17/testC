/* A Python extension module has four parts:
        + The header file Python.h.
        + The C function we want to expose as the interface from our module.
        + A table mapping the names of our functions as Python developers see 
        them to C functions inside the extension module.
        + An initialization function.
*/ 

/* ------------------ 1st part: The header fule ------------------ */
#include <Python.h>

/* ------------------ 2nd part: The C functions ------------------ */
/* Function 2: A C fibonacci implementation this is nothing special and looks 
   exactly like a normal C version of fibonacci would look.*/
int Cfib(int **n){

    printf("hello Cfib");

    // return in1;

}

// Our Python binding to our C function
// This will take one and only one non-keyword argument
static PyObject* fib(PyObject* self, PyObject* args)
{
    // instantiate our `n` value
    int **n;

    // if our `n` value
    if(!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    // return our computed fib number    
    return Py_BuildValue("i", Cfib(n));
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
    return PyModule_Create(&myModule);
}