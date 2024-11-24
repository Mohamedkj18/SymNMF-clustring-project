#define PY_SSIZE_T_CLEAN
#include <Python.h>

double **sym(double **mat, double **symMat, int r, int c);
double *ddg(double **mat, double **diag, double **symMat, int r, int c);
double **norm(double **mat, double **diag, double **symMat, double **NSM, int r, int c);
double **symnmf(double **W, double **H, int r, int k);
static PyObject *fitSym(PyObject *self, PyObject *args)
{
    PyObject *pyMat, *dataPoint, *symMat, *row;
    int ROWS, COLS, i, j;
    double **mat, **res;

    // Declare variables at the beginning
    if (!PyArg_ParseTuple(args, "O", &pyMat))
        return NULL;

    ROWS = (int)PyObject_Length(pyMat);
    COLS = (int)PyObject_Length(PyList_GetItem(pyMat, 0));

    mat = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        mat[i] = (double *)calloc(COLS, sizeof(double));
        dataPoint = PyList_GetItem(pyMat, i);
        if (!PyList_Check(dataPoint))
        {
            PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }
        for (j = 0; j < COLS; j++)
        {
            mat[i][j] = (double)PyFloat_AsDouble(PyList_GetItem(dataPoint, j));
        }
    }

    res = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        res[i] = (double *)calloc(ROWS, sizeof(double));
    }

    sym(mat, res, ROWS, COLS);

    symMat = PyList_New(ROWS);
    for (i = 0; i < ROWS; i++)
    {
        row = PyList_New(ROWS);
        for (j = 0; j < ROWS; j++)
        {

            PyList_SetItem(row, j, Py_BuildValue("d", res[i][j]));
        }
        PyList_SetItem(symMat, i, row);
    }

    for (i = 0; i < ROWS; i++)
    {
        free(res[i]);
        free(mat[i]);
    }
    free(res);
    free(mat);

    return symMat;
}

static PyObject *fitNorm(PyObject *self, PyObject *args)
{
    PyObject *pyMat, *dataPoint, *NSM, *row;
    int ROWS, COLS, i, j;
    double **mat, **symMat, **diag, **res;

    // Declare variables at the beginning
    if (!PyArg_ParseTuple(args, "O", &pyMat))
        return NULL;

    ROWS = (int)PyObject_Length(pyMat);
    COLS = (int)PyObject_Length(PyList_GetItem(pyMat, 0));

    mat = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        mat[i] = (double *)calloc(COLS, sizeof(double));
        dataPoint = PyList_GetItem(pyMat, i);
        if (!PyList_Check(dataPoint))
        {
            PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }
        for (j = 0; j < COLS; j++)
        {
            mat[i][j] = (double)PyFloat_AsDouble(PyList_GetItem(dataPoint, j));
        }
    }

    res = (double **)calloc(ROWS, sizeof(double *));
    diag = (double **)calloc(ROWS, sizeof(double *));
    symMat = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        res[i] = (double *)calloc(ROWS, sizeof(double));
        diag[i] = (double *)calloc(ROWS, sizeof(double));
        symMat[i] = (double *)calloc(ROWS, sizeof(double));
    }

    norm(mat, diag, symMat, res, ROWS, COLS);
    NSM = PyList_New(ROWS);
    for (i = 0; i < ROWS; i++)
    {
        row = PyList_New(ROWS);
        for (j = 0; j < ROWS; j++)
        {

            PyList_SetItem(row, j, Py_BuildValue("d", res[i][j]));
        }
        PyList_SetItem(NSM, i, row);
    }

    for (i = 0; i < ROWS; i++)
    {
        free(mat[i]);
        free(res[i]);
        free(diag[i]);
        free(symMat[i]);
    }
    free(mat);
    free(res);
    free(diag);
    free(symMat);

    return NSM;
}

static PyObject *fitDdg(PyObject *self, PyObject *args)
{
    PyObject *pyMat, *dataPoint, *diag, *row;
    int ROWS, COLS, i, j;
    double **mat, **res, **symMat;

    // Declare variables at the beginning
    if (!PyArg_ParseTuple(args, "O", &pyMat))
        return NULL;

    ROWS = (int)PyObject_Length(pyMat);
    COLS = (int)PyObject_Length(PyList_GetItem(pyMat, 0));

    mat = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        mat[i] = (double *)calloc(COLS, sizeof(double));
        dataPoint = PyList_GetItem(pyMat, i);
        if (!PyList_Check(dataPoint))
        {
            PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }
        for (j = 0; j < COLS; j++)
        {
            mat[i][j] = (double)PyFloat_AsDouble(PyList_GetItem(dataPoint, j));
        }
    }

    res = (double **)calloc(ROWS, sizeof(double *));
    symMat = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        res[i] = (double *)calloc(ROWS, sizeof(double));
        symMat[i] = (double *)calloc(ROWS, sizeof(double));
    }

    ddg(mat, res, symMat, ROWS, COLS);

    diag = PyList_New(ROWS);
    for (i = 0; i < ROWS; i++)
    {
        row = PyList_New(ROWS);
        for (j = 0; j < ROWS; j++)
        {

            PyList_SetItem(row, j, Py_BuildValue("d", res[i][j]));
        }
        PyList_SetItem(diag, i, row);
    }

    for (i = 0; i < ROWS; i++)
    {
        free(res[i]);
        free(mat[i]);
        free(symMat[i]);
    }
    free(res);
    free(mat);
    free(symMat);

    return diag;
}

static PyObject *fitSymnmf(PyObject *self, PyObject *args)
{
    PyObject *pyW, *pyH, *dataPoint1, *dataPoint2, *optim, *row;
    int ROWS, i, j, k;
    double **W, **H;

    // Declare variables at the beginning
    if (!PyArg_ParseTuple(args, "OOii", &pyW, &pyH, &k, &ROWS))
        return NULL;

    W = (double **)calloc(ROWS, sizeof(double *));
    H = (double **)calloc(ROWS, sizeof(double *));
    for (i = 0; i < ROWS; i++)
    {
        W[i] = (double *)calloc(ROWS, sizeof(double));
        H[i] = (double *)calloc(k, sizeof(double));
        dataPoint1 = PyList_GetItem(pyW, i);
        dataPoint2 = PyList_GetItem(pyH, i);
        if (!PyList_Check(dataPoint1) || !PyList_Check(dataPoint2))
        {
            PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }
        for (j = 0; j < ROWS; j++)
        {
            W[i][j] = (double)PyFloat_AsDouble(PyList_GetItem(dataPoint1, j));
            if (j < k)
                H[i][j] = (double)PyFloat_AsDouble(PyList_GetItem(dataPoint2, j));
        }
    }

    symnmf(W, H, ROWS, k);

    optim = PyList_New(ROWS);
    for (i = 0; i < ROWS; i++)
    {
        row = PyList_New(k);
        for (j = 0; j < k; j++)
        {

            PyList_SetItem(row, j, Py_BuildValue("d", H[i][j]));
        }
        PyList_SetItem(optim, i, row);
    }

    for (i = 0; i < ROWS; i++)
    {
        free(W[i]);
        free(H[i]);
    }
    free(W);
    free(H);
    return optim;
}

static PyMethodDef fit_FunctionTable[] = {
    {"sym", (PyCFunction)fitSym, METH_VARARGS, "Symmetric matrix function"},
    {"norm", (PyCFunction)fitNorm, METH_VARARGS, "Normalization function"},
    {"ddg", (PyCFunction)fitDdg, METH_VARARGS, "DDG function"},
    {"symnmf", (PyCFunction)fitSymnmf, METH_VARARGS, "Symmetric NMF function"},
    {NULL, NULL, 0, NULL} // Sentinel to mark end of table
};

static struct PyModuleDef symnmfModule = {
    PyModuleDef_HEAD_INIT,
    "symnmfsp", // Module name
    "Python wrapper for custom C extension",
    -1,
    fit_FunctionTable};

PyMODINIT_FUNC PyInit_symnmfsp(void)
{
    PyObject *m;

    m = PyModule_Create(&symnmfModule);
    if (!m)
    {
        return NULL;
    }
    return m;
}
