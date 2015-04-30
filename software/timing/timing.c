/*
 * timing.c
 * Apr 29, 2015 10:43:37 EDT
 * Copyright 2015
 *        
 * Andre Young <andre.young@cfa.harvard.edu>
 * Harvard-Smithsonian Center for Astrophysics
 * 60 Garden Street, Cambridge
 * MA 02138
 * 
 * Changelog:
 * 	AY: Created 2015-04-29
 */

/*
 * This module provides access to clock_gettime() within Python.
 * See man CLOCK_GETRES for details.
 */

#include <Python.h>
#include "structmember.h"

//~ // Flag to enable debugging reference counts
//~ #define __DEBUG__REFCNT

/*
 * Looks like time.h is already provided by Python.h
*/
//#include <time.h> 

// Forward declarations
static PyTypeObject timing_TimeType;
void inittiming(void); 
static PyObject * timing_time_to_Time(struct timespec t);

/*
 * Time object attributes.
 */
typedef struct {
    PyObject_HEAD
    long seconds;
    long nanoseconds;
} timing_TimeObject;

/*
 * Time object desctructor.
 */
static void
Time_dealloc(timing_TimeObject *self)
{
#ifdef __DEBUG__REFCNT
	printf("Time_dealloc: refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
#endif
	self->ob_type->tp_free((PyObject *)self);
}

/*
 * Time object constructor (new).
 */
static PyObject *
Time_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	timing_TimeObject *self;
	
	self = (timing_TimeObject *)type->tp_alloc(type, 0);
	if (self != NULL)
	{
		self->seconds = 0;
		self->nanoseconds = 0;
	}
	
#ifdef __DEBUG__REFCNT
	printf("Time_new: ");
	printf("refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
#endif
	
	return (PyObject *)self;
}

/*
 * Time object initialization (init).
 */
static int
Time_init(timing_TimeObject *self, PyObject *args, PyObject *kwds)
{
	
#ifdef __DEBUG__REFCNT
	printf("Time_init: ");
	printf("refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
#endif

	self->seconds = (long)0;
	self->nanoseconds = (long)0;
	
	static char *kwlist[] = {"seconds","nanoseconds"};
	
	if (! PyArg_ParseTupleAndKeywords(args, kwds, "|ll", kwlist, 
					&self->seconds, &self->nanoseconds) )
	{
		return -1;
	}
	
	return 0;
}

/*
 * Time object str()
 */
static PyObject *
Time_str(PyObject *self)
{
	long sec = ((timing_TimeObject *)self)->seconds;
	long nsec = ((timing_TimeObject *)self)->nanoseconds;
	double t = (double)sec + 1e-9*(double)nsec;
	char str[100];
	sprintf(str,"%.9f seconds",t);
	return Py_BuildValue("s",str);
}

/*
 * Time.seconds getter
 */
static PyObject *
Time_getseconds(timing_TimeObject *self, void *closure)
{
	PyObject *sec = Py_BuildValue("l",self->seconds);
	
#ifdef __DEBUG__REFCNT
	printf("Time_getseconds: ");
	printf("refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
	printf("Time_getseconds: ");
	printf("refcount to sec is %lu\n", (unsigned long)Py_REFCNT(sec));
#endif

	return sec;
}

/*
 * Time.seconds setter
 */
static int
Time_setseconds(timing_TimeObject *self, PyObject *value, void *closure)
{
	PyErr_SetString(PyExc_TypeError,
					"timing.Time seconds cannot be assigned.");
	
	return -1;
}

/*
 * Time.nanoseconds getter
 */
static PyObject *
Time_getnanoseconds(timing_TimeObject *self, void *closure)
{
	PyObject *nsec = Py_BuildValue("l",self->nanoseconds);
	
#ifdef __DEBUG__REFCNT
	printf("Time_getnanoseconds: ");
	printf("refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
	printf("Time_getnanoseconds: ");
	printf("refcount to nsec is %lu\n", (unsigned long)Py_REFCNT(nsec));
#endif
	
	return nsec;
}

/*
 * Time.nanoseconds setter
 */
static int
Time_setnanoseconds(timing_TimeObject *self, PyObject *value, void *closure)
{
	PyErr_SetString(PyExc_TypeError,
				"timing.Time nanoseconds cannot be assigned.");
	
	return -1;
}

/*
 * Time operator overloads: Add
 */ 
static PyObject *
Time_Add(PyObject *self, PyObject *other)
{
	struct timespec t;
	
	t.tv_nsec = ((timing_TimeObject *)self)->nanoseconds + ((timing_TimeObject *)other)->nanoseconds;
	t.tv_sec = ((timing_TimeObject *)self)->seconds + ((timing_TimeObject *)other)->seconds;
	
	return timing_time_to_Time(t);
}

/*
 * Time operator overloads: Subtract
 */ 
static PyObject *
Time_Subtract(PyObject *self, PyObject *other)
{
	struct timespec t;
	
	t.tv_nsec = ((timing_TimeObject *)self)->nanoseconds - ((timing_TimeObject *)other)->nanoseconds;
	t.tv_sec = ((timing_TimeObject *)self)->seconds - ((timing_TimeObject *)other)->seconds;
	
	return timing_time_to_Time(t);
}

/*
 * Time object method: time
 * Returns the sum of seconds and nanoseconds as double.
 */
static PyObject *
Time_time(timing_TimeObject *self)
{
	double t = (double)self->seconds + 1e-9*(double)self->nanoseconds;
	
	PyObject *total_time = Py_BuildValue("d",t);

#ifdef __DEBUG__REFCNT
	printf("Time_time: ");
	printf("refcount to self is %lu\n", (unsigned long)Py_REFCNT(self));
	printf("Time_time: ");
	printf("refcount to total_time is %lu\n", (unsigned long)Py_REFCNT(total_time));
#endif
	
	return total_time;
}

/*
 * Time object getters/setters
 */
static PyGetSetDef Time_getseters[] = {
	{"seconds", (getter)Time_getseconds, (setter)Time_setseconds, "Whole seconds portion of Time object."},
	{"nanoseconds", (getter)Time_getnanoseconds, (setter)Time_setnanoseconds, "Nanoseconds portion of Time object."},
	{NULL}
};

/*
 * Time object members.
 */
static PyMemberDef Time_members[] = {
	// These are no longer used since we made attributes read-only.
	//~ {"seconds", T_LONG, offsetof(timing_TimeObject, seconds), 0, "Seconds."},
	//~ {"nanoseconds", T_LONG, offsetof(timing_TimeObject, nanoseconds), 0, "Nanoseconds."},
	{NULL}
};

/*
 * Time object methods.
 */
static PyMethodDef Time_methods[] = {
	{"time", (PyCFunction)Time_time, METH_NOARGS, 
		"Return the total time of the Time object as double."},
	{NULL}
};

/*
 * Time number methods
 */
static PyNumberMethods Time_NumberMethods[] = {
	{&Time_Add,                       /*binaryfunc nb_add;*/
	&Time_Subtract,                       /*binaryfunc nb_subtract;*/
	0,                       /*binaryfunc nb_multiply;*/
	0,                       /*binaryfunc nb_divide;*/
	0,                       /*binaryfunc nb_remainder;*/
	0,                       /*binaryfunc nb_divmod;*/
	0,                       /*ternaryfunc nb_power;*/
	0,                       /*unaryfunc nb_negative;*/
	0,                       /*unaryfunc nb_positive;*/
	0,                       /*unaryfunc nb_absolute;*/
	0,                       /*inquiry nb_nonzero;*/       /* Used by PyObject_IsTrue */
	0,                       /*unaryfunc nb_invert;*/
	0,                       /*binaryfunc nb_lshift;*/
	0,                       /*binaryfunc nb_rshift;*/
	0,                       /*binaryfunc nb_and;*/
	0,                       /*binaryfunc nb_xor;*/
	0,                       /*binaryfunc nb_or;*/
	0,                       /*coercion nb_coerce;*/       /* Used by the coerce() function */
	0,                       /*unaryfunc nb_int;*/
	0,                       /*unaryfunc nb_long;*/
	0,                       /*unaryfunc nb_float;*/
	0,                       /*unaryfunc nb_oct;*/
	0,                       /*unaryfunc nb_hex;*/
	/* Added in release 2.0 */
	0,                       /*binaryfunc nb_inplace_add;*/
	0,                       /*binaryfunc nb_inplace_subtract;*/
	0,                       /*binaryfunc nb_inplace_multiply;*/
	0,                       /*binaryfunc nb_inplace_divide;*/
	0,                       /*binaryfunc nb_inplace_remainder;*/
	0,                       /*ternaryfunc nb_inplace_power;*/
	0,                       /*binaryfunc nb_inplace_lshift;*/
	0,                       /*binaryfunc nb_inplace_rshift;*/
	0,                       /*binaryfunc nb_inplace_and;*/
	0,                       /*binaryfunc nb_inplace_xor;*/
	0,                       /*binaryfunc nb_inplace_or;*/
	/* Added in release 2.2 */
	0,                       /*binaryfunc nb_floor_divide;*/
	0,                       /*binaryfunc nb_true_divide;*/
	0,                       /*binaryfunc nb_inplace_floor_divide;*/
	0,                       /*binaryfunc nb_inplace_true_divide;*/
	/* Added in release 2.5 */
	0},                       /*unaryfunc nb_index;*/
};

/*
 * Define Time type.
 */
static PyTypeObject timing_TimeType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "timing.Time",             /*tp_name*/
    sizeof(timing_TimeObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Time_dealloc,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    Time_NumberMethods,        /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    (reprfunc)Time_str,        /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "Time object for accurate timing.",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Time_methods,              /* tp_methods */
    Time_members,              /* tp_members */
    Time_getseters,            /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Time_init,       /* tp_init */
    0,                         /* tp_alloc */
    Time_new,                  /* tp_new */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

// Could probably be removed...
int
main(int argc, char *argv[])
{
	Py_SetProgramName(argv[0]);
	
	Py_Initialize();
	
	inittiming();
	
	PySys_SetArgv(argc, argv);
	
	Py_Exit(0);
	
	return EXIT_SUCCESS;
}

/*
 * Module method that returns the current Thread CPU time as a Time object.
 */
static PyObject *
timing_get_thread_cpu_time(PyObject *self, PyObject *args)
{
	struct timespec t;
	
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
	
	return timing_time_to_Time(t);
}

/*
 * Module method that returns the current Process CPU time as a Time object.
 */
static PyObject *
timing_get_process_cpu_time(PyObject *self, PyObject *args)
{
	struct timespec t;
	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
	
	return timing_time_to_Time(t);

}

static PyObject *
timing_time_to_Time(struct timespec t)
{
		PyObject *new_args = Py_BuildValue("ll",(long)t.tv_sec,t.tv_nsec);
#ifdef __DEBUG__REFCNT
	printf("timing_time_to_Time:main: ");
	printf("refcount to new_args is %lu\n", (unsigned long)Py_REFCNT(new_args));
#endif
	PyObject *new_kwargs = PyDict_New();	
#ifdef __DEBUG__REFCNT
	printf("timing_time_to_Time:main: ");
	printf("refcount to new_kwargs is %lu\n", (unsigned long)Py_REFCNT(new_kwargs));
#endif
	timing_TimeObject *new_Time = (timing_TimeObject *)Time_new(&timing_TimeType, NULL, NULL);
#ifdef __DEBUG__REFCNT
	printf("timing_time_to_Time:main: ");
	printf("refcount to new_Time is %lu\n", (unsigned long)Py_REFCNT(new_Time));
#endif
	if (Time_init(new_Time, new_args, new_kwargs) == 0)
	{
		Py_DECREF(new_args);
		Py_DECREF(new_kwargs);
#ifdef __DEBUG__REFCNT
		printf("timing_time_to_Time:init_success: ");
		printf("refcount to new_Time is %lu\n", (unsigned long)Py_REFCNT(new_Time));
		printf("timing_time_to_Time:init_success: ");
		printf("refcount to new_args is %lu\n", (unsigned long)Py_REFCNT(new_args));
		printf("timing_time_to_Time:init_success: ");
		printf("refcount to new_kwargs is %lu\n", (unsigned long)Py_REFCNT(new_kwargs));
#endif
		return (PyObject *)new_Time;
	}
	else
	{
		Py_DECREF(new_args);
		Py_DECREF(new_kwargs);
		Py_DECREF(new_Time);
#ifdef __DEBUG__REFCNT
		printf("timing_time_to_Time:init_failure: ");
		printf("refcount to new_Time is %lu\n", (unsigned long)Py_REFCNT(new_Time));
		printf("timing_time_to_Time:init_failure: ");
		printf("refcount to new_args is %lu\n", (unsigned long)Py_REFCNT(new_args));
		printf("timing_time_to_Time:init_failure: ");
		printf("refcount to new_kwargs is %lu\n", (unsigned long)Py_REFCNT(new_kwargs));
#endif
		return NULL;
	}
}

// Add module methods
static PyMethodDef timing_Methods[] = {
	{"get_thread_cpu_time", timing_get_thread_cpu_time, METH_NOARGS, "Get thread CPU time."},
	{"get_process_cpu_time", timing_get_process_cpu_time, METH_NOARGS, "Get process CPU time."},
	{NULL, NULL, 0, NULL}
};

/*
 * Module initialization routine.
 */
PyMODINIT_FUNC
inittiming(void)
{
	PyObject* m;
	
	timing_TimeType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&timing_TimeType) < 0)
	{
		return;
	}
	
	m = Py_InitModule3("timing", timing_Methods, "Module that provides access to clock_gettime() for accurate timing measurements.");
	
	Py_INCREF(&timing_TimeType);
	
	PyImport_AddModule("timing");
	PyModule_AddObject(m, "Time", (PyObject *)&timing_TimeType);
	
	// Add various clock resolutions as constants
	struct timespec t;
	clock_getres(CLOCK_PROCESS_CPUTIME_ID, &t);	
	PyModule_AddObject(m, "CLOCK_RES_PROCESS_CPU", timing_time_to_Time(t));
	clock_getres(CLOCK_THREAD_CPUTIME_ID, &t);	
	PyModule_AddObject(m, "CLOCK_RES_THREAD_CPU", timing_time_to_Time(t));
}

