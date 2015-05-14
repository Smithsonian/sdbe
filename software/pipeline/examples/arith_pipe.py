import logging
from numpy import arange, array
from pipeline.core import Args, Steps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('')

def load_args(i):
    logger.info("Starting with a value of {0}".format(i))
    return Args(logger=logger, data=i)

def add_one(args):
    args.logger.info("Adding 1.0 to {0}".format(args.data))
    args.data += 1.0
    return args

def mul_two(args):
    args.logger.info("Multiplying {0} by 2.0".format(args.data))
    args.data *= 2.0
    return args

def unload_args(args):
    args.logger.info("Ending with a value of {0}".format(args.data))
    return args.data

top = Steps(

    load_args,

    add_one,

    Steps(

        add_one,
        mul_two,

        ),

    Steps(

        Steps(

            mul_two,
            mul_two,

            ),

        add_one,

        ),

    unload_args,

)

data_in = arange(10)
top.evaluate(data_in)
data_out = array(list(top))

logger.info("{0} -> {1}".format(data_in, data_out))
assert all(data_out == (((((data_in + 1) + 1) * 2) * 2) * 2) + 1)
