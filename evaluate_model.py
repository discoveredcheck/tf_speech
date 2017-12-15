import tensorflow as tf
import models
import input_data
from argument_parser import create_parser

if __name__ == '__main__':
    parser = create_parser()
    FLAGS, unparsed = parser.parse_known_args()