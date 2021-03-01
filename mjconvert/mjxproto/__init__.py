import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from mjxproto.mjx_pb2_grpc import *
from mjxproto.mjx_pb2 import *